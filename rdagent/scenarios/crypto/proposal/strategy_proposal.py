"""The crypto strategy proposal: an economic thesis, and the tradeable rule it turns into.

The crypto scenario used to propose factors. A factor is ranked against the other instruments at the same
time, and crypto's cross-section is dominated by one common market factor: the instruments mostly rise and
fall together, so what is left after the common factor is largely microstructure -- thin books, stop runs,
manipulation -- and a single factor's cross-sectional IC is a weak, unstable statistic. Whatever ordering
a factor does find tends to live in a handful of the most liquid instruments, which is why the factors
that came out were never evaluable across a cross-section.

This module proposes **strategies** instead. A strategy reads one instrument's own bars (and its funding)
and produces one PnL path for that instrument; the same rule with the same parameters runs on every
instrument and the paths are pooled. The mechanism is a property of the market -- forced liquidations,
crowded leverage, thin liquidity -- so it is not tied to a particular instrument, and it is a statement
about a conditional distribution rather than a ranking.

The two phenomena in scope are mean reversion and trend following. Everything the model must say about
them is economic: which friction produces the return, who is on the other side, why the transfer is not
arbitraged away, what risk the strategy bears, and what the data should show if the mechanism is real.
The rest of this module exists to keep the model honest about all of that:

* the prompt withholds every identifier -- no instrument, venue or year -- because a model that knows the
  name also knows how that name turned out, and then writes a rule that happens to fit the past;
* time is counted in bars and every parameter is dimensionless, so a rule cannot encode a price level or
  a remembered regime;
* :meth:`CryptoStrategyHypothesis2Experiment.convert_response` refuses a rule that names a phenomenon
  outside the two in scope or arrives without a signal and a position, and ``convert`` retries the model
  instead of accepting it.

Render the prompts and call the model::

    python -m rdagent.scenarios.crypto.proposal.strategy_proposal
    python -m rdagent.scenarios.crypto.proposal.strategy_proposal --print-only
"""

from __future__ import annotations

import argparse
import json

from rdagent.components.proposal import LLMHypothesis2Experiment, LLMHypothesisGen
from rdagent.core.proposal import ExperimentPlan, Hypothesis, Scenario, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.crypto.experiment.strategy_experiment import (
    SPECIFICATION_FIELDS,
    CryptoStrategyExperiment,
    CryptoStrategyScenario,
    StrategyTask,
)
from rdagent.scenarios.crypto.proposal.factor_proposal import _load_dotenv_and_refresh_settings
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry

CRYPTO_PROMPTS = "scenarios.crypto.prompts"
"""The crypto scenario's shared prompts, next to this scenario's package."""

HYPOTHESIS_GEN_PROMPTS = "components.proposal.prompts"
"""The trace-rendering prompts shared by every scenario.

A ``.prompts:`` uri resolves next to the module that uses it, so the absolute uri is used here to read
the same templates as the other scenarios from inside the crypto package.
"""


class CryptoStrategyHypothesisGen(LLMHypothesisGen):
    """Propose one economic mechanism and the tradeable rule it implies."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self.targets = "trading strategies"

    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        """Generate the next hypothesis; ``render_prompts`` is shared with the CLI."""
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt, user_prompt = self.render_prompts(context_dict, plan)

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str],
        )

        return self.convert_response(resp)

    def render_prompts(self, context_dict: dict, plan: ExperimentPlan | None = None) -> tuple[str, str]:
        """The (system, user) prompts of hypothesis generation; shared by ``gen`` and the CLI.

        The system prompt is this scenario's own: the economic gate, the two phenomena in scope and the
        no-hindsight rules are the point of the strategy proposal, and the generic hypothesis prompt is
        written for factors. The user prompt, which renders the trace and the feedback, is the shared one.
        """
        system_prompt = T(f"{CRYPTO_PROMPTS}:strategy_hypothesis_gen_system_prompt").r(
            scenario=self.scen.get_scenario_all_desc(filtered_tag=self.targets),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
            user_instruction=plan.get("user_instruction", None) if plan is not None else None,
        )
        user_prompt = T(f"{HYPOTHESIS_GEN_PROMPTS}:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=context_dict.get("last_hypothesis_and_feedback", ""),
            sota_hypothesis_and_feedback=context_dict.get("sota_hypothesis_and_feedback", ""),
            RAG=context_dict["RAG"],
        )
        return system_prompt, user_prompt

    def prepare_context(self, trace: Trace) -> tuple[dict, bool]:
        hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:hypothesis_and_feedback").r(trace=trace)
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1],
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Start with the simplest mechanism the phenomenon implies: one state variable, one holding "
                "period, one bounded position"
                if len(trace.hist) < 15
                else "Now vary the conditioning: ask when the same mechanism must be flat, how its return "
                "decays with the holding period, and what the funding rate says about who is crowded"
            ),
            "hypothesis_output_format": T(f"{CRYPTO_PROMPTS}:strategy_hypothesis_output_format").r(),
            "hypothesis_specification": T(f"{CRYPTO_PROMPTS}:strategy_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        return Hypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )


class CryptoStrategyHypothesis2Experiment(LLMHypothesis2Experiment):
    """Turn the mechanism into the LaTeX rule a backtest can run."""

    def __init__(self) -> None:
        super().__init__()
        self.targets = "trading strategies"

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> tuple[dict, bool]:
        hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:hypothesis_and_feedback").r(trace=trace)
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        return {
            "target_hypothesis": str(hypothesis),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": T(f"{CRYPTO_PROMPTS}:strategy_experiment_output_format").r(),
            "target_list": [],
            "RAG": None,
        }, True

    def render_prompts(self, hypothesis: Hypothesis, trace: Trace) -> tuple[str, str]:
        """The (system, user) prompts of the conversion; shared by ``convert`` and the CLI."""
        context, _ = self.prepare_context(hypothesis, trace)
        system_prompt = T(f"{HYPOTHESIS_GEN_PROMPTS}:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(f"{HYPOTHESIS_GEN_PROMPTS}:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=context["hypothesis_and_feedback"],
            last_hypothesis_and_feedback="",
            sota_hypothesis_and_feedback="",
            target_list=context["target_list"],
            RAG=context["RAG"],
        )
        return system_prompt, user_prompt

    @wait_retry(retry_n=5)
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> CryptoStrategyExperiment:
        """Ask the model for the rule, retrying on a response that is not a runnable strategy."""
        system_prompt, user_prompt = self.render_prompts(hypothesis, trace)
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt,
            system_prompt,
            json_mode=True,
            json_target_type=dict[str, dict[str, str | dict]],
        )
        return self.convert_response(resp, hypothesis, trace)

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> CryptoStrategyExperiment:
        response_dict = json.loads(response)

        # Every strategy proposed in an earlier round is already in the library; proposing it again would
        # only re-derive it.
        based_experiments = [CryptoStrategyExperiment(sub_tasks=[])] + [
            experiment
            for experiment, feedback in trace.hist
            if feedback and isinstance(experiment, CryptoStrategyExperiment)
        ]
        implemented = {task.strategy_name for experiment in based_experiments for task in experiment.sub_tasks}

        tasks = []
        for strategy_name, specification in response_dict.items():
            if strategy_name in implemented:
                continue
            unknown = sorted(set(specification) - set(SPECIFICATION_FIELDS))
            if unknown:
                # A stray key is not worth failing the round over, but it usually means the model invented
                # a field instead of using the one it was asked for, so it is logged.
                logger.warning(f"The strategy {strategy_name!r} carries unknown fields {unknown}; they are dropped.")
            tasks.append(
                StrategyTask(
                    strategy_name=strategy_name,
                    # `description` and `parameters` are passed explicitly: the first is a positional
                    # argument of Task, the second is coerced because the model does not always return a dict.
                    **{
                        name: specification.get(name, "")
                        for name in SPECIFICATION_FIELDS
                        if name not in ("description", "parameters")
                    },
                    description=specification.get("description", ""),
                    parameters=_parameters_of(specification.get("parameters")),
                ),
            )

        experiment = CryptoStrategyExperiment(tasks, hypothesis=hypothesis)
        experiment.based_experiments = based_experiments
        return experiment


def _parameters_of(value: object) -> dict[str, str]:
    """The parameter declarations, whatever shape the model returned them in."""
    if isinstance(value, dict):
        return {str(name): str(meaning) for name, meaning in value.items()}
    return {}


def main(argv: list[str] | None = None) -> int:
    """Render both stages' prompts, call the model, and print what it answers."""
    parser = argparse.ArgumentParser(
        prog="python -m rdagent.scenarios.crypto.proposal.strategy_proposal",
        description="Render the crypto strategy prompts, call the model, and print its answer.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="render and print the prompts without calling the model",
    )
    parser.add_argument(
        "--runtime-environment",
        action="store_true",
        help=(
            "execute the factor environment to list its packages; off by default so the prompt renders "
            "without a configured environment"
        ),
    )
    args = parser.parse_args(argv)

    _load_dotenv_and_refresh_settings()

    scenario = CryptoStrategyScenario(with_runtime_environment=args.runtime_environment)
    trace = Trace(scen=scenario)
    generator = CryptoStrategyHypothesisGen(scenario)
    converter = CryptoStrategyHypothesis2Experiment()

    context_dict, json_flag = generator.prepare_context(trace)
    system_prompt, user_prompt = generator.render_prompts(context_dict)

    print(f"model: {LLM_SETTINGS.chat_model}")
    print(f"base: {LLM_SETTINGS.openai_api_base or '<the provider default>'}")
    print("\n===== hypothesis system prompt =====")
    print(system_prompt)
    print("\n===== hypothesis user prompt =====")
    print(user_prompt)

    if args.print_only:
        # Stage 2 is rendered against a placeholder mechanism so its shape is visible without a model call.
        placeholder = Hypothesis(
            hypothesis="<the mechanism proposed by the hypothesis stage>",
            reason="",
            concise_reason="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )
        _print_experiment_prompts(converter, placeholder, trace)
        return 0

    try:
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str],
        )
    except Exception as error:  # the CLI reports the failure instead of a traceback
        print(f"\n===== model call failed =====\n{type(error).__name__}: {error}")
        return 1

    print("\n===== hypothesis response =====")
    print(response)

    try:
        hypothesis = generator.convert_response(response)
    except json.JSONDecodeError as error:
        print(f"Could not parse the response as JSON: {error}")
        return 1

    print("\n===== hypothesis =====")
    print(hypothesis)

    hypothesis_system_prompt, hypothesis_user_prompt = converter.render_prompts(hypothesis, trace)
    print("\n===== experiment system prompt =====")
    print(hypothesis_system_prompt)
    print("\n===== experiment user prompt =====")
    print(hypothesis_user_prompt)

    try:
        response = APIBackend().build_messages_and_create_chat_completion(
            hypothesis_user_prompt,
            hypothesis_system_prompt,
            json_mode=True,
            json_target_type=dict[str, dict[str, str | dict]],
        )
    except Exception as error:
        print(f"\n===== model call failed =====\n{type(error).__name__}: {error}")
        return 1

    print("\n===== experiment response =====")
    print(response)

    try:
        experiment = converter.convert_response(response, hypothesis, trace)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Could not turn the response into a strategy: {error}")
        return 1

    print("\n===== strategies =====")
    for task in experiment.sub_tasks:
        print()
        print(task.get_task_brief_information())
    return 0


def _print_experiment_prompts(
    converter: CryptoStrategyHypothesis2Experiment, hypothesis: Hypothesis, trace: Trace,
) -> None:
    system_prompt, user_prompt = converter.render_prompts(hypothesis, trace)
    print("\n===== experiment system prompt =====")
    print(system_prompt)
    print("\n===== experiment user prompt =====")
    print(user_prompt)


if __name__ == "__main__":
    raise SystemExit(main())
