"""The crypto factor proposal: hypothesis generation and hypothesis to experiment conversion.

The crypto scenario is factor-only. ``gen`` is overridden here: it keeps the implementation of
:class:`LLMHypothesisGen`, but living in this module lets the crypto flow diverge from the shared one
without touching the other scenarios.

Render the hypothesis prompt and call the model to print its answer::

    python -m rdagent.scenarios.crypto.proposal.factor_proposal
    python -m rdagent.scenarios.crypto.proposal.factor_proposal --print-only
"""

import argparse
import json
import os
from typing import Tuple

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import ExperimentPlan, Hypothesis, Scenario, Trace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.crypto.experiment.factor_experiment import CryptoFactorExperiment, CryptoFactorScenario
from rdagent.utils.agent.tpl import T

CRYPTO_PROMPTS = "scenarios.crypto.prompts"
"""The crypto scenario's shared prompts, next to this scenario's package."""

HYPOTHESIS_GEN_PROMPTS = "components.proposal.prompts"
"""The hypothesis generation prompts shared by every scenario.

A ``.prompts:`` uri resolves next to the module that uses it, so the absolute uri is used here to
read the same templates as :meth:`LLMHypothesisGen.gen` from inside the crypto package.
"""


class CryptoFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)

    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        """Generate the next hypothesis; the same implementation as ``LLMHypothesisGen.gen``."""
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt, user_prompt = self.render_prompts(context_dict, plan)

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
        )

        return self.convert_response(resp)

    def render_prompts(self, context_dict: dict, plan: ExperimentPlan | None = None) -> Tuple[str, str]:
        """The (system, user) prompts of hypothesis generation; shared by ``gen`` and the CLI."""
        system_prompt = T(f"{HYPOTHESIS_GEN_PROMPTS}:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=(
                self.scen.get_scenario_all_desc(filtered_tag=self.targets)
                if self.targets in ["factor", "model"]
                else self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
            ),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
            user_instruction=plan.get("user_instruction", None) if plan is not None else None,
        )
        user_prompt = T(f"{HYPOTHESIS_GEN_PROMPTS}:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=(
                context_dict["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context_dict else ""
            ),
            sota_hypothesis_and_feedback=(
                context_dict["sota_hypothesis_and_feedback"] if "sota_hypothesis_and_feedback" in context_dict else ""
            ),
            RAG=context_dict["RAG"],
        )
        return system_prompt, user_prompt

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:hypothesis_and_feedback").r(trace=trace)
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Start with simple price-volume factors that are quick to implement and test"
                if len(trace.hist) < 15
                else "Now explore factors with stronger cross-sectional or microstructure mechanisms, and condition them on the calendar (session, funding and expiry timing)"
            ),
            "hypothesis_output_format": T(f"{CRYPTO_PROMPTS}:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T(f"{CRYPTO_PROMPTS}:factor_hypothesis_specification").r(),
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


class CryptoFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            T(f"{CRYPTO_PROMPTS}:hypothesis_and_feedback").r(trace=trace)
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        return {
            "target_hypothesis": str(hypothesis),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": T(f"{CRYPTO_PROMPTS}:factor_experiment_output_format").r(),
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)

        # Every factor implemented in an earlier successful round is already in the library; proposing
        # it again would only re-implement it.
        based_experiments = [CryptoFactorExperiment(sub_tasks=[])] + [
            experiment
            for experiment, feedback in trace.hist
            if feedback and isinstance(experiment, CryptoFactorExperiment)
        ]
        implemented = {task.factor_name for experiment in based_experiments for task in experiment.sub_tasks}

        tasks = [
            FactorTask(
                factor_name=factor_name,
                factor_description=factor_information["description"],
                factor_formulation=factor_information["formulation"],
                variables=factor_information.get("variables", {}),
            )
            for factor_name, factor_information in response_dict.items()
            if factor_name not in implemented
        ]

        experiment = CryptoFactorExperiment(tasks, hypothesis=hypothesis)
        experiment.based_experiments = based_experiments
        return experiment


def _load_dotenv_and_refresh_settings() -> None:
    """Load `.env` and refresh the LLM settings that were built when the package was imported.

    `python -m` imports this module -- and with it `rdagent.oai.llm_conf` -- before `main` runs, so a
    `load_dotenv` alone would come too late for the settings. The instances are updated in place
    because the backend modules hold references to them.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if not load_dotenv(".env"):
        load_dotenv()

    from rdagent.oai.llm_conf import LLMSettings

    # Only the fields the environment actually sets are copied; pydantic deprecates a few of the
    # untouched ones and would warn on assignment.
    fresh = LLMSettings()
    for field in LLMSettings.model_fields:
        if os.environ.get(field.upper()) is not None:
            setattr(LLM_SETTINGS, field, getattr(fresh, field))

    try:
        from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMSettings
    except ImportError:
        return
    fresh = LiteLLMSettings()
    for field in LiteLLMSettings.model_fields:
        if os.environ.get(field.upper()) is not None:
            setattr(LITELLM_SETTINGS, field, getattr(fresh, field))


def main(argv: list[str] | None = None) -> int:
    """Print the hypothesis prompts, call the model, and print what it answers."""
    parser = argparse.ArgumentParser(
        prog="python -m rdagent.scenarios.crypto.proposal.factor_proposal",
        description="Render the crypto hypothesis prompt, call the model, and print its answer.",
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

    scenario = CryptoFactorScenario(with_runtime_environment=args.runtime_environment)
    generator = CryptoFactorHypothesisGen(scenario)
    context_dict, json_flag = generator.prepare_context(Trace(scen=scenario))
    system_prompt, user_prompt = generator.render_prompts(context_dict)

    print(f"model: {LLM_SETTINGS.chat_model}")
    print(f"base: {LLM_SETTINGS.openai_api_base or '<the provider default>'}")
    print(
        f"api key: {'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING (export ARK_API_KEY or OPENAI_API_KEY)'}"
    )
    print("\n===== system prompt =====")
    print(system_prompt)
    print("\n===== user prompt =====")
    print(user_prompt)

    if args.print_only:
        return 0

    try:
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
        )
    except Exception as error:  # the CLI reports the failure instead of a traceback
        print(f"\n===== model call failed =====\n{type(error).__name__}: {error}")
        return 1

    print("\n===== model response =====")
    print(response)

    print("\n===== hypothesis =====")
    try:
        hypothesis = generator.convert_response(response)
    except json.JSONDecodeError as error:
        print(f"Could not parse the response as JSON: {error}")
        return 1
    print(hypothesis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
