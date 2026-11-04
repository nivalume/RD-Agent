"""
Factor workflow with session control
"""

import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fire
from rdagent.app.crypto_rd_loop.conf import StrategyBasePropSetting
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, FactorEmptyError
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.scenarios.crypto.conf import CryptoSettings
from rdagent.scenarios.crypto.experiment.factor_experiment import NO_DATA_SCHEMA
from rdagent.scenarios.crypto.proposal.strategy_proposal import (
    CryptoStrategyHypothesis2Experiment,
    CryptoStrategyHypothesisGen,
)


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error("Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: str | None = None,
    base_features_path: str | None = None,
    **kwargs,
):
    """
    Auto R&D Evolving loop for fintech factors.

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter

    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if path is None:
        factor_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        factor_loop = FactorRDLoop.load(path, checkout=checkout)

    factor_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        factor_loop._set_interactor(*kwargs["user_interaction_queues"])
        factor_loop._interact_init_params()
    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


STAGES = ("hypothesis", "experiment")
"""The proposal stages, in the order they run. ``experiment`` is the hypothesis-to-strategy step."""

PLACEHOLDER_HYPOTHESIS_TEXT = "<the mechanism proposed by the hypothesis stage>"


def run(
    stages: Sequence[str] | None = None,
    *,
    print_only: bool = False,
    with_runtime_environment: bool = False,
) -> int:
    """Run the crypto strategy proposal stages, or print their prompts.

    ``stages`` names a subset of :data:`STAGES`. Omitted, both run, hypothesis first. ``print_only``
    renders the prompts and leaves the model uncalled; the experiment prompt then uses a placeholder
    hypothesis. Naming only the experiment stage still runs the hypothesis stage when the model is
    called, because the experiment stage needs its result.
    """
    selected = _stages(stages)
    calls_model = not print_only
    if calls_model and not os.environ.get("OPENAI_API_KEY", ""):
        _print_missing_api_key()
        return 1

    setting, scenario, generator, converter, trace = _prepare(with_runtime_environment=with_runtime_environment)
    _announce(setting, scenario, selected, print_only=print_only)
    hypothesis, ran_hypothesis, failed = _hypothesis_stage(selected, generator, trace, calls_model=calls_model)
    if failed:
        return 1
    experiment, failed = _experiment_stage(selected, converter, trace, hypothesis, calls_model=calls_model)
    if failed or print_only:
        return 1 if failed else 0
    _summarize(
        selected,
        calls_model=calls_model,
        ran_hypothesis=ran_hypothesis,
        hypothesis=hypothesis,
        experiment=experiment,
    )
    return 0


def _prepare(
    *,
    with_runtime_environment: bool,
) -> tuple[
    StrategyBasePropSetting,
    Any,
    CryptoStrategyHypothesisGen,
    CryptoStrategyHypothesis2Experiment,
    Trace,
]:
    setting = StrategyBasePropSetting()
    scenario = import_class(setting.scen)(with_runtime_environment=with_runtime_environment)
    return (
        setting,
        scenario,
        CryptoStrategyHypothesisGen(scenario),
        CryptoStrategyHypothesis2Experiment(),
        Trace(scen=scenario),
    )


def _announce(setting: StrategyBasePropSetting, scenario: Any, selected: Sequence[str], *, print_only: bool) -> None:
    print(f"CHAT_MODEL      = {LLM_SETTINGS.chat_model}")
    print(f"OPENAI_API_BASE = {LLM_SETTINGS.openai_api_base}")
    print(f"SCENARIO        = {setting.scen}")
    print(f"DATA_ROOT       = {CryptoSettings().data_root_dir}")
    print(f"STAGES          = {', '.join(selected)}")
    print(f"PRINT_ONLY      = {print_only}")
    source_data = scenario.get_source_data_desc()
    if NO_DATA_SCHEMA in source_data:
        print("DATA_SCHEMAS    = none; the prompt says no data schema is available")
        return
    print(f"DATA_SCHEMAS    = {source_data.count('### ')} dataset(s)")


def _hypothesis_stage(
    selected: Sequence[str],
    generator: CryptoStrategyHypothesisGen,
    trace: Trace,
    *,
    calls_model: bool,
) -> tuple[Hypothesis | None, bool, bool]:
    """Return the hypothesis, whether the stage ran, and whether the model call failed."""
    forced = "experiment" in selected and calls_model
    if "hypothesis" not in selected and not forced:
        return None, False, False
    if "hypothesis" not in selected:
        print("The experiment stage needs a hypothesis, so the hypothesis stage runs first.")
    context, _json_flag = generator.prepare_context(trace)
    _print_prompts("hypothesis", *generator.render_prompts(context))
    if not calls_model:
        return None, True, False
    try:
        hypothesis = generator.gen(trace)
    except Exception as error:  # noqa: BLE001
        _print_stage_error("hypothesis", error)
        return None, True, True
    _print_hypothesis(hypothesis)
    return hypothesis, True, False


def _experiment_stage(
    selected: Sequence[str],
    converter: CryptoStrategyHypothesis2Experiment,
    trace: Trace,
    hypothesis: Hypothesis | None,
    *,
    calls_model: bool,
) -> tuple[Any, bool]:
    """Return the experiment and whether the model call failed."""
    if "experiment" not in selected:
        return None, False
    if hypothesis is None:
        print("The model was not called, so the experiment prompt uses a placeholder hypothesis.")
        hypothesis = _placeholder_hypothesis()
    _print_prompts("experiment", *converter.render_prompts(hypothesis, trace))
    if not calls_model:
        return None, False
    try:
        experiment = converter.convert(hypothesis, trace)
    except Exception as error:  # noqa: BLE001
        _print_stage_error("experiment", error)
        return None, True
    _print_strategies(experiment)
    return experiment, False


def _placeholder_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis=PLACEHOLDER_HYPOTHESIS_TEXT,
        reason="",
        concise_reason="",
        concise_observation="",
        concise_justification="",
        concise_knowledge="",
    )


def _summarize(
    selected: Sequence[str],
    *,
    calls_model: bool,
    ran_hypothesis: bool,
    hypothesis: Hypothesis | None,
    experiment: Any,
) -> None:
    print("\n===== summary =====")
    generated = hypothesis is not None and hypothesis.hypothesis != PLACEHOLDER_HYPOTHESIS_TEXT
    if ran_hypothesis and generated:
        print("hypothesis: generated")
    if "experiment" in selected and calls_model:
        print(f"strategies: {len(experiment.sub_tasks)}")


def _print_missing_api_key() -> None:
    print(
        "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell that launches this script; "
        ".env maps ARK_API_KEY to OPENAI_API_KEY.",
        file=sys.stderr,
    )


def _print_stage_error(stage: str, error: Exception) -> None:
    print(f"\n===== {stage} call failed =====\n{type(error).__name__}: {error}", file=sys.stderr)


def _stages(stages: Sequence[str] | None) -> list[str]:
    """The requested stages in run order."""
    requested = list(STAGES if stages is None else stages)
    unknown = [name for name in requested if name not in STAGES]
    if not unknown:
        return [name for name in STAGES if name in requested]
    known = ", ".join(STAGES)
    message = f"unknown stage {unknown[0]!r}; choose from {known}"
    raise ValueError(message)


def _print_prompts(stage: str, system_prompt: str, user_prompt: str) -> None:
    print(f"\n===== {stage} system prompt =====")
    print(system_prompt)
    print(f"\n===== {stage} user prompt =====")
    print(user_prompt)


def _print_hypothesis(hypothesis: Any) -> None:
    print("\n===== hypothesis =====")
    print(hypothesis.hypothesis)
    print(f"\nreason:\n{hypothesis.reason}")
    for field in (
        "concise_reason",
        "concise_observation",
        "concise_justification",
        "concise_knowledge",
    ):
        value = getattr(hypothesis, field, None)
        if value:
            print(f"\n{field}:\n{value}")


def _print_strategies(experiment: Any) -> None:
    print("\n===== strategies =====")
    for task in experiment.sub_tasks:
        print()
        print(task.get_task_brief_information())


if __name__ == "__main__":
    fire.Fire(main)
