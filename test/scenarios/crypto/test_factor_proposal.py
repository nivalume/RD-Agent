"""The crypto factor proposal builds hypotheses and experiments without an LLM call."""

import json
from unittest.mock import patch

import pytest

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.scenarios.crypto.experiment.factor_experiment import (
    CryptoFactorExperiment,
    CryptoFactorScenario,
)
from rdagent.scenarios.crypto.proposal.factor_proposal import (
    CryptoFactorHypothesis2Experiment,
    CryptoFactorHypothesisGen,
    main,
)

pytestmark = pytest.mark.offline

HYPOTHESIS_RESPONSE = {
    "hypothesis": "Momentum in the last 24 bars reverses on the weekend.",
    "reason": "Retail leverage is thinner when traditional desks are closed.",
    "concise_reason": "Weekend reversal.",
    "concise_observation": "Reversals cluster on weekends.",
    "concise_justification": "Thin liquidity overshoots.",
    "concise_knowledge": "Condition momentum signals on the weekend.",
}

EXPERIMENT_RESPONSE = {
    "weekend_reversal": {
        "description": "[Reversal Factor] 24 bar return reversed on weekends.",
        "formulation": "$$-r_{24} \\cdot 1_{weekend}$$",
        "variables": {"r_{24}": "24 bar return"},
    },
    "implemented_factor": {
        "description": "[Momentum Factor] already in the library.",
        "formulation": "$$m$$",
        "variables": {},
    },
}


@pytest.fixture
def data_root(tmp_path):
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "market.crypto.bar.v1.json").write_text(
        json.dumps(
            {
                "slug": "market.crypto.bar",
                "version": 1,
                "name": "CryptoBar",
                "description": "Crypto bar history.",
                "columns": [{"name": "close", "type": "decimal", "precision": 38, "scale": 18}],
                "annotations": {"fields": {"close": {"description": "Last traded price in the bar."}}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "crypto" / "data" / "crypto_bar").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def scenario(data_root) -> CryptoFactorScenario:
    return CryptoFactorScenario(with_runtime_environment=False, data_root_dir=data_root)


@pytest.fixture
def hypothesis_gen(scenario) -> CryptoFactorHypothesisGen:
    return CryptoFactorHypothesisGen(scenario)


@pytest.fixture
def hypothesis(hypothesis_gen) -> Hypothesis:
    return hypothesis_gen.convert_response(json.dumps(HYPOTHESIS_RESPONSE))


def test_gen_is_overridden(hypothesis_gen):
    assert "gen" in CryptoFactorHypothesisGen.__dict__


def test_convert_response_builds_a_hypothesis(hypothesis):
    assert hypothesis.hypothesis == HYPOTHESIS_RESPONSE["hypothesis"]
    assert hypothesis.reason == HYPOTHESIS_RESPONSE["reason"]
    assert hypothesis.concise_knowledge == HYPOTHESIS_RESPONSE["concise_knowledge"]


def test_prepare_context_on_the_first_round(scenario, hypothesis_gen):
    context, json_flag = hypothesis_gen.prepare_context(Trace(scen=scenario))

    assert json_flag is True
    assert context["hypothesis_and_feedback"].startswith("No previous hypothesis")
    assert "JSON" in context["hypothesis_output_format"]
    assert "1-5 Factors per Generation" in context["hypothesis_specification"]
    assert "simple price-volume factors" in context["RAG"]


def test_gen_prompts_carry_the_scenario_and_the_trace(scenario, hypothesis_gen):
    trace = Trace(scen=scenario)
    trace.hist.append(
        (
            CryptoFactorExperiment([FactorTask("old_factor", "old", "$$m$$")], hypothesis=None),
            HypothesisFeedback(
                reason="works",
                decision=True,
                observations="IC 0.03",
                hypothesis_evaluation="supported",
                new_hypothesis="try a session-conditioned variant",
                code_change_summary="",
            ),
        )
    )

    with patch("rdagent.scenarios.crypto.proposal.factor_proposal.APIBackend") as backend:
        backend.return_value.build_messages_and_create_chat_completion.return_value = json.dumps(
            HYPOTHESIS_RESPONSE
        )
        generated = hypothesis_gen.gen(trace)

    user_prompt, system_prompt = (
        backend.return_value.build_messages_and_create_chat_completion.call_args.args
    )
    assert "Crypto assets traded on centralized exchanges" in system_prompt
    assert "The clock:" in system_prompt
    assert "The source data you can use:" in system_prompt
    assert "market.crypto.bar" in system_prompt
    assert "# Trial 1:" in user_prompt
    assert "IC 0.03" in user_prompt
    assert generated.hypothesis == HYPOTHESIS_RESPONSE["hypothesis"]


def test_prepare_context_on_the_first_round_for_the_experiment(scenario, hypothesis):
    context, json_flag = CryptoFactorHypothesis2Experiment().prepare_context(
        hypothesis, Trace(scen=scenario)
    )

    assert json_flag is True
    assert context["target_list"] == []
    assert "JSON" in context["experiment_output_format"]
    assert "reverses on the weekend" in context["target_hypothesis"]


def test_convert_response_keeps_the_proposed_factors(scenario, hypothesis):
    trace = Trace(scen=scenario)
    experiment = CryptoFactorHypothesis2Experiment().convert_response(
        json.dumps(EXPERIMENT_RESPONSE), hypothesis, trace
    )

    assert isinstance(experiment, CryptoFactorExperiment)
    assert [task.factor_name for task in experiment.sub_tasks] == list(EXPERIMENT_RESPONSE)
    assert len(experiment.based_experiments) == 1
    assert experiment.hypothesis is hypothesis


def test_convert_response_drops_a_factor_already_in_the_library(scenario, hypothesis):
    trace = Trace(scen=scenario)
    trace.hist.append(
        (
            CryptoFactorExperiment([FactorTask("implemented_factor", "old", "$$m$$")], hypothesis=None),
            HypothesisFeedback(reason="works", decision=True, code_change_summary=""),
        )
    )

    experiment = CryptoFactorHypothesis2Experiment().convert_response(
        json.dumps(EXPERIMENT_RESPONSE), hypothesis, trace
    )

    assert [task.factor_name for task in experiment.sub_tasks] == ["weekend_reversal"]
    assert len(experiment.based_experiments) == 2


def test_render_prompts_carries_the_scenario(scenario, hypothesis_gen):
    context, _ = hypothesis_gen.prepare_context(Trace(scen=scenario))

    system_prompt, user_prompt = hypothesis_gen.render_prompts(context)

    assert "Crypto assets traded on centralized exchanges" in system_prompt
    assert "The clock:" in system_prompt
    assert "No previous hypothesis and feedback available" in user_prompt


def test_main_prints_the_prompt_without_calling_the_model(capsys):
    with patch("rdagent.scenarios.crypto.proposal.factor_proposal._load_dotenv_and_refresh_settings"):
        assert main(["--print-only"]) == 0

    printed = capsys.readouterr().out
    assert "===== system prompt =====" in printed
    assert "===== user prompt =====" in printed
    assert "===== model response =====" not in printed


def test_main_prints_the_model_answer(capsys):
    with (
        patch("rdagent.scenarios.crypto.proposal.factor_proposal._load_dotenv_and_refresh_settings"),
        patch("rdagent.scenarios.crypto.proposal.factor_proposal.APIBackend") as backend,
    ):
        backend.return_value.build_messages_and_create_chat_completion.return_value = json.dumps(
            HYPOTHESIS_RESPONSE
        )
        assert main([]) == 0

    printed = capsys.readouterr().out
    assert "===== model response =====" in printed
    assert "===== hypothesis =====" in printed
    assert "Hypothesis: Momentum in the last 24 bars reverses on the weekend." in printed


def test_main_reports_a_failed_model_call(capsys):
    with (
        patch("rdagent.scenarios.crypto.proposal.factor_proposal._load_dotenv_and_refresh_settings"),
        patch("rdagent.scenarios.crypto.proposal.factor_proposal.APIBackend") as backend,
    ):
        backend.return_value.build_messages_and_create_chat_completion.side_effect = RuntimeError("no key")
        assert main([]) == 1

    printed = capsys.readouterr().out
    assert "===== model call failed =====" in printed
    assert "RuntimeError: no key" in printed

