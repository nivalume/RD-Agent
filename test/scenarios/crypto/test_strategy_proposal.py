"""The crypto strategy proposal builds a thesis and its rule without an LLM call."""

import json
from unittest.mock import patch

import pytest
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.scenarios.crypto.experiment.strategy_experiment import (
    CryptoStrategyExperiment,
    CryptoStrategyScenario,
    StrategyTask,
)
from rdagent.scenarios.crypto.proposal.strategy_proposal import (
    CryptoStrategyHypothesis2Experiment,
    CryptoStrategyHypothesisGen,
    main,
)
from rdagent.utils.agent.tpl import T

pytestmark = pytest.mark.offline

HYPOTHESIS_RESPONSE = {
    "hypothesis": "A leverage cascade overshoots and is given back once the forced selling is done.",
    "reason": "Liquidations hit a thin book, so the price pays for liquidity that is only available after it.",
    "concise_reason": "Cascade overshoot reverts.",
    "concise_observation": "Large down moves are given back within a few bars.",
    "concise_justification": "Forced selling is not information.",
    "concise_knowledge": "Condition a reversal on a displacement, not on a level.",
}

STRATEGY_RESPONSE = {
    "cascade_overshoot_reversal": {
        "description": "[Mean Reversion] Buys a displacement back after a leverage cascade.",
        "phenomenon": "mean_reversion",
        "mechanism": "forced liquidations push the price past fair value in a thin book",
        "return_source": "the liquidated crowd pays the spread on the way out",
        "risk_borne": "a trend day that never reverts",
        "signal": "$$d_t = r_t / \\sigma_t$$",
        "position": "$$w_t = -\\operatorname{clip}(d_t / z, -1, 1)$$",
        "horizon": "enter at the close of bar t, exit after h bars, rebalance every bar",
        "gate": "$$\\sigma_t > q(\\sigma)$$",
        "testable_prediction": "the conditional h-bar forward return is positive and decays by 3h",
        "fails_when": "the cascade is the start of a repricing rather than a liquidity event",
        "parameters": {"z": "entry threshold in standard deviations", "h": "holding period in bars"},
    },
    "implemented_strategy": {
        "description": "[Trend Following] already in the library.",
        "phenomenon": "trend_following",
        "signal": "$$m_t$$",
        "position": "$$w_t = \\operatorname{sign}(m_t)$$",
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
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "crypto" / "data" / "crypto_bar").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def scenario(data_root) -> CryptoStrategyScenario:
    return CryptoStrategyScenario(with_runtime_environment=False, data_root_dir=data_root)


@pytest.fixture
def hypothesis_gen(scenario) -> CryptoStrategyHypothesisGen:
    return CryptoStrategyHypothesisGen(scenario)


@pytest.fixture
def hypothesis(hypothesis_gen) -> Hypothesis:
    return hypothesis_gen.convert_response(json.dumps(HYPOTHESIS_RESPONSE))


def test_gen_is_overridden(hypothesis_gen):
    assert "gen" in CryptoStrategyHypothesisGen.__dict__


def test_convert_response_builds_a_hypothesis(hypothesis):
    assert hypothesis.hypothesis == HYPOTHESIS_RESPONSE["hypothesis"]
    assert hypothesis.reason == HYPOTHESIS_RESPONSE["reason"]
    assert hypothesis.concise_knowledge == HYPOTHESIS_RESPONSE["concise_knowledge"]


def test_prepare_context_on_the_first_round(scenario, hypothesis_gen):
    context, json_flag = hypothesis_gen.prepare_context(Trace(scen=scenario))

    assert json_flag is True
    assert context["hypothesis_and_feedback"].startswith("No previous hypothesis")
    assert "JSON" in context["hypothesis_output_format"]
    assert "One mechanism per Generation" in context["hypothesis_specification"]
    assert "Start with the simplest mechanism" in context["RAG"]


def test_gen_prompts_carry_the_scenario_and_the_trace(scenario, hypothesis_gen):
    trace = Trace(scen=scenario)
    trace.hist.append(
        (
            CryptoStrategyExperiment(
                [
                    StrategyTask(
                        "old_strategy",
                        phenomenon="mean_reversion",
                        signal="$$s_t$$",
                        position="$$w_t = -s_t$$",
                    ),
                ],
                hypothesis=None,
            ),
            HypothesisFeedback(
                reason="works",
                decision=True,
                observations="Sharpe 1.1 pooled over the universe",
                hypothesis_evaluation="supported",
                new_hypothesis="condition the reversal on funding",
                code_change_summary="",
            ),
        ),
    )

    with patch("rdagent.scenarios.crypto.proposal.strategy_proposal.APIBackend") as backend:
        backend.return_value.build_messages_and_create_chat_completion.return_value = json.dumps(
            HYPOTHESIS_RESPONSE,
        )
        generated = hypothesis_gen.gen(trace)

    user_prompt, system_prompt = backend.return_value.build_messages_and_create_chat_completion.call_args.args
    assert "Crypto assets traded on centralized exchanges" in system_prompt
    assert "The clock:" in system_prompt
    assert "market.crypto.bar" in system_prompt
    assert "What every proposal must state" in system_prompt
    assert "Do not name any instrument, token, exchange, country, date, year" in system_prompt
    # The factor path's framing must not leak into a strategy prompt.
    assert "The factor is a characteristic" not in system_prompt
    assert "# Trial 1:" in user_prompt
    assert "Sharpe 1.1 pooled over the universe" in user_prompt
    assert generated.hypothesis == HYPOTHESIS_RESPONSE["hypothesis"]


def test_prepare_context_on_the_first_round_for_the_experiment(scenario, hypothesis):
    context, json_flag = CryptoStrategyHypothesis2Experiment().prepare_context(hypothesis, Trace(scen=scenario))

    assert json_flag is True
    assert context["target_list"] == []
    assert "JSON" in context["experiment_output_format"]
    assert "Hypothesis: A leverage cascade overshoots" in context["target_hypothesis"]


def test_convert_response_keeps_the_proposed_strategies(scenario, hypothesis):
    trace = Trace(scen=scenario)
    experiment = CryptoStrategyHypothesis2Experiment().convert_response(
        json.dumps(STRATEGY_RESPONSE), hypothesis, trace,
    )

    assert isinstance(experiment, CryptoStrategyExperiment)
    assert [task.strategy_name for task in experiment.sub_tasks] == list(STRATEGY_RESPONSE)
    assert len(experiment.based_experiments) == 1
    assert experiment.hypothesis is hypothesis
    assert experiment.sub_tasks[0].phenomenon == "mean_reversion"
    assert experiment.sub_tasks[0].parameters["h"] == "holding period in bars"


def test_convert_response_drops_a_strategy_already_in_the_library(scenario, hypothesis):
    trace = Trace(scen=scenario)
    trace.hist.append(
        (
            CryptoStrategyExperiment(
                [
                    StrategyTask(
                        "implemented_strategy",
                        phenomenon="trend_following",
                        signal="$$m_t$$",
                        position="$$w_t = m_t$$",
                    ),
                ],
                hypothesis=None,
            ),
            HypothesisFeedback(reason="works", decision=True, code_change_summary=""),
        ),
    )

    experiment = CryptoStrategyHypothesis2Experiment().convert_response(
        json.dumps(STRATEGY_RESPONSE), hypothesis, trace,
    )

    assert [task.strategy_name for task in experiment.sub_tasks] == ["cascade_overshoot_reversal"]
    assert len(experiment.based_experiments) == 2


def test_convert_response_normalises_the_phenomenon(scenario, hypothesis):
    response = {
        "name": {
            "description": "[Mean Reversion] x",
            "phenomenon": "Mean-Reversion",
            "signal": "$$s_t$$",
            "position": "$$w_t$$",
        },
    }

    experiment = CryptoStrategyHypothesis2Experiment().convert_response(
        json.dumps(response), hypothesis, Trace(scen=scenario),
    )

    assert experiment.sub_tasks[0].phenomenon == "mean_reversion"


def test_convert_response_rejects_an_unknown_phenomenon(scenario, hypothesis):
    response = {"name": {"phenomenon": "carry", "signal": "$$s_t$$", "position": "$$w_t$$"}}

    with pytest.raises(ValueError, match="not a phenomenon"):
        CryptoStrategyHypothesis2Experiment().convert_response(json.dumps(response), hypothesis, Trace(scen=scenario))


def test_convert_response_rejects_a_rule_without_a_position(scenario, hypothesis):
    response = {"name": {"phenomenon": "mean_reversion", "signal": "$$s_t$$"}}

    with pytest.raises(ValueError, match="has no position"):
        CryptoStrategyHypothesis2Experiment().convert_response(json.dumps(response), hypothesis, Trace(scen=scenario))


def test_convert_response_drops_an_unknown_field(scenario, hypothesis):
    response = {
        "cascade_overshoot_reversal": {
            **STRATEGY_RESPONSE["cascade_overshoot_reversal"],
            "notes": "an invented field",
        },
    }

    experiment = CryptoStrategyHypothesis2Experiment().convert_response(
        json.dumps(response), hypothesis, Trace(scen=scenario),
    )

    assert [task.strategy_name for task in experiment.sub_tasks] == ["cascade_overshoot_reversal"]


def test_the_task_brief_information_carries_the_economic_logic():
    task = StrategyTask(
        "cascade_overshoot_reversal",
        phenomenon="mean_reversion",
        mechanism="forced liquidations",
        return_source="the liquidated crowd pays the spread",
        risk_borne="a trend day that never reverts",
        signal="$$d_t = r_t / \\sigma_t$$",
        position="$$w_t = -d_t$$",
        horizon="exit after h bars",
        testable_prediction="the forward return is positive",
        fails_when="the cascade is a repricing",
        parameters={"h": "holding period in bars"},
    )

    brief = task.get_task_brief_information()

    assert "strategy_name: cascade_overshoot_reversal" in brief
    assert "phenomenon: mean_reversion" in brief
    assert "return_source: the liquidated crowd pays the spread" in brief
    assert "signal: $$d_t = r_t / \\sigma_t$$" in brief
    assert "parameters: " in brief


def test_the_strategy_prompts_name_no_instrument():
    for key in (
        "strategy_hypothesis_gen_system_prompt",
        "strategy_hypothesis_specification",
        "strategy_experiment_output_format",
    ):
        text = T(f"scenarios.crypto.prompts:{key}").r(
            scenario="", hypothesis_output_format="", hypothesis_specification="", user_instruction=None,
        )
        for leaked in ("BTC", "ETH", "Bitcoin", "Binance", "OKX", "Bybit"):
            assert leaked not in text, f"{key} names {leaked}"


def test_main_prints_the_prompt_without_calling_the_model(capsys):
    with patch("rdagent.scenarios.crypto.proposal.strategy_proposal._load_dotenv_and_refresh_settings"):
        assert main(["--print-only"]) == 0

    printed = capsys.readouterr().out
    assert "===== hypothesis system prompt =====" in printed
    assert "===== hypothesis user prompt =====" in printed
    assert "===== experiment system prompt =====" in printed
    assert "===== hypothesis response =====" not in printed


def test_main_prints_the_model_answer(capsys):
    with (
        patch("rdagent.scenarios.crypto.proposal.strategy_proposal._load_dotenv_and_refresh_settings"),
        patch("rdagent.scenarios.crypto.proposal.strategy_proposal.APIBackend") as backend,
    ):
        backend.return_value.build_messages_and_create_chat_completion.side_effect = [
            json.dumps(HYPOTHESIS_RESPONSE),
            json.dumps(STRATEGY_RESPONSE),
        ]
        assert main([]) == 0

    printed = capsys.readouterr().out
    assert "===== hypothesis response =====" in printed
    assert "===== experiment response =====" in printed
    assert "===== strategies =====" in printed
    assert "Hypothesis: A leverage cascade overshoots" in printed
    assert "strategy_name: cascade_overshoot_reversal" in printed


def test_main_reports_a_failed_model_call(capsys):
    with (
        patch("rdagent.scenarios.crypto.proposal.strategy_proposal._load_dotenv_and_refresh_settings"),
        patch("rdagent.scenarios.crypto.proposal.strategy_proposal.APIBackend") as backend,
    ):
        backend.return_value.build_messages_and_create_chat_completion.side_effect = RuntimeError("no key")
        assert main([]) == 1

    printed = capsys.readouterr().out
    assert "===== model call failed =====" in printed
    assert "RuntimeError: no key" in printed
