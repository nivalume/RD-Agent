"""The strategy runner, the gates, the feedback and the report, on a synthetic panel.

The backtests run for real, in child processes; the chat backend is stubbed. The point of most
tests is what the model is shown: never a test-segment number while ``hide_lockbox`` is on.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.phasor_coder.spec import validate_factor_spec
from rdagent.components.coder.strategy_coder import StrategyWorkspace
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.core.scenario import Scenario
from rdagent.log.storage import FileStorage
from rdagent.scenarios.crypto.experiment.strategy_experiment import CryptoStrategyExperiment, StrategyTask
from rdagent.scenarios.crypto.strategy import engine, report
from rdagent.scenarios.crypto.strategy import results as R
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.developer import StrategyExperiment2Feedback, StrategyRunner

pytestmark = pytest.mark.offline

ROWS = 1200  # 200 days of 4h bars from 2024-01-01


# Knows the move of the eight bars it will hold: the runner does not look for look-ahead (the coder
# does), so here it stands for a strategy that passes every gate, one more bar of delay included.
def _oracle(columns: str = ":") -> dict:
    """Knows the move of the eight bars it will hold, on the given columns: the runner does not look
    for look-ahead (the coder does), so here it stands for a strategy that passes every gate, one
    more bar of delay included."""
    return {
        "mode": "python",
        "code": (
            "import numpy as np\n\n"
            "def calculate(fields, index, symbols):\n"
            "    close = fields['close']\n"
            "    out = np.zeros_like(close)\n"
            f"    out[:-9, {columns}] = np.sign(close[9:, {columns}] / close[1:-8, {columns}] - 1)\n"
            "    return out\n"
        ),
        "fields": ["close"],
        "reason": "fixture",
    }


ORACLE = _oracle()
TREND = {
    "mode": "expression",
    "expression": "Sign(Mean($close, {n}) - Mean($close, 4*{n}))",
    "params": {"n": [3, 6]},
    "fields": ["close"],
    "reason": "fixture",
}


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "crypto perpetuals"

    @property
    def rich_style_description(self) -> str:
        return ""

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:
        return "crypto perpetuals, 4h bars"

    def get_runtime_environment(self) -> str:
        return ""


@pytest.fixture
def panel(tmp_path, monkeypatch):
    generator = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(generator.normal(0, 0.01, size=(ROWS, 3)), axis=0))
    times = pd.date_range("2024-01-01 04:00", periods=ROWS, freq="4h", tz="UTC").rename("time")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    directory = tmp_path / "panel"
    directory.mkdir()
    for name in engine.FIELDS:
        values = np.zeros_like(close) if name == "funding_rate" else close
        pd.DataFrame(values, index=times, columns=symbols).to_parquet(directory / f"{name}.parquet")
    pd.DataFrame(True, index=times, columns=symbols).to_parquet(directory / "pool.parquet")
    for name, value in {
        "PANEL_DIR": str(directory),
        "PYTHON_BIN": sys.executable,
        "VOL_LOOKBACK": "30",
        "TRAIN_START": "2024-01-01",
        "TRAIN_END": "2024-03-31",
        "SELECT_START": "2024-04-01",
        "SELECT_END": "2024-05-31",
        "LOCKBOX_START": "2024-06-01",
    }.items():
        monkeypatch.setenv(f"CRYPTO_STRATEGY_{name}", value)
    return directory


def _task(name: str) -> StrategyTask:
    return StrategyTask(strategy_name=name, phenomenon="trend_following", signal="s", position="p")


def _experiment(**specs) -> CryptoStrategyExperiment:
    exp = CryptoStrategyExperiment([_task(name) for name in specs], hypothesis=Hypothesis("h", "r", "", "", "", ""))
    for index, spec in enumerate(specs.values()):
        workspace = StrategyWorkspace(target_task=exp.sub_tasks[index])
        workspace.inject_files(**{"strategy.json": json.dumps(spec)})
        exp.sub_workspace_list[index] = workspace
    return exp


class _Backend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        return self

    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, **kwargs) -> str:
        self.calls.append({"user": user_prompt, "system": system_prompt})
        return json.dumps(
            {"observations": "o", "hypothesis_evaluation": "e", "new_hypothesis": "n", "reason": "because"}
        )


@pytest.fixture
def backend(monkeypatch):
    stub = _Backend()
    monkeypatch.setattr("rdagent.scenarios.crypto.strategy.developer.APIBackend", stub)
    return stub


@pytest.fixture
def ran(panel):
    exp = _experiment(oracle=ORACLE, trend=TREND)
    exp.prior_trial_sharpes = [0.01, -0.02]
    return StrategyRunner(_Scenario()).develop(exp)


def _test_numbers(exp) -> list[str]:
    """The test-segment numbers of every strategy, as the prompts would print them."""
    found = []
    for result in exp.strategy_results.values():
        for case in result["cases"].values():
            for name in ("sharpe", "annual_return", "alpha_sharpe"):
                found.append(f"{round(case['lockbox'][name], 3)}")
    return found


# -- the runner --------------------------------------------------------------------------------


def test_the_runner_backtests_every_strategy_and_hides_the_test_segment(ran):
    assert set(ran.strategy_results) == {"oracle", "trend"}
    assert all(result["ok"] for result in ran.strategy_results.values())
    assert set(ran.strategy_results["trend"]["cases"]["n=3"]) == {"train", "select", "lockbox"}
    assert ran.strategy_results["trend"]["robustness"]["trials"] == 2 + 1 + 2  # earlier, oracle, own cases
    assert len(ran.trial_sharpes) == 3
    shown = json.dumps(ran.result)
    assert '"lockbox"' not in shown
    assert not any(f": {number}," in shown for number in _test_numbers(ran))
    assert set(ran.result["oracle"]) >= {"case", "train", "select", "gates_failed"}
    assert (ran.sub_workspace_list[0].workspace_path / "returns.parquet").exists()


def test_a_strategy_the_coder_rejected_is_not_backtested(panel):
    exp = _experiment(oracle=ORACLE, broken=TREND)

    class _Feedback:
        def __init__(self, decision):
            self.final_decision = decision

    exp.prop_dev_feedback = [_Feedback(True), _Feedback(False)]
    exp = StrategyRunner(_Scenario()).develop(exp)
    assert exp.strategy_results["broken"]["ok"] is False
    assert exp.result["broken"]["ran"] is False


def test_the_test_segment_is_shown_when_it_is_not_hidden(panel, monkeypatch):
    monkeypatch.setenv("CRYPTO_STRATEGY_HIDE_TEST", "false")
    exp = StrategyRunner(_Scenario()).develop(_experiment(oracle=ORACLE))
    assert "lockbox" in exp.result["oracle"]


# -- the gates ---------------------------------------------------------------------------------


def _result(select=1.0, delayed=0.9, p_value=0.01):
    return {
        "ok": True,
        "selected_case": "default",
        "cases": {"default": {"select": {"sharpe": select, "alpha_sharpe": select}}},
        "robustness": {"select_sharpe_delayed_one_more_bar": delayed, "permutation_p_value_train_and_select": p_value},
    }


@pytest.mark.parametrize(
    "result,failed",
    [
        (_result(), []),
        (_result(select=-0.1), ["select Sharpe"]),
        (_result(delayed=0.3), ["delay"]),
        (_result(p_value=0.2), ["p-value"]),
        ({"ok": False, "problems": ["boom"]}, ["did not run: boom"]),
    ],
)
def test_the_gates(result, failed):
    found = R.gate_failures(result, CryptoStrategySettings(_env_file=None))
    assert len(found) == len(failed)
    assert all(word in sentence for word, sentence in zip(failed, found))


# -- the feedback ------------------------------------------------------------------------------


def test_the_feedback_accepts_the_best_and_never_shows_the_test_segment(ran, backend):
    feedback = StrategyExperiment2Feedback(_Scenario()).generate_feedback(ran, Trace(scen=_Scenario()))
    assert feedback.decision and feedback.acceptable
    assert feedback.observations == "o" and feedback.new_hypothesis == "n"
    (call,) = backend.calls
    assert "Accepted: oracle" in call["user"]
    assert '"lockbox"' not in call["user"] and "lockbox_diagnostics" not in call["user"]
    assert '"select"' in call["user"] and "signed_forward_return_by_horizon" in call["user"]


def _accepted(exp, name):
    exp.accepted_strategy = name
    return exp, HypothesisFeedback(reason="", decision=True)


def test_a_copy_of_an_accepted_strategy_is_not_added(ran, panel, backend):
    trace = Trace(scen=_Scenario())
    trace.hist.append(_accepted(ran, "oracle"))
    again = StrategyRunner(_Scenario()).develop(_experiment(oracle=ORACLE))
    feedback = StrategyExperiment2Feedback(_Scenario()).generate_feedback(again, trace)
    assert not feedback.decision and feedback.acceptable and again.accepted_strategy is None
    user = backend.calls[-1]["user"]
    assert "correlate 1.00 with an accepted strategy" in user
    assert "passed the gates but add too little to the book" in user
    assert "# The book so far" in user and '"lockbox"' not in user


def test_a_strategy_new_to_the_book_is_added(panel, backend):
    first = StrategyRunner(_Scenario()).develop(_experiment(btc=_oracle("0")))
    trace = Trace(scen=_Scenario())
    trace.hist.append(_accepted(first, "btc"))
    other = StrategyRunner(_Scenario()).develop(_experiment(sol=_oracle("2"), btc_again=_oracle("0")))
    feedback = StrategyExperiment2Feedback(_Scenario()).generate_feedback(other, trace)
    assert feedback.decision and other.accepted_strategy == "sol"
    assert abs(other.judged["sol"]["contribution"]["max_abs_correlation"]) < 0.3
    assert other.judged["sol"]["contribution"]["gain"] > 0.05
    assert other.judged["btc_again"]["not_new"]
    assert "Accepted: sol passed the gates and joins the book, raising its select Sharpe" in backend.calls[-1]["user"]


def test_the_runner_keeps_only_train_and_valid_returns(ran):
    frame = ran.strategy_returns["oracle"]
    assert set(frame["segment"].astype(str)) == {"train", "select"}
    assert frame.index.max() < pd.Timestamp("2024-06-01", tz="UTC")
    assert frame["returns"].dtype == np.float32


def test_nothing_passing_is_rejected_even_when_the_model_fails(ran, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr("rdagent.scenarios.crypto.strategy.developer.APIBackend", broken)
    monkeypatch.setenv("CRYPTO_STRATEGY_MIN_VALID_SHARPE", "1000")
    feedback = StrategyExperiment2Feedback(_Scenario()).generate_feedback(ran, Trace(scen=_Scenario()))
    assert not feedback.decision and not feedback.acceptable
    assert "no strategy of this round passed" in feedback.reason


# -- the report --------------------------------------------------------------------------------


def _log(ran, tmp_path):
    log = tmp_path / "log"
    storage = FileStorage(log)
    ran.accepted_strategy = "oracle"
    storage.log(ran, tag="Loop_0.running.runner result.1")
    storage.log(HypothesisFeedback(reason="", decision=True), tag="Loop_0.feedback.feedback.1")
    return log


def test_the_report_keeps_the_lockbox_closed_by_default(ran, tmp_path, capsys):
    pytest.importorskip("plotly")
    log = _log(ran, tmp_path)
    assert report.main([str(log)]) == 0
    printed = capsys.readouterr().out
    assert "lockbox" not in printed.split("\n")[0] and "The lockbox is closed" in printed
    assert "| 0 | oracle |" in printed and "PBO over the select period" in printed
    assert not (log / report.MARKER).exists()
    found = report.collect(log, CryptoStrategySettings(_env_file=None))
    assert not any(key.startswith("lockbox") for row in found["rows"] for key in row)
    frame = found["curves"]["0:oracle"][0]
    assert set(frame["segment"]) == {"train", "select"}
    page = (log / "strategy_report.html").read_text()
    assert "The lockbox is closed" in page


def test_opening_the_lockbox_shows_it_once_and_leaves_a_mark(ran, tmp_path, capsys):
    pytest.importorskip("plotly")
    log = _log(ran, tmp_path)
    assert report.main([str(log), "--open-lockbox"]) == 0
    printed = capsys.readouterr().out
    assert "lockbox_deflated_sharpe" in printed and "Book (oracle)" in printed
    marker = json.loads((log / report.MARKER).read_text())
    assert marker["strategies"] == ["0:oracle", "0:trend"] and marker["lockbox_start"] == "2024-06-01"
    found = report.collect(log, CryptoStrategySettings(_env_file=None), open_lockbox=True)
    assert found["trials"] == 3
    oracle = next(row for row in found["rows"] if row["strategy"] == "oracle")
    assert oracle["in_book"] and oracle["lockbox_sharpe"] > 5 and 0.9 < oracle["lockbox_deflated_sharpe"] <= 1
    assert found["book"]["lockbox_sharpe"] > 5
    # Each member at the members' average risk: the book's returns are returns, not z-scores.
    book = found["book"]["returns"].to_numpy()
    assert np.abs(book).max() < 0.2
    page = (log / "strategy_report.html").read_text()
    assert page.count("plotly-graph-div") >= 4 and "Book: oracle" in page
    assert report.main([str(log)]) == 0
    assert "The lockbox was opened on" in capsys.readouterr().out


def test_pbo_separates_noise_from_a_real_edge():
    rng = np.random.default_rng(0)
    index = pd.date_range("2023-01-01", periods=2000, freq="4h", tz="UTC")
    segment = pd.Series("select", index=index)

    def experiment(columns):
        exp = CryptoStrategyExperiment([], hypothesis=None)
        exp.strategy_returns = {
            f"s{k}": pd.DataFrame({"returns": column, "segment": segment}) for k, column in enumerate(columns)
        }
        return (0, exp, None)

    noise = [rng.normal(0, 0.01, 2000) for _ in range(10)]
    edge = [rng.normal(0.003, 0.01, 2000)] + [rng.normal(0, 0.01, 2000) for _ in range(9)]
    assert 0.25 < report.pbo([experiment(noise)])["pbo"] < 0.75
    assert report.pbo([experiment(edge)])["pbo"] < 0.05
    assert np.isnan(report.pbo([experiment(noise[:1])])["pbo"])


def test_walk_forward_chooses_each_case_on_the_past_only(panel):
    settings = CryptoStrategySettings(_env_file=None)
    loaded = engine.Panel.load(panel)
    spec = validate_factor_spec(TREND, available_fields=list(engine.FIELDS))
    found = report.walk_forward(spec, loaded, settings, freq="ME", min_periods=2)
    # Train and select run January to May: March, April and May are traded, each on what came before.
    assert [start[:7] for start, _ in found["choices"]] == ["2024-03", "2024-04", "2024-05"]
    assert found["cases"] == 2 and found["returns"].index.min() >= pd.Timestamp("2024-03-01", tz="UTC")
    assert found["returns"].index.max() < pd.Timestamp("2024-06-01", tz="UTC")
