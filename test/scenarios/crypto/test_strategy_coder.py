"""The strategy coder: its prompt, its artefact, and its checks on a debug panel.

The chat backend is stubbed; the checks run for real, in a child process, on a small synthetic
panel written the way jarvis-history writes one.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.strategy_coder import (
    StrategyCoSTEERSettings,
    StrategyEvaluatorForCoder,
    StrategyEvolvingStrategy,
    StrategyWorkspace,
)
from rdagent.core.scenario import Scenario
from rdagent.scenarios.crypto.experiment.strategy_experiment import StrategyTask
from rdagent.scenarios.crypto.strategy import engine

pytestmark = pytest.mark.offline

TREND = {
    "mode": "expression",
    "expression": "Sign(Mean($close, {n}) - Mean($close, 4*{n}))",
    "params": {"n": [3, 6]},
    "fields": ["close"],
    "reason": "Fast mean above the slow one is long.",
}
PEEKING = {
    "mode": "python",
    "code": (
        "import numpy as np\n\n"
        "def calculate(fields, index, symbols):\n"
        "    close = fields['close']\n"
        "    following = np.vstack([close[1:], np.full((1, close.shape[1]), np.nan)])\n"
        "    return np.sign(following - close)\n"
    ),
    "fields": ["close"],
    "reason": "Reads the next bar.",
}
HOLDING = {
    "mode": "python",
    "code": (
        "import numpy as np\n\n"
        "def calculate(fields, index, symbols):\n"
        "    close = fields['close']\n"
        "    position = np.zeros_like(close)\n"
        "    for t in range(1, len(close)):\n"
        "        up = close[t] > close[t - 1] * 1.01\n"
        "        down = close[t] < close[t - 1] * 0.99\n"
        "        position[t] = np.where(up, 1.0, np.where(down, 0.0, position[t - 1]))\n"
        "    return position\n"
    ),
    "fields": ["close"],
    "reason": "Holds after a jump until a drop: it depends on its own past.",
}


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "crypto perpetuals"

    @property
    def rich_style_description(self) -> str:
        return "crypto perpetuals"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:
        return "crypto perpetuals, 4h bars"

    def get_runtime_environment(self) -> str:
        return ""


def _task(name="ma_cross") -> StrategyTask:
    return StrategyTask(
        strategy_name=name,
        phenomenon="trend_following",
        description="Long the instruments whose fast mean is above the slow one.",
        mechanism="Slow diffusion of information.",
        return_source="Late buyers.",
        risk_borne="Reversals.",
        signal="sign(MA_n - MA_{4n})",
        position="sign of the signal",
        horizon="days",
        gate="none",
        testable_prediction="Signed forward returns are positive for a few days.",
        fails_when="Choppy markets.",
        parameters={"n": "3 to 6 bars"},
    )


@pytest.fixture
def panel_dir(tmp_path, monkeypatch):
    generator = np.random.default_rng(0)
    rows, columns = 500, 3
    close = 100 * np.exp(np.cumsum(generator.normal(0, 0.01, size=(rows, columns)), axis=0))
    times = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC").rename("time")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    debug = tmp_path / "panel" / "debug"
    debug.mkdir(parents=True)
    for name in engine.FIELDS:
        values = np.zeros_like(close) if name == "funding_rate" else close
        pd.DataFrame(values, index=times, columns=symbols).to_parquet(debug / f"{name}.parquet")
    pd.DataFrame(True, index=times, columns=symbols).to_parquet(debug / "pool.parquet")
    monkeypatch.setenv("CRYPTO_STRATEGY_PANEL_DIR", str(tmp_path / "panel"))
    monkeypatch.setenv("CRYPTO_STRATEGY_PYTHON_BIN", sys.executable)
    return debug


def _workspace(spec: dict | str) -> StrategyWorkspace:
    workspace = StrategyWorkspace(target_task=_task())
    workspace.inject_files(**{"strategy.json": spec if isinstance(spec, str) else json.dumps(spec)})
    return workspace


def _evaluate(spec):
    return StrategyEvaluatorForCoder(_Scenario()).evaluate(_task(), _workspace(spec))


# -- the evaluator -----------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [TREND, HOLDING], ids=["expression", "python"])
def test_a_sound_strategy_passes(panel_dir, spec):
    feedback = _evaluate(spec)
    assert feedback.final_decision, feedback.execution
    assert "without look-ahead" in feedback.execution


def test_a_strategy_that_reads_the_next_bar_is_sent_back(panel_dir):
    feedback = _evaluate(PEEKING)
    assert not feedback.final_decision
    assert "Look-ahead" in feedback.execution


@pytest.mark.parametrize(
    "spec,message",
    [
        ({**TREND, "params": {}, "expression": "$close / Mean($close, 5)"}, "range from"),
        ({**TREND, "params": {}, "expression": "$close * 0"}, "flat"),
        ({**TREND, "expression": "Sign(Ref($close, -1) - $close)", "params": {}}, "future"),
        ({**TREND, "fields": ["open_interest"]}, "open_interest"),
        ("not json", "JSON"),
    ],
    ids=["unbounded", "flat", "future", "unknown field", "not json"],
)
def test_broken_strategies_are_sent_back_with_the_reason(panel_dir, spec, message):
    feedback = _evaluate(spec)
    assert not feedback.final_decision
    assert message in feedback.execution, feedback.execution


def test_code_that_raises_is_sent_back_with_its_error(panel_dir):
    broken = {**HOLDING, "code": "def calculate(fields, index, symbols):\n    return 1 / 0\n"}
    feedback = _evaluate(broken)
    assert not feedback.final_decision and "ZeroDivisionError" in feedback.execution


# -- the evolving strategy ---------------------------------------------------------------------


class _Backend:
    chat_token_limit = 10**9

    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def __call__(self, use_chat_cache=False):
        return self

    def build_messages_and_calculate_token(self, user_prompt: str, system_prompt: str) -> int:
        return 1

    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, json_mode=False) -> str:
        self.calls.append({"user": user_prompt, "system": system_prompt, "json_mode": json_mode})
        return json.dumps(self.answer) if json_mode else "critic 1: Clip the position."


@pytest.fixture
def backend(monkeypatch):
    stub = _Backend(TREND)
    for module in ("phasor_coder", "strategy_coder"):
        monkeypatch.setattr(f"rdagent.components.coder.{module}.evolving_strategy.APIBackend", stub)
    return stub


def test_the_prompt_asks_for_a_position_and_the_file_is_strategy_json(backend):
    files = StrategyEvolvingStrategy(_Scenario(), StrategyCoSTEERSettings()).implement_one_task(_task())
    assert set(files) == {"strategy.json"}
    assert json.loads(files["strategy.json"])["expression"] == TREND["expression"]
    (call,) = backend.calls
    assert "target position" in call["system"] and "[-1, 1]" in call["system"]
    assert "`$funding_rate`" in call["system"] and "def calculate(fields, index, symbols)" in call["system"]
    assert "sign(MA_n - MA_{4n})" in call["user"]


def test_a_failed_attempt_is_summarized_and_quoted(backend):
    task = _task()
    failed = _workspace({**TREND, "params": {}, "expression": "$close"})
    feedback = CoSTEERSingleFeedback(
        execution="positions range from 50 to 120", return_checking=None, code="", final_decision=False
    )
    failure = CoSTEERKnowledge(target_task=task, implementation=failed, feedback=feedback)
    fixed = CoSTEERKnowledge(target_task=_task("other"), implementation=_workspace(TREND), feedback=feedback)
    info = task.get_task_information()
    queried = CoSTEERQueriedKnowledgeV2(
        task_to_former_failed_traces={info: ([failure], None)},
        task_to_similar_task_successful_knowledge={info: []},
        task_to_similar_error_successful_knowledge={info: [("unbounded", (failure, fixed))]},
    )
    StrategyEvolvingStrategy(_Scenario(), StrategyCoSTEERSettings()).implement_one_task(task, queried)
    summary, implementation = backend.calls
    assert not summary["json_mode"] and "positions range from 50 to 120" in summary["system"]
    assert '"expression": "$close"' in summary["system"]
    assert "critic 1: Clip the position." in implementation["user"]
    assert '"expression": "$close"' in implementation["user"]
