"""The phasor coder summarizes a failed factor before asking for the next one.

These tests stub the chat backend. They check which call is made and what the
prompt contains, not the quality of a live model response.
"""

import json

import pytest

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.phasor_coder.config import PhasorCoSTEERSettings
from rdagent.components.coder.phasor_coder.evolving_strategy import (
    PhasorMultiProcessEvolvingStrategy,
)
from rdagent.components.coder.phasor_coder.task import PhasorFactorTask
from rdagent.components.coder.phasor_coder.workspace import PhasorFactorWorkspace
from rdagent.core.scenario import Scenario

pytestmark = pytest.mark.offline

CRITIC = "critic 1: $funding_rate is not on this panel; use $close."
VALID_FACTOR = {
    "mode": "expression",
    "expression": "Mean($close, 20) / $close - 1",
    "fields": ["close"],
    "reason": "Twenty-bar momentum of close.",
}


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "crypto perpetuals"

    @property
    def rich_style_description(self) -> str:
        return "crypto perpetuals"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:
        return "crypto perpetuals, one bar grid"

    def get_runtime_environment(self) -> str:
        return ""


class _Backend:
    """Records each completion. The summary prompt is over the limit while it quotes two failures."""

    chat_token_limit = 100
    calls: list[dict]

    def __init__(self, use_chat_cache: bool = False) -> None:
        self.use_chat_cache = use_chat_cache
        self.calls = []

    def build_messages_and_calculate_token(self, user_prompt: str, system_prompt: str) -> int:
        if "Write short critics" in system_prompt and user_prompt.count("Similar failure") > 1:
            return 1_000
        return 1

    def build_messages_and_create_chat_completion(
        self, user_prompt: str, system_prompt: str, json_mode: bool = False
    ) -> str:
        self.calls.append(
            {"user": user_prompt, "system": system_prompt, "json_mode": json_mode}
        )
        if json_mode:
            return json.dumps(VALID_FACTOR)
        return CRITIC


def _task(name: str) -> PhasorFactorTask:
    return PhasorFactorTask(
        factor_name=name,
        factor_description=f"{name} description",
        factor_formulation=f"{name} formulation",
        fields=["close", "volume"],
    )


def _knowledge(task: PhasorFactorTask, spec: dict, execution: str) -> CoSTEERKnowledge:
    workspace = PhasorFactorWorkspace(target_task=task)
    workspace.inject_files(**{"factor.json": json.dumps(spec)})
    feedback = CoSTEERSingleFeedback(
        execution=execution,
        return_checking=None,
        code="",
        final_decision=False,
    )
    return CoSTEERKnowledge(target_task=task, implementation=workspace, feedback=feedback)


def _strategy(backend: _Backend, *, summarize: bool = True) -> PhasorMultiProcessEvolvingStrategy:
    settings = PhasorCoSTEERSettings(v2_error_summary=summarize, llm_retries=2)
    return PhasorMultiProcessEvolvingStrategy(_Scenario(), settings)


def _queried(task: PhasorFactorTask, similar: list) -> CoSTEERQueriedKnowledgeV2:
    info = task.get_task_information()
    failure = _knowledge(
        task,
        {
            "mode": "expression",
            "expression": "Mean($funding_rate, 5)",
            "fields": ["funding_rate"],
            "reason": "Funding momentum.",
        },
        "field(s) not available: $funding_rate; available: close, volume",
    )
    return CoSTEERQueriedKnowledgeV2(
        task_to_former_failed_traces={info: ([failure], None)},
        task_to_similar_task_successful_knowledge={info: []},
        task_to_similar_error_successful_knowledge={info: similar},
    )


def test_error_summary_default_is_on() -> None:
    assert PhasorCoSTEERSettings.model_fields["v2_error_summary"].default is True


def test_failed_task_is_summarized_before_the_next_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "rdagent.components.coder.phasor_coder.evolving_strategy.APIBackend",
        lambda use_chat_cache=False: backend,
    )
    task = _task("funding_mom")
    fixed = _task("close_mom_fixed")
    similar = [
        (
            "unknown field",
            (
                _knowledge(
                    _task("used_open"),
                    {"mode": "expression", "expression": "Mean($open, 5)", "fields": ["open"], "reason": "bad"},
                    "no",
                ),
                _knowledge(
                    fixed,
                    {"mode": "expression", "expression": "Mean($close, 20)", "fields": ["close"], "reason": "fixed"},
                    "ok",
                ),
            ),
        ),
        (
            "bad window",
            (
                _knowledge(
                    _task("used_negative_window"),
                    {
                        "mode": "expression",
                        "expression": "Mean($close, -1)",
                        "fields": ["close"],
                        "reason": "bad window",
                    },
                    "no",
                ),
                _knowledge(
                    fixed,
                    {
                        "mode": "expression",
                        "expression": "Mean($close, 5)",
                        "fields": ["close"],
                        "reason": "fixed window",
                    },
                    "ok",
                ),
            ),
        ),
    ]
    # The summary prompt is over the limit while it still quotes two failures, so
    # one pair is dropped there. The implementation prompt is under the limit and
    # must still see both pairs.
    files = _strategy(backend).implement_one_task(task, queried_knowledge=_queried(task, similar))

    assert [call["json_mode"] for call in backend.calls] == [False, True]
    summary, implementation = backend.calls
    assert "Mean($funding_rate, 5)" in summary["system"]
    assert "field(s) not available: $funding_rate" in summary["system"]
    assert summary["user"].count("Similar failure") == 1
    assert "Mean($close, 20)" in summary["user"]
    assert CRITIC in implementation["user"]
    assert "used_open" in implementation["user"]
    assert "Mean($open, 5)" in implementation["user"]
    assert "used_negative_window" in implementation["user"]
    assert set(files) == {"factor.json"}
    written = json.loads(files["factor.json"])
    assert written["expression"] == VALID_FACTOR["expression"]


def test_first_attempt_does_not_summarize(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "rdagent.components.coder.phasor_coder.evolving_strategy.APIBackend",
        lambda use_chat_cache=False: backend,
    )
    files = _strategy(backend).implement_one_task(_task("fresh"))

    assert [call["json_mode"] for call in backend.calls] == [True]
    assert "Critics" not in backend.calls[0]["user"]
    assert json.loads(files["factor.json"])["expression"] == VALID_FACTOR["expression"]


def test_summary_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "rdagent.components.coder.phasor_coder.evolving_strategy.APIBackend",
        lambda use_chat_cache=False: backend,
    )
    task = _task("funding_mom")
    _strategy(backend, summarize=False).implement_one_task(task, queried_knowledge=_queried(task, []))

    assert [call["json_mode"] for call in backend.calls] == [True]
    assert CRITIC not in backend.calls[0]["user"]


def test_failure_without_similar_fixes_is_quoted_without_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend()
    monkeypatch.setattr(
        "rdagent.components.coder.phasor_coder.evolving_strategy.APIBackend",
        lambda use_chat_cache=False: backend,
    )
    task = _task("funding_mom")
    _strategy(backend).implement_one_task(task, queried_knowledge=_queried(task, []))

    assert [call["json_mode"] for call in backend.calls] == [True]
    prompt = backend.calls[0]["user"]
    assert "Mean($funding_rate, 5)" in prompt
    assert "field(s) not available: $funding_rate" in prompt
    assert "Critics" not in prompt


_PYTHON_FACTOR = json.dumps(
    {
        "mode": "python",
        "code": "def calculate(fields, index=None, symbols=None):\n    return fields['close']\n",
        "fields": ["close"],
        "reason": "The value is the close itself.",
    }
)


def test_main_prints_the_prompt_without_calling_the_model(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_the_model(*args, **kwargs):
        raise AssertionError("the model was called")

    monkeypatch.setattr(
        "rdagent.scenarios.crypto.proposal.factor_proposal._load_dotenv_and_refresh_settings",
        lambda: None,
    )
    monkeypatch.setattr("rdagent.oai.llm_utils.APIBackend", refuse_the_model)

    from rdagent.components.coder.phasor_coder.__main__ import main

    assert main(["--print-only", "--name", "close_momentum"]) == 0
    printed = capsys.readouterr().out
    assert "===== system prompt =====" in printed
    assert "===== user prompt =====" in printed
    assert "close_momentum" in printed
    assert "===== model response =====" not in printed


def test_main_prints_the_generated_factor(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rdagent.scenarios.crypto.proposal.factor_proposal._load_dotenv_and_refresh_settings",
        lambda: None,
    )

    class _Once:
        def build_messages_and_create_chat_completion(self, **kwargs):
            assert kwargs["json_mode"] is True
            assert "close_momentum" in kwargs["user_prompt"]
            return _PYTHON_FACTOR

    monkeypatch.setattr("rdagent.oai.llm_utils.APIBackend", lambda *args, **kwargs: _Once())

    from rdagent.components.coder.phasor_coder.__main__ import main

    assert main(["--name", "close_momentum"]) == 0
    printed = capsys.readouterr().out
    assert "===== model response =====" in printed
    assert "===== factor.json =====" in printed
    assert "===== factor.py =====" not in printed
    assert "def calculate(" in printed
