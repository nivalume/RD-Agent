"""The crypto strategy example only drives the stages exported by the strategy loop."""

import importlib.util
from pathlib import Path

import pytest
from rdagent.app.crypto_rd_loop.strategy_loops import STAGES, run

pytestmark = pytest.mark.offline

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "crypto_strategy.py"


def _example():
    spec = importlib.util.spec_from_file_location("crypto_strategy_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_print_only_renders_the_selected_stage_without_calling_the_model(capsys, monkeypatch):
    calls = []

    class _Backend:
        def build_messages_and_create_chat_completion(self, *args, **kwargs):
            calls.append(args)
            message = "the model was called"
            raise AssertionError(message)

    monkeypatch.setattr(
        "rdagent.scenarios.crypto.proposal.strategy_proposal.APIBackend",
        lambda *args, **kwargs: _Backend(),
    )

    assert run(stages=("hypothesis",), print_only=True) == 0
    assert calls == []

    printed = capsys.readouterr().out
    assert "PRINT_ONLY      = True" in printed
    assert "STAGES          = hypothesis" in printed
    assert "===== hypothesis system prompt =====" in printed
    assert "Propose tradeable strategies" in printed
    assert "===== experiment system prompt =====" not in printed


def test_print_only_experiment_uses_a_placeholder_hypothesis(capsys):
    assert run(stages=("experiment",), print_only=True) == 0

    printed = capsys.readouterr().out
    assert "placeholder hypothesis" in printed
    assert "===== experiment system prompt =====" in printed
    assert "<the mechanism proposed by the hypothesis stage>" in printed
    assert "===== hypothesis system prompt =====" not in printed


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        run(stages=("coding",))


def test_the_example_forwards_stages_and_print_only(monkeypatch):
    monkeypatch.chdir(Path.cwd())
    calls = []

    def _record(stages, *, print_only, with_runtime_environment):
        calls.append((stages, print_only, with_runtime_environment))
        return 0

    example = _example()
    monkeypatch.setattr(example, "run", _record)

    assert example.main(["--print-only", "--stage", "experiment", "--stage", "hypothesis"]) == 0

    assert calls == [(["experiment", "hypothesis"], True, False)]
    assert example.STAGES == STAGES
