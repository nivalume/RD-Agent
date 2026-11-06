"""Run the stages of the crypto qlib factor loop up to the point you choose.

The loop is always the crypto qlib factor one (``CryptoQlibFactorRDLoop`` built from
``CryptoQlibFactorPropSetting``: CryptoQlibFactorScenario / CryptoFactorHypothesisGen /
CryptoQlibFactorHypothesis2Experiment). The script calls ``loop._propose()`` and ``loop._exp_gen()``
directly rather than ``loop.run()``, so nothing is coded, evaluated or backtested and no session is
dumped under ``log/``. ``--mode`` says how far to go:

``prompt``      print the prompts only; the model is not called (the default, and the only mode that
                needs no API key).
``hypothesis``  print the prompt, call the model, print the hypothesis it proposes.
``factor``      print the prompts, call the model for the hypothesis, then call it again to turn the
                hypothesis into factors, and print those.

The prompts printed are the ones the loop's own classes render and send: the model call is
intercepted at the backend, so they cannot drift from the real ones. In ``prompt`` mode the
hypothesis-to-factors prompt needs a hypothesis that does not exist yet, so it is rendered for a
placeholder one, and says so.

For each stage that runs it prints:

1. the exact system prompt and user prompt sent to the LLM,
2. the raw LLM response, plus a check of whether it is pure JSON (modes that call the model),
3. the parsed result (Hypothesis fields, or the generated FactorTasks).

The prompts are recorded before the call, so they are still printed when the LLM request fails (bad
key, timeout, unparsable JSON, ...).

Building the scenario needs the crypto qlib setup described in ``rdagent/scenarios/crypto/qlib/README.md``
(the data under ``~/.qlib/qlib_data/crypto_data`` and ``MODEL_CoSTEER_ENV_TYPE``), in every mode: the
factor coder's tables are written from it and the factor environment is listed in the prompt.

Usage:
    conda activate rdagent
    python examples/crypto_factor.py                       # = --mode prompt
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/crypto_factor.py --mode hypothesis
    python examples/crypto_factor.py --mode factor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env lives in the repository root. Load it before importing rdagent,
# because rdagent creates its settings objects at import time.
load_dotenv(PROJECT_ROOT / ".env")

# Keep relative paths (log/, pickle_cache/) anchored at the repo root.
os.chdir(PROJECT_ROOT)

# CondaConf resolves the environment by name. Default it for callers that
# invoke this file with the env's python directly instead of conda activate.
os.environ.setdefault("CONDA_DEFAULT_ENV", "rdagent")

# Allow running the file from a fresh clone without an editable install.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdagent.app.crypto_rd_loop.qlib_conf import CryptoQlibFactorPropSetting  # noqa: E402
from rdagent.app.crypto_rd_loop.qlib_loops import CryptoQlibFactorRDLoop  # noqa: E402
from rdagent.core.proposal import Hypothesis  # noqa: E402
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend  # noqa: E402
from rdagent.oai.llm_conf import LLM_SETTINGS  # noqa: E402
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS  # noqa: E402

MODES = ("prompt", "hypothesis", "factor")

# Recorded (system prompt, user prompt, raw response) for every LLM call.
LLM_CALLS: list[dict[str, Any]] = []

# False in `prompt` mode: the call is recorded and then stopped before it reaches the model.
CALL_LLM = True


class PromptOnly(BaseException):
    """Stops a stage once its prompt is recorded.

    A ``BaseException``, so the ``wait_retry`` around the converter does not retry it as a failed call.
    """

_ORIGINAL_CHAT_COMPLETION = BaseAPIBackend.build_messages_and_create_chat_completion


def _recording_chat_completion(
    self: Any,
    user_prompt: str,
    system_prompt: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Wrap the backend call so the prompts can be printed afterwards."""
    # Recorded before the call: a failed request still leaves its prompt behind.
    record = {
        "system": LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt,
        "user": user_prompt,
        "response": None,
        "error": None,
        "json_mode": bool(kwargs.get("json_mode")),
        "seconds": 0.0,
        "sent": CALL_LLM,
    }
    LLM_CALLS.append(record)
    if not CALL_LLM:
        raise PromptOnly

    start = time.time()
    try:
        resp = _ORIGINAL_CHAT_COMPLETION(self, user_prompt, system_prompt, *args, **kwargs)
    except BaseException as e:  # record, then let the caller deal with it
        record["error"] = repr(e)
        raise
    finally:
        record["seconds"] = time.time() - start
    record["response"] = resp
    return resp


# Installed at import time so every stage below is recorded. The wrapper only
# records; it forwards all arguments unchanged to the real backend.
BaseAPIBackend.build_messages_and_create_chat_completion = _recording_chat_completion


def _header(title: str) -> None:
    print()
    print("#" * 100)
    print(f"# {title}")
    print("#" * 100)


def _sub(title: str) -> None:
    print()
    print(f"----- {title} -----")


def _first_json(text: str) -> tuple[Any, str] | None:
    """Return the first JSON value in `text` and the text that follows it."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return payload, text[start + end :]
    return None


def _print_json_probe(stage: str, calls: list[dict[str, Any]]) -> None:
    """Show whether the raw response is pure JSON; rdagent's convert_response needs that."""
    for index, call in enumerate(calls, 1):
        raw = call["response"]
        if raw is None:
            continue
        _sub(f"{stage} | JSON probe for call {index}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            found = _first_json(raw)
            print(f"json.loads(raw) failed: {error}")
            if found is None:
                print("no JSON object found at all -> convert_response would raise")
                continue
            payload, tail = found
            print(f"the first JSON object alone parses fine: {len(payload)} top-level keys")
            print(f"trailing non-JSON text: {len(tail)} chars -> convert_response would raise")
            print(f"trailing preview: {tail.strip()[:300]!r}")
        else:
            print(f"json.loads(raw) ok: {len(payload)} top-level keys")


def _print_llm_calls(stage: str, calls: list[dict[str, Any]]) -> None:
    if not calls:
        _sub(f"{stage}: no LLM call recorded")
        return
    for index, call in enumerate(calls, 1):
        timing = f"{call['seconds']:.1f}s" if call["sent"] else "not sent"
        _sub(f"{stage} | LLM call {index}/{len(calls)} (json_mode={call['json_mode']}, {timing})")
        _sub(f"{stage} | system prompt")
        print(call["system"])
        _sub(f"{stage} | user prompt")
        print(call["user"])
        if not call["sent"]:
            continue
        _sub(f"{stage} | raw response")
        print(call["response"] if call["response"] is not None else f"<no response: {call['error']}>")
    _print_json_probe(stage, calls)


def _print_hypothesis(hypothesis: Any) -> None:
    _sub("hypothesis result")
    print(f"hypothesis:\n{hypothesis.hypothesis}")
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


def _print_experiment(experiment: Any) -> None:
    _sub("hypothesis2experiment result")
    print(f"experiment class   : {type(experiment).__name__}")
    print(f"n_sub_tasks        : {len(experiment.sub_tasks)}")
    print(f"n_based_experiments: {len(experiment.based_experiments)}")
    for index, task in enumerate(experiment.sub_tasks, 1):
        print(f"\n[{index}] factor_name : {task.factor_name}")
        print(f"    description : {task.factor_description}")
        print(f"    formulation : {task.factor_formulation}")
        print(f"    variables   : {task.variables}")


PLACEHOLDER_HYPOTHESIS = Hypothesis(
    hypothesis="<the hypothesis the model proposes in the hypothesis stage>",
    reason="<its reason>",
    concise_reason="",
    concise_observation="",
    concise_justification="",
    concise_knowledge="",
)
"""Stands in for the proposed hypothesis when the model is not called, so the second prompt can render."""


def _run_stage(stage: str, step: Any) -> Any:
    """Run one loop step, then print the prompts it sent even if it raised; None if it stopped at the prompt."""
    LLM_CALLS.clear()
    try:
        return step()
    except PromptOnly:
        return None
    finally:
        _print_llm_calls(stage, list(LLM_CALLS))


def main(argv: list[str] | None = None) -> int:
    global CALL_LLM

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="prompt",
        help="prompt: print the prompts only; hypothesis: also call the model for a hypothesis; "
        "factor: also turn the hypothesis into factors (default: prompt)",
    )
    mode = parser.parse_args(argv).mode
    CALL_LLM = mode != "prompt"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"MODE            = {mode}")
    print(f"CHAT_MODEL      = {LLM_SETTINGS.chat_model}")
    print(f"OPENAI_API_BASE = {LLM_SETTINGS.openai_api_base}")
    print(f"QLIB DATA       = {CRYPTO_QLIB_SETTINGS.provider_uri} (market {CRYPTO_QLIB_SETTINGS.market})")
    if CALL_LLM:
        if not api_key:
            print(
                "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell "
                "that launches this script; .env maps ARK_API_KEY to OPENAI_API_KEY.",
                file=sys.stderr,
            )
            return 1
        print(f"OPENAI_API_KEY  = ***{api_key[-4:]} (len={len(api_key)})")

    setting = CryptoQlibFactorPropSetting()
    print(f"SCENARIO        = {setting.scen}")
    loop = CryptoQlibFactorRDLoop(setting)

    # ---------------------------------------------------------------- stage 1
    _header(f"Stage 1/{1 if mode == 'hypothesis' else 2}: hypothesis  (CryptoFactorHypothesisGen.gen)")
    hypothesis = _run_stage("hypothesis", loop._propose)
    stage_1_calls = len(LLM_CALLS)
    if hypothesis is not None:
        _print_hypothesis(hypothesis)
    if mode == "hypothesis":
        _header("summary")
        print(f"stage 1 LLM calls: {stage_1_calls}")
        return 0

    # ---------------------------------------------------------------- stage 2
    _header("Stage 2/2: hypothesis2experiment  (CryptoQlibFactorHypothesis2Experiment.convert)")
    if hypothesis is None:
        print("The model was not called, so the prompt below is rendered for a placeholder hypothesis.")
        hypothesis = PLACEHOLDER_HYPOTHESIS
    experiment = _run_stage("hypothesis2experiment", lambda: loop._exp_gen(hypothesis))
    if experiment is not None:
        _print_experiment(experiment)

    _header("summary")
    print(f"stage 1 LLM calls: {stage_1_calls}")
    print(f"stage 2 LLM calls: {len(LLM_CALLS)}")
    if experiment is not None:
        print(f"generated tasks  : {len(experiment.sub_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
