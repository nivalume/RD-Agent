"""Run only the crypto hypothesis and hypothesis2experiment stages.

The scenario is fixed to crypto perpetual futures: the loop is always built from
``CRYPTO_PROP_SETTING`` (CryptoFactorScenario / CryptoFactorHypothesisGen /
CryptoFactorHypothesis2Experiment) and the script exposes no scenario switch and no coding, evaluation
or backtest. It calls ``CryptoFactorRDLoop.propose()`` and ``CryptoFactorRDLoop.exp_gen()`` directly
rather than ``loop.run()``, so no session is dumped under ``log/``.

For each of the two stages it prints:

1. the exact system prompt and user prompt sent to the LLM,
2. the raw LLM response, plus a check of whether it is pure JSON,
3. the parsed result (Hypothesis fields, or the generated FactorTasks).

The prompts are recorded before the call, so they are still printed when the LLM request fails (bad
key, timeout, unparsable JSON, ...).

The data the scenario describes comes from ``CRYPTO_DATA_ROOT_DIR`` (default ``~/jarvis-data``): the
schemas under ``schemas/`` and the rows under ``crypto/data/``. Without them the prompt simply says no
data schema is available, which is enough for these two stages.

Usage:
    conda activate rdagent
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/crypto_factor.py
"""

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

from rdagent.app.crypto_rd_loop.conf import CRYPTO_PROP_SETTING  # noqa: E402
from rdagent.app.crypto_rd_loop.factor import CryptoFactorRDLoop  # noqa: E402
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend  # noqa: E402
from rdagent.oai.llm_conf import LLM_SETTINGS  # noqa: E402
from rdagent.scenarios.crypto.conf import CryptoSettings  # noqa: E402
from rdagent.scenarios.crypto.experiment.factor_experiment import NO_DATA_SCHEMA  # noqa: E402

# Recorded (system prompt, user prompt, raw response) for every LLM call.
LLM_CALLS: list[dict[str, Any]] = []

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
    }
    LLM_CALLS.append(record)

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
        _sub(f"{stage} | LLM call {index}/{len(calls)} (json_mode={call['json_mode']}, {call['seconds']:.1f}s)")
        _sub(f"{stage} | system prompt")
        print(call["system"])
        _sub(f"{stage} | user prompt")
        print(call["user"])
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


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    data_root = Path(CryptoSettings().data_root_dir)

    print(f"CHAT_MODEL      = {LLM_SETTINGS.chat_model}")
    print(f"OPENAI_API_BASE = {LLM_SETTINGS.openai_api_base}")
    print(f"SCENARIO        = {CRYPTO_PROP_SETTING.scen} (fixed: crypto perpetual futures)")
    print(f"DATA_ROOT       = {data_root}")
    print(f"with runtime env= {CRYPTO_PROP_SETTING.with_runtime_environment}")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell "
            "that launches this script; .env maps ARK_API_KEY to OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1
    print(f"OPENAI_API_KEY  = ***{api_key[-4:]} (len={len(api_key)})")

    loop = CryptoFactorRDLoop(CRYPTO_PROP_SETTING)
    # What the prompt actually got, rather than what the settings say: the scenario drops a dataset
    # whose rows are not on disk, so a configured root can still render an empty section.
    source_data = loop.scen.get_source_data_desc()
    if NO_DATA_SCHEMA in source_data:
        print(f"DATA_SCHEMAS    = none; the prompt says no data schema is available (data root: {data_root})")
    else:
        print(f"DATA_SCHEMAS    = {source_data.count('### ')} dataset(s), {len(source_data)} chars of prompt")

    # ---------------------------------------------------------------- stage 1
    _header("Stage 1/2: hypothesis  (CryptoFactorHypothesisGen.gen)")
    LLM_CALLS.clear()
    try:
        hypothesis = loop.propose({})
    finally:
        # Print the prompts even if the request above raised.
        stage_1_calls = list(LLM_CALLS)
        _print_llm_calls("hypothesis", stage_1_calls)
    _print_hypothesis(hypothesis)

    # ---------------------------------------------------------------- stage 2
    _header("Stage 2/2: hypothesis2experiment  (CryptoFactorHypothesis2Experiment.convert)")
    LLM_CALLS.clear()
    try:
        experiment = loop.exp_gen({"propose": hypothesis})
    finally:
        stage_2_calls = list(LLM_CALLS)
        _print_llm_calls("hypothesis2experiment", stage_2_calls)
    _print_experiment(experiment)

    _header("summary")
    print(f"stage 1 LLM calls: {len(stage_1_calls)}")
    print(f"stage 2 LLM calls: {len(stage_2_calls)}")
    print(f"generated tasks  : {len(experiment.sub_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
