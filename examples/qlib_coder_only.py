"""Run only the qlib factor coder (CoSTEER): fixed factor tasks in, code + feedback out.

No hypothesis generation and no experiment generation: the factor specs come from ``TASKS`` below, so
the only LLM interaction is the coder's own evolving loop. That makes this the shortest way to see
what the coder is actually asked and what it actually writes.

For every LLM call it prints the system prompt, the user prompt and the raw response as they happen,
plus a one-line check of whether the response is the JSON the coder expects (``{"code": ...}``).
At the end it prints, for each task:

1. the ``factor.py`` the coder kept,
2. the last evaluation feedback (execution, value, code, final decision),
3. the factor values the code produced, when they are still readable.

Usage:
    conda activate rdagent
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/qlib_coder_only.py                     # KLEN, up to 2 evolving loops
    python examples/qlib_coder_only.py --task all --max_loop 3

The knowledge base is switched off by default (``--knowledge`` to enable it): after every task it
succeeds, the coder embeds the task into a vector store to learn from it in later runs, and that
needs an embedding model. ``EMBEDDING_MODEL`` in ``.env`` points at the chat model here, which does
not serve embeddings, so the run would end with a 404 from the provider. Point ``EMBEDDING_MODEL``
at a real embedding model if you want that part.

A full run needs the qlib factor environment and ``git_ignore_folder/factor_implementation_source_data``
(the daily_pv.h5 the generated code reads, produced by
``rdagent.scenarios.qlib.experiment.utils.generate_data_folder_from_qlib``). The coder caches an
evaluation by the code's hash under ``pickle_cache/``: if a run was evaluated with a broken
environment, delete that entry (or the whole ``pickle_cache/``) before trying again.
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

# Keep relative paths (log/, pickle_cache/, git_ignore_folder/) anchored at the repo root.
os.chdir(PROJECT_ROOT)

# CondaConf resolves the environment by name. Default it for callers that
# invoke this file with the env's python directly instead of conda activate.
os.environ.setdefault("CONDA_DEFAULT_ENV", "rdagent")

# Allow running the file from a fresh clone without an editable install.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdagent.components.coder.factor_coder import FactorCoSTEER  # noqa: E402
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS  # noqa: E402
from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask  # noqa: E402
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend  # noqa: E402
from rdagent.oai.llm_conf import LLM_SETTINGS  # noqa: E402
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario  # noqa: E402

# The factor specs handed straight to the coder. Nothing generates them: they are the input of this
# script, written the way hypothesis2experiment would have written them.
TASKS = {
    "KLEN": FactorTask(
        factor_name="KLEN",
        factor_description=(
            "[Price Volume Factor] Bar length: how wide the bar's range is relative to its open, "
            "a proxy for intrabar volatility. Higher means a wider trading range."
        ),
        factor_formulation=r"$\frac{high - low}{open}$",
        variables={
            "$high": "highest price of the bar",
            "$low": "lowest price of the bar",
            "$open": "open price of the bar",
        },
    ),
    "CORD5": FactorTask(
        factor_name="CORD5",
        factor_description=(
            "[Price Volume Factor] Correlation between the close-to-close return and the volume "
            "change over the last 5 bars: does a price move come with volume?"
        ),
        factor_formulation=(
            r"$Corr\left(\frac{close}{Ref(close, 1)}, "
            r"\log\left(\frac{volume}{Ref(volume, 1)} + 1\right), 5\right)$"
        ),
        variables={
            "$close": "close price of the bar",
            "$volume": "traded volume of the bar",
            "$Ref(x, n)": "the value of x n bars ago",
            "$Corr(x, y, n)": "rolling correlation of x and y over the last n bars",
        },
    ),
    "VSTD5": FactorTask(
        factor_name="VSTD5",
        factor_description=(
            "[Price Volume Factor] Volume stability: the 5-bar standard deviation of volume "
            "normalised by the current volume. Higher means volume has been unstable."
        ),
        factor_formulation=r"$\frac{Std(volume, 5)}{volume}$",
        variables={
            "$volume": "traded volume of the bar",
            "$Std(x, n)": "rolling standard deviation of x over the last n bars",
        },
    ),
}

# Recorded (system prompt, user prompt, raw response) for every LLM call.
LLM_CALLS: list[dict[str, Any]] = []
_TOTAL_SECONDS = 0.0

_ORIGINAL_CHAT_COMPLETION = BaseAPIBackend.build_messages_and_create_chat_completion


def _json_probe(raw: str, reply: dict[str, Any]) -> str:
    """One line on whether the response is the JSON object the coder parses."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        return f"not pure JSON ({error}) -> the coder's parser would raise"
    if not isinstance(payload, dict):
        return f"JSON but a {type(payload).__name__}, not an object"
    return f"JSON ok, keys {sorted(payload)}"


def _recording_chat_completion(
    self: Any,
    user_prompt: str,
    system_prompt: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Print the prompt, forward the call, print the response."""
    global _TOTAL_SECONDS
    index = len(LLM_CALLS) + 1
    record = {
        "system": LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt,
        "user": user_prompt,
        "response": None,
        "error": None,
        "json_mode": bool(kwargs.get("json_mode")),
        "seconds": 0.0,
        "probe": "",
    }
    LLM_CALLS.append(record)

    print()
    print(f"{'=' * 100}")
    print(f"# LLM call {index} (json_mode={record['json_mode']}, system {len(record['system'])} chars, "
          f"user {len(record['user'])} chars)")
    print(f"{'=' * 100}")
    print("----- system prompt -----")
    print(record["system"])
    print("----- user prompt -----")
    print(record["user"])
    print("----- raw response -----", flush=True)

    start = time.time()
    try:
        resp = _ORIGINAL_CHAT_COMPLETION(self, user_prompt, system_prompt, *args, **kwargs)
    except BaseException as e:  # record, then let the caller deal with it
        record["error"] = repr(e)
        record["seconds"] = time.time() - start
        _TOTAL_SECONDS += record["seconds"]
        print(f"<no response: {record['error']}>", flush=True)
        raise
    record["seconds"] = time.time() - start
    _TOTAL_SECONDS += record["seconds"]
    record["response"] = resp
    record["probe"] = _json_probe(resp, record)
    print(resp)
    print(f"----- JSON probe -----\n{record['probe']} ({record['seconds']:.1f}s)", flush=True)
    return resp


# Installed at import time so every coder call is printed. The wrapper only records and prints; it
# forwards all arguments unchanged to the real backend.
BaseAPIBackend.build_messages_and_create_chat_completion = _recording_chat_completion


def _header(title: str) -> None:
    print()
    print("#" * 100)
    print(f"# {title}")
    print("#" * 100)


def _feedback_lines(feedback: Any) -> list[str]:
    """The evaluation fields, whatever concrete feedback class the coder produced."""
    lines = []
    for field in (
        "execution_feedback",
        "code_feedback",
        "value_feedback",
        "final_feedback",
        "final_decision",
        "value_generated_flag",
    ):
        if hasattr(feedback, field):
            value = getattr(feedback, field)
            if value not in (None, ""):
                lines.append(f"{field}:\n{value}")
    return lines


def _print_feedback(feedback: Any) -> None:
    """One evaluation per task: execution, value, code, final decision."""
    print()
    print("----- last evaluation feedback -----")
    singles = getattr(feedback, "feedback_list", None)
    if singles is None:
        print("\n\n".join(_feedback_lines(feedback)) or "<the coder produced no evaluation fields>")
        return
    for index, single in enumerate(singles, 1):
        print(f"\n--- task {index} ---")
        print("\n\n".join(_feedback_lines(single)) or "<no evaluation fields>")


def _print_values(workspace: Any) -> None:
    """The factor values the kept code produced, when the workspace still has them."""
    result_path = Path(workspace.workspace_path) / "result.h5"
    if not result_path.exists():
        return
    try:
        import pandas as pd

        values = pd.read_hdf(result_path, key="data")
    except Exception as error:  # reading the result is a nicety, not part of the demo
        print(f"----- result.h5 -----\n<could not read: {error!r}>")
        return
    print(f"----- result.h5 -----\nshape {values.shape}, index {list(values.index.names)}, "
          f"columns {list(values.columns)}")
    print(values.head(3).to_string())


def _print_result(experiment: FactorExperiment) -> None:
    _header("coder result")
    print(f"experiment      : {type(experiment).__name__}")
    print(f"n_sub_tasks     : {len(experiment.sub_tasks)}")
    print(f"prop_dev_feedback: {type(experiment.prop_dev_feedback).__name__}")

    if experiment.prop_dev_feedback is not None:
        _print_feedback(experiment.prop_dev_feedback)

    for index, (task, workspace) in enumerate(zip(experiment.sub_tasks, experiment.sub_workspace_list), 1):
        print()
        print("-" * 100)
        print(f"[{index}] {task.factor_name}")
        print(f"    description : {task.factor_description}")
        print(f"    formulation : {task.factor_formulation}")
        print("-" * 100)
        if workspace is None:
            print("<the coder produced no workspace for this task>")
            continue
        code = (workspace.file_dict or {}).get("factor.py")
        print("----- factor.py -----")
        print(code if code else "<the coder did not keep any code>")
        _print_values(workspace)
        print(f"----- workspace -----\n{workspace.workspace_path}")


def main(task: str = "KLEN", max_loop: int = 2, knowledge: bool = False) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"CHAT_MODEL      = {LLM_SETTINGS.chat_model}")
    print(f"OPENAI_API_BASE = {LLM_SETTINGS.openai_api_base}")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell "
            "that launches this script; .env maps ARK_API_KEY to OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1
    print(f"OPENAI_API_KEY  = ***{api_key[-4:]} (len={len(api_key)})")

    names = list(TASKS) if task.lower() == "all" else [task.upper()]
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        print(f"ERROR: unknown task(s) {unknown}; choose from {list(TASKS)} or 'all'.", file=sys.stderr)
        return 1
    tasks = [TASKS[name] for name in names]

    # The evolving loop stops after `max_loop` implementations; the default 10 is too many to watch.
    FACTOR_COSTEER_SETTINGS.max_loop = max_loop
    # The workspace runs `factor.py` with `python` from PATH; use the interpreter running this script,
    # so the example also works when the env was not activated.
    FACTOR_COSTEER_SETTINGS.python_bin = sys.executable
    print(f"MAX_LOOP        = {max_loop}")
    print(f"TASKS           = {[t.factor_name for t in tasks]}")
    print(f"DATA_FOLDER     = {FACTOR_COSTEER_SETTINGS.data_folder}")

    _header("scenario")
    start = time.time()
    scen = QlibFactorScenario()
    print(f"scenario        : {type(scen).__name__} ({time.time() - start:.1f}s)")
    runtime_env = str(scen.get_runtime_environment()).strip()
    try:
        runtime_env_keys = sorted(json.loads(runtime_env))
        runtime_env_summary = f"packages/environment reported as {runtime_env_keys}"
    except (json.JSONDecodeError, TypeError):
        runtime_env_summary = (runtime_env.splitlines() or ["?"])[0][:100]
    print(f"runtime env     : {runtime_env_summary}")
    print(f"source data     : {len(scen.get_source_data_desc())} chars")

    experiment = FactorExperiment(sub_tasks=tasks)
    _header(f"coder: {len(tasks)} task(s), up to {max_loop} loops")
    print(f"knowledge base  : {'on' if knowledge else 'off (--knowledge to enable)'}")
    start = time.time()
    try:
        # `knowledge_self_gen` is what writes a succeeded task into the coder's knowledge base; that
        # write embeds the task, so it needs a working `EMBEDDING_MODEL`. Off by default here.
        # `with_knowledge` must stay on: the multi-process evolving strategy needs the queried
        # knowledge (an empty knowledge base is fine, it just returns nothing).
        experiment = FactorCoSTEER(scen, knowledge_self_gen=knowledge).develop(experiment)
    finally:
        # Keep the prompts visible even when the coder raises (bad key, unparsable JSON, ...).
        elapsed = time.time() - start
        _header("LLM calls summary")
        print(f"calls           : {len(LLM_CALLS)}")
        print(f"time in the LLM : {_TOTAL_SECONDS:.1f}s of {elapsed:.1f}s")
        for index, call in enumerate(LLM_CALLS, 1):
            print(f"  [{index}] {call['seconds']:5.1f}s  json_mode={call['json_mode']!s:5}  {call['probe']}")

    _print_result(experiment)
    return 0


if __name__ == "__main__":
    import fire

    fire.Fire(main)
