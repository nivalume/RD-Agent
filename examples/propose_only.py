"""Run only the Qlib factor hypothesis (propose) stage.

This calls FactorRDLoop._propose() directly, i.e. only
QlibFactorHypothesisGen.gen(). It does not run exp_gen, coding,
Docker, qlib backtests, or embedding/RAG.

Usage:
    conda activate rdagent
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/propose_only.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env lives in the repository root. Load it before importing rdagent,
# because rdagent creates its settings objects at import time.
load_dotenv(PROJECT_ROOT / ".env")

# Keep relative paths (git_ignore_folder/, log/) anchored at the repo root.
os.chdir(PROJECT_ROOT)

# CondaConf resolves the environment by name. Default it for callers that
# invoke this file with the env's python directly instead of conda activate.
os.environ.setdefault("CONDA_DEFAULT_ENV", "rdagent")

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING  # noqa: E402
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop  # noqa: E402


def main() -> int:
    model = os.environ.get("CHAT_MODEL", "")
    base_url = os.environ.get("OPENAI_API_BASE", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    print(f"CHAT_MODEL      = {model}")
    print(f"OPENAI_API_BASE = {base_url}")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell "
            "that launches this script; .env maps ARK_API_KEY to OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1
    print(f"OPENAI_API_KEY  = ***{api_key[-4:]} (len={len(api_key)})")
    print()

    loop = FactorRDLoop(FACTOR_PROP_SETTING)
    hypothesis = loop._propose()

    print("\n=== hypothesis ===")
    print(hypothesis.hypothesis)
    print("\n=== reason ===")
    print(hypothesis.reason)

    for field in (
        "concise_reason",
        "concise_observation",
        "concise_justification",
        "concise_knowledge",
    ):
        value = getattr(hypothesis, field, None)
        if value:
            print(f"\n=== {field} ===")
            print(value)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
