"""Run only the propose stage of a scenario.

This calls the scenario's hypothesis generator directly. It does not run exp_gen,
coding, Docker, backtests, or embedding/RAG.

Usage:
    conda activate rdagent
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/propose_only.py                              # qlib factors
    python examples/propose_only.py --scenario crypto            # crypto factors
    python examples/propose_only.py --scenario crypto --data-root ~/jarvis-data
"""

from __future__ import annotations

import argparse
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
from rdagent.core.proposal import Hypothesis, Trace  # noqa: E402


def propose_qlib() -> Hypothesis:
    """Qlib factors: the factor loop's propose step (it also logs and applies user interaction)."""
    return FactorRDLoop(FACTOR_PROP_SETTING)._propose()


def propose_crypto(data_root: Path | None) -> Hypothesis:
    """Crypto factors: the crypto hypothesis generator on a fresh trace."""
    from rdagent.scenarios.crypto.experiment.factor_experiment import CryptoFactorScenario
    from rdagent.scenarios.crypto.proposal.factor_proposal import CryptoFactorHypothesisGen

    scenario = CryptoFactorScenario(with_runtime_environment=False, data_root_dir=data_root)
    return CryptoFactorHypothesisGen(scenario).gen(Trace(scen=scenario))


SCENARIOS = {
    "qlib": propose_qlib,
    "crypto": propose_crypto,
}
"""Scenario name -> propose callable. Add a function (and an entry) for a new scenario."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="qlib",
        help="which scenario proposes; qlib when omitted",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="the data root whose schemas the crypto scenario renders; CRYPTO_DATA_ROOT_DIR when omitted",
    )
    args = parser.parse_args(argv)

    if args.data_root is not None and args.scenario != "crypto":
        parser.error("--data-root only applies to --scenario crypto")

    model = os.environ.get("CHAT_MODEL", "")
    base_url = os.environ.get("OPENAI_API_BASE", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    print(f"CHAT_MODEL      = {model}")
    print(f"OPENAI_API_BASE = {base_url}")
    print(f"SCENARIO        = {args.scenario}")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is empty. Export ARK_API_KEY in the shell "
            "that launches this script; .env maps ARK_API_KEY to OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1
    print(f"OPENAI_API_KEY  = ***{api_key[-4:]} (len={len(api_key)})")
    print()

    hypothesis = propose_crypto(args.data_root) if args.scenario == "crypto" else propose_qlib()

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
