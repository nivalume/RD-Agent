"""Run the crypto strategy proposal by calling the strategy loop.

The work lives in :mod:`rdagent.app.crypto_rd_loop.strategy_loops`. This script loads the repository
``.env``, then calls :func:`~rdagent.app.crypto_rd_loop.strategy_loops.run`. ``--stage`` chooses which
proposal stages run, and ``--print-only`` renders their prompts without calling the model.

Stages, in order:

``hypothesis``  the economic thesis.
``experiment``  the tradeable rule that thesis becomes.

Usage:
    conda activate rdagent
    python examples/crypto_strategy.py --print-only
    python examples/crypto_strategy.py --print-only --stage hypothesis
    export ARK_API_KEY=...        # .env maps it to OPENAI_API_KEY
    python examples/crypto_strategy.py
    python examples/crypto_strategy.py --stage hypothesis
    python examples/crypto_strategy.py --stage hypothesis --stage experiment
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

# Keep relative paths (log/, pickle_cache/) anchored at the repo root.
os.chdir(PROJECT_ROOT)

# Allow running the file from a fresh clone without an editable install.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdagent.app.crypto_rd_loop.strategy_loops import STAGES, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse the stage and print-only flags and call the strategy loop."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help=(
            "proposal stage to run; repeat to name several. "
            f"Default: {', '.join(STAGES)}, in that order"
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="render the prompts of the selected stages and do not call the model",
    )
    parser.add_argument(
        "--runtime-environment",
        action="store_true",
        help="list the factor environment's packages in the prompt; off by default",
    )
    args = parser.parse_args(argv)
    return run(
        stages=args.stage,
        print_only=args.print_only,
        with_runtime_environment=args.runtime_environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
