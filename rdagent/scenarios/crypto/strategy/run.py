"""Run a strategy in its own process: the coder's checks, or the full backtest.

    python -m rdagent.scenarios.crypto.strategy.run check strategy.json --panel <debug panel> --out check.json
    python -m rdagent.scenarios.crypto.strategy.run backtest strategy.json --panel <panel> --out result.json \
        [--trials trials.json]

A strategy in python mode is code the model wrote, so it runs here, in a child process with a time
limit, not in the loop's process. The settings come from ``CRYPTO_STRATEGY_`` variables; the result
is written as JSON for the parent to read. :func:`run_in_process` is the parent's side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rdagent.components.coder.phasor_coder.spec import FactorSpecError, validate_factor_spec
from rdagent.scenarios.crypto.strategy import engine
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings

SPEC_FILE = "strategy.json"


def load_spec(path: Path) -> engine.FactorSpec:
    data = json.loads(path.read_text())
    return validate_factor_spec(data, available_fields=list(engine.FIELDS))


def _check(spec_path: Path, panel_path: Path) -> dict[str, Any]:
    result = engine.check(load_spec(spec_path), engine.Panel.load(panel_path))
    return {"ok": result.ok, **asdict(result)}


RETURNS_FILE = "returns.parquet"


def _backtest(spec_path: Path, panel_path: Path, trials: list[float], returns_out: Path) -> dict[str, Any]:
    settings = CryptoStrategySettings()
    return engine.evaluate(
        load_spec(spec_path),
        engine.Panel.load(panel_path),
        settings.segments(),
        settings.backtest(),
        trial_sharpes=trials,
        returns_out=returns_out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.strategy.run")
    parser.add_argument("stage", choices=("check", "backtest"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=Path, help="per-bar Sharpe ratios of earlier trials, a JSON list")
    args = parser.parse_args(argv)
    try:
        if args.stage == "check":
            found = _check(args.spec, args.panel)
        else:
            trials = json.loads(args.trials.read_text()) if args.trials and args.trials.exists() else []
            found = _backtest(args.spec, args.panel, trials, args.out.parent / RETURNS_FILE)
            found["ok"] = True
    except (FactorSpecError, engine.StrategyError) as error:
        found = {"ok": False, "problems": [str(error)]}
    except Exception:  # anything else is reported, not lost with the process
        found = {"ok": False, "problems": [traceback.format_exc(limit=5)]}
    args.out.write_text(json.dumps(found, indent=2, default=float))
    return 0


def run_in_process(
    stage: str,
    workspace: Path,
    panel: Path,
    python_bin: str,
    timeout: int,
    trials: list[float] | None = None,
) -> dict[str, Any]:
    """Run a stage in a child process on the workspace's ``strategy.json``; its JSON result, or a
    result naming the failure."""
    out = workspace / f"{stage}.json"
    out.unlink(missing_ok=True)
    command = [python_bin, "-m", "rdagent.scenarios.crypto.strategy.run", stage, str(workspace / SPEC_FILE)]
    command += ["--panel", str(panel), "--out", str(out)]
    if trials is not None:
        trials_path = workspace / "trials.json"
        trials_path.write_text(json.dumps(trials))
        command += ["--trials", str(trials_path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=workspace)
    except subprocess.TimeoutExpired:
        return {"ok": False, "problems": [f"It ran longer than {timeout}s and was stopped."]}
    if not out.exists():
        tail = (completed.stderr or completed.stdout)[-3000:]
        return {"ok": False, "problems": [f"The run wrote no result (exit {completed.returncode}):\n{tail}"]}
    return json.loads(out.read_text())


if __name__ == "__main__":
    sys.exit(main())
