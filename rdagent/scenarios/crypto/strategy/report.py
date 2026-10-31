"""Every strategy a loop tested, on all three segments, with the test Sharpe deflated by the trials.

    python -m rdagent.scenarios.crypto.strategy.report log/<session> [--json out.json]

This is the only place the test segment is shown. The deflated Sharpe of a strategy's test returns
counts every trial of the session (each case of each strategy, accepted or not): it is the
probability that the true Sharpe is above zero after the search that found it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdagent.scenarios.crypto.strategy import results as R
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.run import RETURNS_FILE


def _experiments(log: Path) -> list[tuple[int, Any, Any]]:
    """(loop index, experiment, feedback) of every round that was backtested, in order."""
    from rdagent.log.storage import FileStorage

    storage = FileStorage(log)
    feedback = {}
    for message in storage.iter_msg(tag="feedback"):
        feedback[_loop(message.tag)] = message.content
    found = []
    for message in storage.iter_msg(tag="runner result"):
        if hasattr(message.content, "strategy_results"):
            index = _loop(message.tag)
            found.append((index, message.content, feedback.get(index)))
    return sorted(found, key=lambda item: item[0])


def _loop(tag: str) -> int:
    for part in tag.split("."):
        if part.startswith("Loop_"):
            return int(part.removeprefix("Loop_"))
    return -1


def deflated(returns: np.ndarray, trials: list[float]) -> float:
    import phasor

    returns = returns[np.isfinite(returns)]
    if len(trials) < 2 or returns.size < 3:
        return float("nan")
    return float(phasor.robust.deflated_sharpe(returns, np.asarray(trials)))


def rows(log: Path, settings: CryptoStrategySettings) -> tuple[list[dict[str, Any]], int]:
    """One row a strategy, and the number of trials the deflated Sharpe counts."""
    experiments = _experiments(log)
    trials = [value for _, exp, _ in experiments for value in (getattr(exp, "trial_sharpes", None) or [])]
    found = []
    for index, exp, feedback in experiments:
        workspaces = {task.strategy_name: ws for task, ws in zip(exp.sub_tasks, exp.sub_workspace_list)}
        for name, result in exp.strategy_results.items():
            row: dict[str, Any] = {
                "loop": index,
                "strategy": name,
                "accepted_round": bool(feedback and feedback.decision),
            }
            if not result.get("ok"):
                found.append({**row, "ran": False})
                continue
            chosen = R.selected(result)
            row["case"] = result.get("selected_case")
            for segment in ("train", "valid", "test"):
                values = chosen.get(segment, {})
                row[f"{segment}_sharpe"] = values.get("sharpe")
                row[f"{segment}_alpha_sharpe"] = values.get("alpha_sharpe")
            row["test_annual_return"] = chosen.get("test", {}).get("annual_return")
            row["test_max_drawdown"] = chosen.get("test", {}).get("max_drawdown")
            row["gates_failed"] = len(R.gate_failures(result, settings))
            row["test_deflated_sharpe"] = float("nan")
            workspace = workspaces.get(name)
            path = workspace.workspace_path / RETURNS_FILE if workspace is not None else None
            if path is not None and path.exists():
                frame = pd.read_parquet(path)
                row["test_deflated_sharpe"] = deflated(
                    frame.loc[frame["segment"] == "test", "returns"].to_numpy(), trials
                )
            found.append(row)
    return found, len(trials)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.strategy.report")
    parser.add_argument("log", type=Path, help="the session's log directory")
    parser.add_argument("--json", type=Path, help="also write the rows as JSON")
    args = parser.parse_args(argv)
    settings = CryptoStrategySettings()
    found, trials = rows(args.log, settings)
    if not found:
        print(f"No backtested strategy under {args.log}.")
        return 1
    columns = [
        "loop", "strategy", "case", "gates_failed", "accepted_round",
        "train_sharpe", "valid_sharpe", "valid_alpha_sharpe",
        "test_sharpe", "test_alpha_sharpe", "test_annual_return", "test_max_drawdown", "test_deflated_sharpe",
    ]  # fmt: skip
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in found:
        print("| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |")
    print(f"\n{len(found)} strategies, {trials} trials (every case of every strategy) in the deflated Sharpe.")
    if args.json:
        args.json.write_text(json.dumps(found, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
