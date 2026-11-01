"""Every strategy a loop tested, on all three segments, and the book of accepted strategies, with
the test Sharpe deflated by the trials.

    python -m rdagent.scenarios.crypto.strategy.report log/<session> [--json out.json] [--html out.html]

This is the only place the test segment is shown. The deflated Sharpe counts every trial of the
session (each case of each strategy, accepted or not): it is the probability that the true Sharpe
is above zero after the search that found it. The HTML page (``<log>/strategy_report.html`` by
default) draws each strategy's value and drawdown with the three segments shaded, its delay curve
and position quintiles, and the same for the book.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdagent.scenarios.crypto.strategy import results as R
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.run import RETURNS_FILE

COLUMNS = [
    "loop", "strategy", "case", "gates_failed", "in_book", "max_corr_to_book",
    "train_sharpe", "valid_sharpe", "valid_alpha_sharpe",
    "test_sharpe", "test_alpha_sharpe", "test_sortino", "test_calmar", "test_annual_return",
    "test_max_drawdown", "test_trade_win_rate", "test_deflated_sharpe",
]  # fmt: skip


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


def _returns_file(exp: Any, name: str) -> pd.DataFrame | None:
    for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
        if task.strategy_name == name and workspace is not None:
            path = workspace.workspace_path / RETURNS_FILE
            if path.exists():
                return pd.read_parquet(path)
    return None


def _sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    import phasor

    value = (
        float(phasor.returns.metric(returns, "sharpe_ratio", ann_factor=bars_per_year))
        if returns.size > 1
        else math.nan
    )
    return value if math.isfinite(value) else float("nan")


def collect(log: Path, settings: CryptoStrategySettings) -> dict[str, Any]:
    """The rows of the table, every strategy's returns, the book and the trial count."""
    experiments = _experiments(log)
    trials = [value for _, exp, _ in experiments for value in (getattr(exp, "trial_sharpes", None) or [])]
    rows, curves, book, bars_per_year = [], {}, [], float("nan")
    for index, exp, feedback in experiments:
        accepted = getattr(exp, "accepted_strategy", None) if feedback and feedback.decision else None
        judged = getattr(exp, "judged", None) or {}
        for name, result in exp.strategy_results.items():
            row: dict[str, Any] = {"loop": index, "strategy": name, "in_book": name == accepted}
            if not result.get("ok"):
                rows.append({**row, "ran": False})
                continue
            bars_per_year = float(result.get("bars_per_year", bars_per_year))
            chosen = R.selected(result)
            row["case"] = result.get("selected_case")
            for segment in ("train", "valid", "test"):
                values = chosen.get(segment, {})
                row[f"{segment}_sharpe"] = values.get("sharpe")
                row[f"{segment}_alpha_sharpe"] = values.get("alpha_sharpe")
            test = chosen.get("test", {})
            for key, column in (("sortino", "test_sortino"), ("calmar", "test_calmar"),
                                ("annual_return", "test_annual_return"), ("max_drawdown", "test_max_drawdown"),
                                ("trade_win_rate", "test_trade_win_rate")):  # fmt: skip
                row[column] = test.get(key)
            row["gates_failed"] = len(R.gate_failures(result, settings))
            row["max_corr_to_book"] = judged.get(name, {}).get("contribution", {}).get("max_abs_correlation")
            row["test_deflated_sharpe"] = float("nan")
            frame = _returns_file(exp, name)
            if frame is not None:
                row["test_deflated_sharpe"] = deflated(
                    frame.loc[frame["segment"] == "test", "returns"].to_numpy(), trials
                )
                curves[f"{index}:{name}"] = (frame, result)
                if row["in_book"]:
                    book.append((name, frame))
            rows.append(row)
    found: dict[str, Any] = {"rows": rows, "trials": len(trials), "curves": curves, "bars_per_year": bars_per_year}
    if book:
        found["book"] = _book(book, trials, bars_per_year)
    return found


def _book(members: list[tuple[str, pd.DataFrame]], trials: list[float], bars_per_year: float) -> dict[str, Any]:
    """The accepted strategies scaled to their train volatility and equally weighted, on every segment."""
    index = members[0][1].index
    for _, frame in members[1:]:
        index = index.union(frame.index)
    segment = members[0][1]["segment"].reindex(index).fillna("")
    for _, frame in members[1:]:
        segment = segment.where(segment != "", frame["segment"].reindex(index).fillna(""))
    raw, scales = [], []
    for _, frame in members:
        values = frame["returns"].reindex(index).fillna(0.0).to_numpy(dtype=np.float64)
        raw.append(values)
        sd = values[(segment == "train").to_numpy()].std(ddof=1)
        scales.append(sd if sd > 0 else np.nan)
    # Each member at the same train risk -- the members' average, so the book's returns stay returns.
    level = float(np.nanmean(scales)) if np.isfinite(scales).any() else 1.0
    returns = np.mean([values * (level / sd if np.isfinite(sd) else 1.0) for values, sd in zip(raw, scales)], axis=0)
    out = {"members": [name for name, _ in members], "returns": pd.Series(returns, index=index), "segment": segment}
    for name in ("train", "valid", "test"):
        part = returns[(segment == name).to_numpy()]
        out[f"{name}_sharpe"] = _sharpe(part, bars_per_year)
    out["test_deflated_sharpe"] = deflated(returns[(segment == "test").to_numpy()], trials)
    return out


def _segments(segment: pd.Series) -> dict[str, tuple[int, int]]:
    values = segment.astype(str).to_numpy()
    found = {}
    for name in ("train", "valid", "test"):
        rows = np.flatnonzero(values == name)
        if rows.size:
            found[name] = (int(rows[0]), int(rows[-1]) + 1)
    return found


def write_html(found: dict[str, Any], path: Path) -> None:
    """The charts, one section a strategy and one for the book."""
    import phasor.plotting as plotting

    parts: list[str] = []
    first = True

    def add(figure: Any) -> None:
        nonlocal first
        parts.append(figure.to_html(full_html=False, include_plotlyjs=first))
        first = False

    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell(row.get(column, '')))}</td>" for column in COLUMNS) + "</tr>"
        for row in found["rows"]
    )
    parts.append(
        "<h1>Strategy report</h1>"
        f"<p>{len(found['rows'])} strategies, {found['trials']} trials in the deflated Sharpe.</p>"
        "<table border=1 cellspacing=0 cellpadding=3><tr>"
        + "".join(f"<th>{column}</th>" for column in COLUMNS)
        + f"</tr>{table}</table>"
    )
    if "book" in found:
        book = found["book"]
        parts.append(
            f"<h2>Book: {html.escape(', '.join(book['members']))}</h2><p>Sharpe train {_cell(book['train_sharpe'])}, "
            f"valid {_cell(book['valid_sharpe'])}, test {_cell(book['test_sharpe'])}; test deflated Sharpe "
            f"{_cell(book['test_deflated_sharpe'])}</p>"
        )
        add(plotting.returns(book["returns"].to_numpy(), index=book["returns"].index, segments=_segments(book["segment"]),
                             title="book", names=["book"]))  # fmt: skip
    for label, (frame, result) in found["curves"].items():
        parts.append(f"<h2>{html.escape(label)} ({html.escape(str(result.get('selected_case')))})</h2>")
        add(plotting.returns(frame["returns"].to_numpy(), index=frame.index, segments=_segments(frame["segment"]),
                             title=label, names=[label]))  # fmt: skip
        robustness = result.get("robustness", {})
        curve = robustness.get("valid_sharpe_by_extra_delay_before_funding") or {}
        if curve:
            add(plotting.bars(curve, title="valid Sharpe by extra delay (bars)"))
        quintiles = result.get("diagnostics", {}).get("valid", {}).get("next_return_by_position_quintile") or {}
        if quintiles:
            add(plotting.bars(quintiles, title="valid: next-bar return by position quintile"))
    path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>Strategy report</title></head><body>"
                    + "\n".join(parts) + "</body></html>")  # fmt: skip


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.strategy.report")
    parser.add_argument("log", type=Path, help="the session's log directory")
    parser.add_argument("--json", type=Path, help="also write the rows as JSON")
    parser.add_argument("--html", type=Path, help="the charts (default: <log>/strategy_report.html)")
    args = parser.parse_args(argv)
    found = collect(args.log, CryptoStrategySettings())
    if not found["rows"]:
        print(f"No backtested strategy under {args.log}.")
        return 1
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in found["rows"]:
        print("| " + " | ".join(_cell(row.get(column, "")) for column in COLUMNS) + " |")
    print(
        f"\n{len(found['rows'])} strategies, {found['trials']} trials (every case of every strategy) in the deflated Sharpe."
    )
    if "book" in found:
        book = found["book"]
        print(
            f"Book ({', '.join(book['members'])}): Sharpe train {_cell(book['train_sharpe'])}, valid "
            f"{_cell(book['valid_sharpe'])}, test {_cell(book['test_sharpe'])}, test deflated {_cell(book['test_deflated_sharpe'])}"
        )
    target = args.html or args.log / "strategy_report.html"
    try:
        write_html(found, target)
        print(f"Charts: {target}")
    except ImportError as error:
        print(f"No charts ({error}).")
    if args.json:
        args.json.write_text(json.dumps(found["rows"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
