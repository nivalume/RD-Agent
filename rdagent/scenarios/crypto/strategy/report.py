"""Every strategy a loop tested and the book of accepted strategies: on train and select during the
research, and on the lockbox once, at its end.

    python -m rdagent.scenarios.crypto.strategy.report log/<session>                  # train and select
    python -m rdagent.scenarios.crypto.strategy.report log/<session> --open-lockbox   # and the lockbox

Without ``--open-lockbox`` nothing of the lockbox is shown. Opening it writes
``<log>/lockbox_opened.json``; a later run says it was opened, because a strategy chosen after
seeing the lockbox has no out-of-sample period left until new data arrives.

Beside the backtests: the probability of backtest overfitting (PBO, by CSCV over every trial's
select-period returns: how often the trial that is best on one half of the blocks is below the
median on the other), and an expanding walk-forward of each strategy in the book (each quarter's
parameter case chosen on the quarters before it, within train and select). The lockbox Sharpe is
deflated by every trial of the session. The HTML page (``<log>/strategy_report.html`` by default)
draws each strategy's value and drawdown with the periods shaded, its walk-forward, delay curve
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

from rdagent.scenarios.crypto.strategy import engine
from rdagent.scenarios.crypto.strategy import results as R
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.run import RETURNS_FILE, SPEC_FILE

MARKER = "lockbox_opened.json"
SEEN = ("train", "select")

COLUMNS = [
    "loop", "strategy", "case", "gates_failed", "in_book", "max_corr_to_book",
    "train_sharpe", "select_sharpe", "select_alpha_sharpe", "walk_forward_sharpe",
]  # fmt: skip
LOCKBOX_COLUMNS = [
    "lockbox_sharpe", "lockbox_alpha_sharpe", "lockbox_sortino", "lockbox_calmar", "lockbox_annual_return",
    "lockbox_max_drawdown", "lockbox_trade_win_rate", "lockbox_deflated_sharpe",
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


def _spec(exp: Any, name: str) -> Any:
    from rdagent.components.coder.phasor_coder.spec import validate_factor_spec

    for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
        if task.strategy_name == name and workspace is not None and workspace.file_dict.get(SPEC_FILE):
            return validate_factor_spec(
                json.loads(workspace.file_dict[SPEC_FILE]), available_fields=list(engine.FIELDS)
            )
    return None


def walk_forward(
    spec: Any, panel: engine.Panel, settings: CryptoStrategySettings, freq: str = "QE", min_periods: int = 4
) -> dict[str, Any]:
    """Each period (a quarter) of train and select traded with the parameter case that had the best
    Sharpe over every period before it (at least ``min_periods``), stitched into one series that is
    out of sample at every point."""
    from phasor import validation

    found = engine.positions(spec, panel)
    runs = {label: engine.backtest(raw, panel, settings.backtest()).returns for label, raw in found.items()}
    rows = settings.segments().rows(panel)
    seen = np.flatnonzero(rows["train"] | rows["select"])
    times = pd.to_datetime(panel.index[seen], unit="ns")
    pieces, index, choices = [], [], []
    for split in validation.calendar(times.to_numpy(), freq, train=min_periods, test=1, expanding=True):
        fit, trade = seen[split.train], seen[split.test]
        label = max(runs, key=lambda case: engine._finite(_sharpe(runs[case][fit], panel.bars_per_year)))
        pieces.append(runs[label][trade])
        index.append(times[split.test])
        choices.append((str(times[split.test][0].date()), label))
    if not pieces:
        return {}
    returns = pd.Series(np.concatenate(pieces), index=pd.DatetimeIndex(np.concatenate(index)).tz_localize("UTC"))
    return {
        "returns": returns,
        "sharpe": _sharpe(returns.to_numpy(), panel.bars_per_year),
        "choices": choices,
        "cases": len(found),
    }


def pbo(experiments: list[tuple[int, Any, Any]]) -> dict[str, Any]:
    """CSCV over the select-period returns of every trial that ran: the probability that the
    trial best on half of the blocks is at or below the median on the other half."""
    import phasor

    members = [frame for _, exp, _ in experiments for frame in (getattr(exp, "strategy_returns", None) or {}).values()]
    if len(members) < 2:
        return {"trials": len(members), "pbo": float("nan")}
    returns, _, select, _ = R.book_frame(members)
    perf = returns[select]
    splits = min(16, (len(perf) // 2) * 2)
    value = float(phasor.robust.pbo_cscv(np.asfortranarray(perf), n_splits=splits)) if splits >= 2 else float("nan")
    return {"trials": len(members), "pbo": value, "splits": splits}


def collect(log: Path, settings: CryptoStrategySettings, open_lockbox: bool = False) -> dict[str, Any]:
    """The rows of the table, every strategy's returns, the book, the trial count, PBO and the
    walk-forward of the book's strategies. Lockbox numbers only with ``open_lockbox``."""
    experiments = _experiments(log)
    trials = [value for _, exp, _ in experiments for value in (getattr(exp, "trial_sharpes", None) or [])]
    rows, curves, book, bars_per_year = [], {}, [], float("nan")
    panel = None
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
            for segment in ("train", "select", "lockbox") if open_lockbox else SEEN:
                values = chosen.get(segment, {})
                row[f"{segment}_sharpe"] = values.get("sharpe")
                row[f"{segment}_alpha_sharpe"] = values.get("alpha_sharpe")
            if open_lockbox:
                lockbox = chosen.get("lockbox", {})
                for key in ("sortino", "calmar", "annual_return", "max_drawdown", "trade_win_rate"):
                    row[f"lockbox_{key}"] = lockbox.get(key)
            row["gates_failed"] = len(R.gate_failures(result, settings))
            row["max_corr_to_book"] = judged.get(name, {}).get("contribution", {}).get("max_abs_correlation")
            frame = _returns_file(exp, name)
            if frame is not None:
                if open_lockbox:
                    row["lockbox_deflated_sharpe"] = deflated(
                        frame.loc[frame["segment"] == "lockbox", "returns"].to_numpy(), trials
                    )
                else:
                    frame = frame[frame["segment"].isin(SEEN)]
                curves[f"{index}:{name}"] = [frame, result, None]
                if row["in_book"]:
                    book.append((name, frame))
            if row["in_book"]:
                spec = _spec(exp, name)
                try:
                    panel = panel or engine.Panel.load(settings.panel_path)
                    found = walk_forward(spec, panel, settings) if spec is not None else {}
                except (OSError, ValueError, engine.StrategyError) as error:
                    found = {"error": str(error)}
                row["walk_forward_sharpe"] = found.get("sharpe")
                if f"{index}:{name}" in curves:
                    curves[f"{index}:{name}"][2] = found
            rows.append(row)
    found: dict[str, Any] = {
        "rows": rows,
        "trials": len(trials),
        "curves": curves,
        "bars_per_year": bars_per_year,
        "pbo": pbo(experiments),
        "lockbox": open_lockbox,
    }
    if book:
        found["book"] = _book(book, trials, bars_per_year, open_lockbox)
    return found


def _book(
    members: list[tuple[str, pd.DataFrame]], trials: list[float], bars_per_year: float, open_lockbox: bool = False
) -> dict[str, Any]:
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
    for name in ("train", "select", "lockbox") if open_lockbox else SEEN:
        part = returns[(segment == name).to_numpy()]
        out[f"{name}_sharpe"] = _sharpe(part, bars_per_year)
    if open_lockbox:
        out["lockbox_deflated_sharpe"] = deflated(returns[(segment == "lockbox").to_numpy()], trials)
    return out


def _segments(segment: pd.Series) -> dict[str, tuple[int, int]]:
    values = segment.astype(str).to_numpy()
    found = {}
    for name in ("train", "select", "lockbox"):
        rows = np.flatnonzero(values == name)
        if rows.size:
            found[name] = (int(rows[0]), int(rows[-1]) + 1)
    return found


def _columns(found: dict[str, Any]) -> list[str]:
    return COLUMNS + (LOCKBOX_COLUMNS if found["lockbox"] else [])


def _book_line(book: dict[str, Any]) -> str:
    line = f"Sharpe train {_cell(book['train_sharpe'])}, select {_cell(book['select_sharpe'])}"
    if "lockbox_sharpe" in book:
        line += f", lockbox {_cell(book['lockbox_sharpe'])}, lockbox deflated {_cell(book['lockbox_deflated_sharpe'])}"
    return line


def write_html(found: dict[str, Any], path: Path) -> None:
    """The charts, one section a strategy and one for the book."""
    import phasor.plotting as plotting

    parts: list[str] = []
    first = True

    def add(figure: Any) -> None:
        nonlocal first
        parts.append(figure.to_html(full_html=False, include_plotlyjs=first))
        first = False

    columns = _columns(found)
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in found["rows"]
    )
    parts.append(
        "<h1>Strategy report</h1>"
        f"<p>{len(found['rows'])} strategies, {found['trials']} trials. PBO over the select period: "
        f"{_cell(found['pbo']['pbo'])} ({found['pbo']['trials']} strategies). "
        + ("The lockbox is open." if found["lockbox"] else "The lockbox is closed: train and select only.")
        + "</p><table border=1 cellspacing=0 cellpadding=3><tr>"
        + "".join(f"<th>{column}</th>" for column in columns)
        + f"</tr>{table}</table>"
    )
    if "book" in found:
        book = found["book"]
        parts.append(f"<h2>Book: {html.escape(', '.join(book['members']))}</h2><p>{_book_line(book)}</p>")
        add(plotting.returns(book["returns"].to_numpy(), index=book["returns"].index, segments=_segments(book["segment"]),
                             title="book", names=["book"]))  # fmt: skip
    for label, (frame, result, forward) in found["curves"].items():
        parts.append(f"<h2>{html.escape(label)} ({html.escape(str(result.get('selected_case')))})</h2>")
        add(plotting.returns(frame["returns"].to_numpy(), index=frame.index, segments=_segments(frame["segment"]),
                             title=label, names=[label]))  # fmt: skip
        if forward and "returns" in forward:
            parts.append(
                f"<p>Walk-forward: Sharpe {_cell(forward['sharpe'])} over {len(forward['choices'])} quarters, "
                f"{forward['cases']} parameter cases; chosen: "
                + html.escape(", ".join(f"{start} {case}" for start, case in forward["choices"]))
                + "</p>"
            )
            add(plotting.returns(forward["returns"].to_numpy(), index=forward["returns"].index,
                                 title=f"{label}: walk-forward", names=["walk-forward"]))  # fmt: skip
        robustness = result.get("robustness", {})
        curve = robustness.get("select_sharpe_by_extra_delay_before_funding") or {}
        if curve:
            add(plotting.bars(curve, title="select Sharpe by extra delay (bars)"))
        quintiles = result.get("diagnostics", {}).get("select", {}).get("next_return_by_position_quintile") or {}
        if quintiles:
            add(plotting.bars(quintiles, title="select: next-bar return by position quintile"))
    path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>Strategy report</title></head><body>"
                    + "\n".join(parts) + "</body></html>")  # fmt: skip


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


def _marker(log: Path, found: dict[str, Any], settings: CryptoStrategySettings) -> dict[str, Any] | None:
    """The lockbox marker: read it, and write it the first time the lockbox is opened."""
    from datetime import datetime, timezone

    path = log / MARKER
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = None
    if found["lockbox"] and previous is None:
        path.write_text(
            json.dumps(
                {
                    "opened": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "lockbox_start": settings.lockbox_start,
                    "lockbox_end": settings.lockbox_end,
                    "strategies": [f"{row['loop']}:{row['strategy']}" for row in found["rows"]],
                },
                indent=1,
            )
        )
    return previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.strategy.report")
    parser.add_argument("log", type=Path, help="the session's log directory")
    parser.add_argument("--open-lockbox", action="store_true", help="show the lockbox period (once, at the end)")
    parser.add_argument("--json", type=Path, help="also write the rows as JSON")
    parser.add_argument("--html", type=Path, help="the charts (default: <log>/strategy_report.html)")
    args = parser.parse_args(argv)
    settings = CryptoStrategySettings()
    found = collect(args.log, settings, open_lockbox=args.open_lockbox)
    if not found["rows"]:
        print(f"No backtested strategy under {args.log}.")
        return 1
    previous = _marker(args.log, found, settings)
    if previous is not None:
        print(
            f"The lockbox was opened on {previous.get('opened')}: a strategy chosen after that has no out-of-sample "
            f"period left; the next one is the data after {previous.get('lockbox_end') or 'that date'}."
        )
    columns = _columns(found)
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in found["rows"]:
        print("| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |")
    print(
        f"\n{len(found['rows'])} strategies, {found['trials']} trials (every case of every strategy). "
        f"PBO over the select period: {_cell(found['pbo']['pbo'])}."
    )
    if not found["lockbox"]:
        print("The lockbox is closed; --open-lockbox shows it, once the research is done.")
    if "book" in found:
        book = found["book"]
        print(f"Book ({', '.join(book['members'])}): {_book_line(book)}")
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
