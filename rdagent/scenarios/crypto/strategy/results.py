"""What is made of a strategy's evaluation: the hard gates, the summary the model sees, and what a
strategy adds to the book of strategies already accepted.

An evaluation is the dict :func:`engine.evaluate` returns (``ok`` set by the runner). The gates and
the book use train and select numbers only, so they work with the lockbox hidden.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.engine import without_lockbox

SUMMARY_METRICS = (
    "sharpe",
    "alpha_sharpe",
    "sortino",
    "calmar",
    "annual_return",
    "max_drawdown",
    "max_drawdown_recovery_bars",
    "beta_to_market",
    "hit_ratio",
    "trade_win_rate",
    "trades",
    "return_concentration",
    "annual_turnover",
    "holding_bars",
    "share_of_bars_in_market",
    "costs_over_gross_pnl",
)
"""The per-segment numbers of a strategy's summary: what the trace shows in every later prompt."""


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def selected(result: dict[str, Any]) -> dict[str, Any]:
    """The chosen case's segments: ``{"train": {...}, "select": {...}, "lockbox": {...}}``."""
    return result.get("cases", {}).get(result.get("selected_case"), {})


def gate_failures(result: dict[str, Any], settings: CryptoStrategySettings) -> list[str]:
    """The hard gates a strategy failed, each as a sentence; empty when it passed them all."""
    if not result.get("ok"):
        return ["it did not run: " + "; ".join(str(problem) for problem in result.get("problems", []))]
    failures = []
    select = _number(selected(result).get("select", {}).get("sharpe"))
    robustness = result.get("robustness", {})
    if not select > settings.min_select_sharpe:
        failures.append(f"the select Sharpe {select:.2f} is not above {settings.min_select_sharpe:.2f}")
    delayed = _number(robustness.get("select_sharpe_delayed_one_more_bar"))
    if select > 0 and not delayed >= select * (1 - settings.max_delay_loss):
        failures.append(
            f"one more bar of delay takes the select Sharpe from {select:.2f} to {delayed:.2f}, "
            f"a loss above {settings.max_delay_loss:.0%}: the edge lives in the bar it is decided on"
        )
    p_value = _number(robustness.get("permutation_p_value_train_and_select"))
    if not p_value <= settings.max_p_value:
        failures.append(
            f"the block-permutation p-value {p_value:.3f} is above {settings.max_p_value}: shuffled returns "
            "earn as much with the same positions"
        )
    ratio = _number(robustness.get("parameter_neighbourhood", {}).get("train", {}).get("ratio"))
    if settings.min_neighbour_ratio is not None and math.isfinite(ratio) and ratio < settings.min_neighbour_ratio:
        failures.append(
            f"the neighbouring parameter cases average {ratio:.0%} of the chosen case's train Sharpe, below "
            f"{settings.min_neighbour_ratio:.0%}: the chosen case is a peak, not a plateau"
        )
    return failures


def warnings(result: dict[str, Any]) -> list[str]:
    """What is worth the model's attention without rejecting the strategy."""
    if not result.get("ok"):
        return []
    found = []
    robustness = result.get("robustness", {})
    curve = {int(k): _number(v) for k, v in robustness.get("select_sharpe_by_extra_delay_before_funding", {}).items()}
    base = curve.get(0, float("nan"))
    later = max((value for delay, value in curve.items() if delay > 0 and math.isfinite(value)), default=float("nan"))
    if math.isfinite(base) and math.isfinite(later) and later > max(base, 0) + 0.25:
        found.append(
            f"the select Sharpe rises from {base:.2f} to {later:.2f} when the positions arrive later: the signal "
            "trades too early"
        )
    ratio = _number(robustness.get("parameter_neighbourhood", {}).get("train", {}).get("ratio"))
    if math.isfinite(ratio) and ratio < 0.5:
        found.append(f"the neighbouring parameter cases make {ratio:.0%} of the chosen case's train Sharpe")
    select = selected(result).get("select", {})
    concentration = _number(select.get("return_concentration"))
    if math.isfinite(concentration) and concentration > 0.5:
        found.append(f"{concentration:.0%} of the select gains came from the best 5% of bars")
    return found


def per_bar_trials(result: dict[str, Any]) -> list[float]:
    """The per-bar select Sharpe of each case: what the deflated Sharpe of later trials counts."""
    bars_per_year = _number(result.get("bars_per_year"))
    if not result.get("ok") or not bars_per_year > 0:
        return []
    found = []
    for case in result.get("cases", {}).values():
        value = _number(case.get("select", {}).get("sharpe"))
        if math.isfinite(value):
            found.append(value / math.sqrt(bars_per_year))
    return found


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    return value


def summary(result: dict[str, Any], settings: CryptoStrategySettings) -> dict[str, Any]:
    """A strategy's evaluation in a few numbers: what the trace shows in every later prompt.
    The lockbox is left out unless ``hide_lockbox`` is off."""
    if not result.get("ok"):
        return {"ran": False, "problems": [str(problem)[:500] for problem in result.get("problems", [])]}
    shown = result if not settings.hide_lockbox else without_lockbox(result)
    segments = selected(shown)
    robustness = shown.get("robustness", {})
    return {
        "case": shown.get("selected_case"),
        **{
            segment: {name: _rounded(values.get(name)) for name in SUMMARY_METRICS}
            for segment, values in segments.items()
        },
        "select_sharpe_delayed_one_more_bar": _rounded(robustness.get("select_sharpe_delayed_one_more_bar")),
        "select_sharpe_of_each_case": {
            label: _rounded(value) for label, value in robustness.get("select_sharpe_of_each_case", {}).items()
        },
        "permutation_p_value": _rounded(robustness.get("permutation_p_value_train_and_select")),
        "gates_failed": gate_failures(result, settings),
        "warnings": warnings(result),
    }


# -- the book of accepted strategies -----------------------------------------------------------


def book_frame(members: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """The members' returns side by side on one time index (missing bars are flat), and the
    train and select masks. Each member is a frame with ``returns`` and ``segment`` columns."""
    index = members[0].index
    for frame in members[1:]:
        index = index.union(frame.index)
    returns = np.column_stack(
        [frame["returns"].reindex(index).fillna(0.0).to_numpy(dtype=np.float64) for frame in members]
    )
    segment = pd.Series("", index=index, dtype=object)
    for frame in members:
        segment.loc[frame.index] = frame["segment"].astype(str).to_numpy()
    return returns, (segment == "train").to_numpy(), (segment == "select").to_numpy(), index


def contribution(book: list[pd.DataFrame], candidate: pd.DataFrame, bars_per_year: float) -> dict[str, Any]:
    """What adding the candidate does to the book: ``phasor.allocation.marginal_contribution`` on
    the select segment, each strategy scaled by its train volatility."""
    from phasor import allocation

    returns, train, select, _ = book_frame([*book, candidate])
    found = allocation.marginal_contribution(
        returns[:, :-1] if book else None, returns[:, -1], fit=train, rows=select, ann_factor=bars_per_year
    )
    return {key: (_rounded(value) if isinstance(value, float) else value) for key, value in found.items()} | {
        "correlations": [_rounded(value) for value in found["correlations"]]
    }


def accepts(found: dict[str, Any], book_size: int, settings: CryptoStrategySettings) -> list[str]:
    """Why a strategy that passed the gates is still not added to the book; empty when it is new."""
    if book_size == 0:
        return []
    reasons = []
    correlation = _number(found.get("max_abs_correlation"))
    if math.isfinite(correlation) and correlation > settings.max_correlation:
        reasons.append(
            f"its select returns correlate {correlation:.2f} with an accepted strategy (limit "
            f"{settings.max_correlation:.2f}): it is a copy of what the book already holds"
        )
    gain = _number(found.get("gain"))
    if not gain >= settings.min_portfolio_gain:
        reasons.append(
            f"adding it changes the book's select Sharpe by {gain:+.2f}, less than {settings.min_portfolio_gain:+.2f}"
        )
    return reasons
