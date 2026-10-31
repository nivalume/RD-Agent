"""What is made of a strategy's evaluation: the hard gates, the score, and what the model sees.

An evaluation is the dict :func:`engine.evaluate` returns (``ok`` set by the runner). The gates
use train and valid numbers only, so they can be applied with the test segment hidden.
"""

from __future__ import annotations

import math
from typing import Any

from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.engine import without_test

SUMMARY_METRICS = (
    "sharpe",
    "alpha_sharpe",
    "annual_return",
    "max_drawdown",
    "beta_to_market",
    "annual_turnover",
    "holding_bars",
    "share_of_bars_in_market",
    "costs_over_gross_pnl",
)
"""The per-segment numbers of a strategy's one-line summary."""


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def selected(result: dict[str, Any]) -> dict[str, Any]:
    """The chosen case's segments: ``{"train": {...}, "valid": {...}, "test": {...}}``."""
    return result.get("cases", {}).get(result.get("selected_case"), {})


def score(result: dict[str, Any]) -> float:
    """What ranks strategies: the chosen case's valid alpha Sharpe, after costs and funding."""
    value = _number(selected(result).get("valid", {}).get("alpha_sharpe"))
    return value if math.isfinite(value) else float("-inf")


def gate_failures(result: dict[str, Any], settings: CryptoStrategySettings) -> list[str]:
    """The hard gates a strategy failed, each as a sentence; empty when it passed them all."""
    if not result.get("ok"):
        return ["it did not run: " + "; ".join(str(problem) for problem in result.get("problems", []))]
    failures = []
    valid = _number(selected(result).get("valid", {}).get("sharpe"))
    robustness = result.get("robustness", {})
    if not valid > settings.min_valid_sharpe:
        failures.append(f"the valid Sharpe {valid:.2f} is not above {settings.min_valid_sharpe:.2f}")
    delayed = _number(robustness.get("valid_sharpe_delayed_one_more_bar"))
    if valid > 0 and not delayed >= valid * (1 - settings.max_delay_loss):
        failures.append(
            f"one more bar of delay takes the valid Sharpe from {valid:.2f} to {delayed:.2f}, "
            f"a loss above {settings.max_delay_loss:.0%}: the edge lives in the bar it is decided on"
        )
    p_value = _number(robustness.get("permutation_p_value_train_and_valid"))
    if not p_value <= settings.max_p_value:
        failures.append(
            f"the block-permutation p-value {p_value:.3f} is above {settings.max_p_value}: shuffled returns "
            "earn as much with the same positions"
        )
    return failures


def per_bar_trials(result: dict[str, Any]) -> list[float]:
    """The per-bar valid Sharpe of each case: what the deflated Sharpe of later trials counts."""
    bars_per_year = _number(result.get("bars_per_year"))
    if not result.get("ok") or not bars_per_year > 0:
        return []
    found = []
    for case in result.get("cases", {}).values():
        value = _number(case.get("valid", {}).get("sharpe"))
        if math.isfinite(value):
            found.append(value / math.sqrt(bars_per_year))
    return found


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    return value


def summary(result: dict[str, Any], settings: CryptoStrategySettings) -> dict[str, Any]:
    """A strategy's evaluation in a few numbers: what the trace shows in every later prompt.
    The test segment is left out unless ``hide_test`` is off."""
    if not result.get("ok"):
        return {"ran": False, "problems": [str(problem)[:500] for problem in result.get("problems", [])]}
    shown = result if not settings.hide_test else without_test(result)
    segments = selected(shown)
    robustness = shown.get("robustness", {})
    return {
        "case": shown.get("selected_case"),
        **{
            segment: {name: _rounded(values.get(name)) for name in SUMMARY_METRICS}
            for segment, values in segments.items()
        },
        "valid_sharpe_delayed_one_more_bar": _rounded(robustness.get("valid_sharpe_delayed_one_more_bar")),
        "valid_sharpe_of_each_case": {
            label: _rounded(value) for label, value in robustness.get("valid_sharpe_of_each_case", {}).items()
        },
        "permutation_p_value": _rounded(robustness.get("permutation_p_value_train_and_valid")),
        "gates_failed": gate_failures(result, settings),
    }
