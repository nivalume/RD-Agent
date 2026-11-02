"""The strategy engine: a panel, the positions a strategy takes on it, and their backtest.

Everything that runs a strategy is here, as plain functions on numpy arrays, so the coder's checks,
the runner and the tests run the same code:

* :class:`Panel` -- the wide tables ``jarvis-data panel`` writes: ``(T, S)`` arrays of bars and
  funding, the bar close times, the symbols and the top-N pool mask;
* :func:`positions` -- a strategy spec (a phasor expression, or numpy code) evaluated into target
  positions in [-1, 1], one array per parameter case;
* :func:`check` -- what the coder must get right before anything is backtested: shapes, bounds,
  not flat, and no look-ahead (recomputed on truncated panels, the values must not move);
* :func:`backtest` -- the positions, in the pool only, executed one bar later, scaled to a target
  volatility per instrument and equally weighted, run through ``phasor.from_orders`` with fees,
  slippage and funding on one shared account;
* :func:`evaluate` -- the backtest of every case on every segment, with robustness statistics and
  the diagnostics that test a strategy's prediction.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdagent.components.coder.phasor_coder.spec import FactorSpec

FIELDS = ("open", "high", "low", "close", "volume", "amount", "count", "taker_buy_amount", "funding_rate")
"""The fields of a panel a strategy may read."""

TOLERANCE = 1e-9


# -- the panel ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    index: np.ndarray  # (T,) int64 UTC nanoseconds: the close of each row's bars
    symbols: tuple[str, ...]
    fields: dict[str, np.ndarray]  # name -> (T, S) float64
    pool: np.ndarray  # (T, S) bool: in the top-N pool

    @classmethod
    def load(cls, directory: Path | str) -> "Panel":
        directory = Path(directory).expanduser()
        close = pd.read_parquet(directory / "close.parquet")
        symbols = tuple(str(name) for name in close.columns)
        fields = {}
        for name in FIELDS:
            frame = pd.read_parquet(directory / f"{name}.parquet").reindex(index=close.index, columns=close.columns)
            fields[name] = np.asfortranarray(frame.to_numpy(dtype=np.float64))
        pool = pd.read_parquet(directory / "pool.parquet").reindex(index=close.index, columns=close.columns)
        index = pd.DatetimeIndex(close.index).asi8.astype(np.int64)
        return cls(index, symbols, fields, pool.fillna(False).to_numpy(dtype=bool))

    @property
    def shape(self) -> tuple[int, int]:
        return self.fields["close"].shape

    @property
    def bars_per_year(self) -> float:
        step = float(np.median(np.diff(self.index))) if len(self.index) > 1 else 86_400e9
        return 365 * 86_400e9 / step

    def head(self, rows: int) -> "Panel":
        """The first `rows` rows: the panel as it was known at the close of row `rows - 1`."""
        return Panel(
            self.index[:rows],
            self.symbols,
            {name: np.asfortranarray(values[:rows]) for name, values in self.fields.items()},
            self.pool[:rows],
        )


# -- positions ---------------------------------------------------------------------------------


class StrategyError(Exception):
    """The strategy cannot be evaluated: it does not compile, raises, or returns the wrong shape."""


def _readonly(panel: Panel) -> dict[str, np.ndarray]:
    found = {}
    for name, values in panel.fields.items():
        view = values.view()
        view.flags.writeable = False
        found[name] = view
    return found


def positions(spec: FactorSpec, panel: Panel) -> dict[str, np.ndarray]:
    """The target positions of every case of the spec, by case label (``default`` without
    parameters): ``(T, S)`` arrays, NaN where the strategy has no opinion. Not clipped: the check
    reports a value outside [-1, 1] instead."""
    import phasor

    rows, columns = panel.shape
    if spec.mode == "expression":
        assert spec.expression is not None
        try:
            if spec.params:
                grid = phasor.FactorGrid.compile({"position": spec.expression}, **spec.params)
                values = np.asarray(grid.evaluate(_readonly(panel))["position"], dtype=np.float64)
                return {
                    _label(case): values[:, index * columns : (index + 1) * columns]
                    for index, case in enumerate(grid.cases)
                }
            expression = phasor.Expression.compile(spec.expression)
            return {"default": np.asarray(expression.evaluate(_readonly(panel)), dtype=np.float64)}
        except Exception as error:  # phasor raises its own errors for a bad expression
            raise StrategyError(f"The expression does not evaluate: {type(error).__name__}: {error}") from error
    assert spec.code is not None
    namespace: dict[str, Any] = {"__name__": "strategy"}
    try:
        exec(compile(spec.code, "strategy.py", "exec"), namespace)  # noqa: S102 - the coder's own code
        calculate = namespace["calculate"]
        values = calculate(_readonly(panel), panel.index.copy(), panel.symbols)
    except Exception as error:
        raise StrategyError(f"calculate() failed: {type(error).__name__}: {error}") from error
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows, columns):
        raise StrategyError(f"calculate() returned shape {values.shape}, not {(rows, columns)}")
    return {"default": values}


def _label(case: dict[str, Any]) -> str:
    return ",".join(f"{name}={value}" for name, value in case.items()) or "default"


def uses_future(spec: FactorSpec) -> bool:
    """Whether phasor knows the expression reads a later row; python code is checked by truncation."""
    if spec.mode != "expression" or spec.expression is None:
        return False
    import phasor

    try:
        if spec.params:
            return bool(phasor.FactorGrid.compile({"position": spec.expression}, **spec.params).uses_future)
        return bool(phasor.Expression.compile(spec.expression).uses_future)
    except Exception:
        return False


# -- the coder's checks ------------------------------------------------------------------------


@dataclass
class CheckResult:
    problems: list[str] = field(default_factory=list)
    cases: int = 0
    seconds: float = 0.0
    in_market: float = 0.0  # share of pool rows with a non-zero position
    mean_abs_position: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOLERANCE * (1 + np.abs(b)))


def check(spec: FactorSpec, panel: Panel, cuts: int = 3, seed: int = 0, time_limit: float = 120.0) -> CheckResult:
    """What must hold before a strategy is backtested. Every problem found is reported."""
    result = CheckResult()
    started = time.perf_counter()
    if uses_future(spec):
        result.problems.append("The expression reads a later row (a negative Ref or a backward fill).")
        return result
    try:
        found = positions(spec, panel)
    except StrategyError as error:
        result.problems.append(str(error))
        return result
    result.seconds = time.perf_counter() - started
    result.cases = len(found)
    if result.seconds > time_limit:
        result.problems.append(f"It took {result.seconds:.0f}s on the debug panel; the limit is {time_limit:.0f}s.")
    for label, values in found.items():
        name = "" if label == "default" else f" (case {label})"
        if np.isinf(values).any():
            result.problems.append(f"The positions{name} hold an infinite value; return NaN or a bounded number.")
            continue
        finite = values[np.isfinite(values)]
        if finite.size and (finite.max() > 1 + TOLERANCE or finite.min() < -1 - TOLERANCE):
            result.problems.append(
                f"The positions{name} range from {finite.min():.3g} to {finite.max():.3g}; a position is a "
                "fraction of the instrument's risk budget in [-1, 1]: Clip it."
            )
        held = np.nan_to_num(values) * panel.pool
        if not np.any(np.abs(held) > TOLERANCE):
            result.problems.append(f"The positions{name} are flat (0 or NaN) on every bar of the pool.")
    first = next(iter(found.values()))
    pool_rows = panel.pool.sum()
    if pool_rows:
        held = np.nan_to_num(first) * panel.pool
        result.in_market = float((np.abs(held) > TOLERANCE).sum() / pool_rows)
        result.mean_abs_position = float(np.abs(held).sum() / pool_rows)
    rows = panel.shape[0]
    generator = np.random.default_rng(seed)
    candidates = np.arange(max(rows // 2, 2), rows)
    for cut in sorted(generator.choice(candidates, size=min(cuts, len(candidates)), replace=False)):
        try:
            truncated = positions(spec, panel.head(int(cut)))
        except StrategyError as error:
            result.problems.append(f"On the panel cut after row {cut} it fails: {error}")
            break
        for label, values in found.items():
            same = _same(truncated[label], values[:cut])
            if not same.all():
                row, column = np.argwhere(~same)[0]
                result.problems.append(
                    f"Look-ahead: computed on the rows up to {cut - 1} only, the position at row {row} of "
                    f"{panel.symbols[column]} is {truncated[label][row, column]:.6g}, and "
                    f"{values[row, column]:.6g} on the whole panel. Row t may read rows <= t only."
                )
                break
        else:
            continue
        break
    return result


# -- the backtest ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestSettings:
    target_vol: float = 0.20  # annualised volatility each instrument is scaled to
    max_leverage: float = 3.0  # most a single instrument's scaling may reach
    vol_lookback: int = 180  # bars of returns that estimate an instrument's volatility
    fee: float = 0.0005
    slippage: float = 0.0002
    delay: int = 1  # bars between the close a position is decided on and the close it trades at


@dataclass
class Backtest:
    returns: np.ndarray  # (T,) the account's return over each bar
    weights: np.ndarray  # (T, S) the weights set at the close of each bar (after delay)
    asset_returns: np.ndarray  # (T, S) close-to-close returns
    funding: np.ndarray  # (T,) funding paid (negative) or received over each bar, as a return
    costs: np.ndarray  # (T,) fees and slippage of each bar, as a return
    portfolio: Any = None  # the phasor Portfolio; a trade is a round trip on one instrument
    cost_rate: float = 0.0  # fee and slippage per unit of turnover


def _ffill(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    return frame.ffill().bfill().to_numpy(dtype=np.float64)


def _shift(values: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros_like(values)
    if rows < len(values):
        out[rows:] = values[: len(values) - rows]
    return out


def backtest(raw: np.ndarray, panel: Panel, settings: BacktestSettings = BacktestSettings()) -> Backtest:
    """The positions traded: in the pool only, scaled to the target volatility, equally weighted
    over the instruments in the pool, held from `delay` bars after the decision, closed on an
    instrument's last bar, with fees, slippage and funding."""
    import phasor

    close = panel.fields["close"]
    listed = np.isfinite(close)
    decided = np.clip(np.nan_to_num(raw), -1.0, 1.0) * panel.pool
    asset_returns = np.full(close.shape, np.nan)
    asset_returns[1:] = close[1:] / close[:-1] - 1
    vol = pd.DataFrame(asset_returns).rolling(settings.vol_lookback, min_periods=settings.vol_lookback // 2).std()
    vol = vol.to_numpy() * math.sqrt(panel.bars_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(vol > 0, np.minimum(settings.target_vol / vol, settings.max_leverage), 0.0)
    scale = np.nan_to_num(scale)
    active = np.maximum(panel.pool.sum(axis=1, keepdims=True), 1)
    weights = _shift(decided * scale / active, settings.delay)
    # Nothing is held where an instrument has no price: before it lists and after it delists.
    weights = np.where(listed, weights, 0.0)
    prices = _ffill(close)
    funding = np.nan_to_num(panel.fields["funding_rate"])
    portfolio = phasor.from_orders(
        prices,
        size=weights,
        size_type="targetpercent",
        direction="both",
        fees=settings.fee,
        slippage=settings.slippage,
        funding_rate=funding,
        cash_sharing=True,
        group_lens=[prices.shape[1]],
        init_cash=1.0,
        ann_factor=panel.bars_per_year,
        trades="positions",
    )
    returns = np.nan_to_num(np.asarray(portfolio.returns(), dtype=np.float64).reshape(-1))
    held = _shift(weights, 1)  # what is held over bar t is what was set at the close of t - 1
    paid = (held * funding).sum(axis=1)
    turnover = np.abs(np.diff(np.vstack([np.zeros((1, weights.shape[1])), weights]), axis=0)).sum(axis=1)
    rate = settings.fee + settings.slippage
    return Backtest(returns, weights, asset_returns, -paid, -rate * turnover, portfolio, rate)


# -- metrics -----------------------------------------------------------------------------------


def _sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    import phasor

    if len(returns) < 2:
        return float("nan")
    value = float(phasor.returns.metric(returns, "sharpe_ratio", ann_factor=bars_per_year))
    return value if math.isfinite(value) else float("nan")


def _span(rows: np.ndarray) -> tuple[int, int]:
    """The first row of a segment and the row after its last (segments are contiguous)."""
    found = np.flatnonzero(rows)
    return int(found[0]), int(found[-1]) + 1


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def market_returns(panel: Panel) -> np.ndarray:
    """The equally weighted return of the pool over each bar (the pool of the bar before)."""
    close = panel.fields["close"]
    asset = np.full(close.shape, np.nan)
    asset[1:] = close[1:] / close[:-1] - 1
    pool = _shift(panel.pool.astype(float), 1).astype(bool) & np.isfinite(asset)
    counts = pool.sum(axis=1)
    total = np.where(pool, np.nan_to_num(asset), 0.0).sum(axis=1)
    return np.where(counts > 0, total / np.maximum(counts, 1), 0.0)


def trade_stats(trades: dict, first: int, stop: int) -> dict[str, Any]:
    """The round trips that closed in rows ``first:stop``: their count, win rate, profit factor
    (gains over losses, in cash) and the mean bars a winner and a loser was held."""
    exit_idx = np.asarray(trades["exit_idx"])
    closed = (np.asarray(trades["status"]) == 1) & (exit_idx >= first) & (exit_idx < stop)
    pnl = np.asarray(trades["pnl"], dtype=np.float64)[closed]
    duration = np.asarray(trades["duration"], dtype=np.float64)[closed]
    wins, losses = pnl > 0, pnl < 0
    lost = -pnl[losses].sum()
    return {
        "trades": int(closed.sum()),
        "trade_win_rate": float(wins.mean()) if pnl.size else float("nan"),
        "profit_factor": float(pnl[wins].sum() / lost) if lost > 0 else float("nan"),
        "mean_winning_trade_bars": float(duration[wins].mean()) if wins.any() else float("nan"),
        "mean_losing_trade_bars": float(duration[losses].mean()) if losses.any() else float("nan"),
    }


def metrics(run: Backtest, panel: Panel, rows: np.ndarray) -> dict[str, Any]:
    """The statistics of a backtest over the rows of a segment (a contiguous boolean mask).

    The return statistics are ``phasor.returns.stats`` of the account's returns against the
    equal-weight market; the trade statistics come from the phasor portfolio's round trips
    (``trades="positions"``: entry to flat on one instrument, resizing included) closed in the
    segment. The rest -- turnover, exposure,
    funding, costs, years, instruments -- phasor does not know about."""
    import phasor

    bars_per_year = panel.bars_per_year
    returns = run.returns[rows]
    if returns.size < 2:
        return {"bars": int(returns.size)}
    market = market_returns(panel)[rows]
    stats = phasor.returns.stats(returns, benchmark=market, ann_factor=bars_per_year)
    beta = _float(stats["beta"])
    alpha = returns - (0.0 if math.isnan(beta) else beta) * market
    weights = run.weights[rows]
    gross = np.abs(weights).sum(axis=1)
    turnover = np.abs(np.diff(run.weights, axis=0, prepend=0))[rows].sum(axis=1)
    contributions = (_shift(run.weights, 1) * np.nan_to_num(run.asset_returns))[rows].sum(axis=0)
    traded = np.abs(weights).sum(axis=0) > 0
    gains = returns.sum()
    years = pd.to_datetime(panel.index[rows], unit="ns", utc=True).year
    yearly = pd.Series(returns).groupby(np.asarray(years)).apply(lambda part: float(np.prod(1 + part) - 1))
    btc = panel.symbols.index("BTCUSDT") if "BTCUSDT" in panel.symbols else None
    found = {
        "bars": int(returns.size),
        "annual_return": _float(stats["annualized_return"]),
        "annual_volatility": _float(stats["annualized_volatility"]),
        "sharpe": _sharpe(returns, bars_per_year),
        "sortino": _float(stats["sortino_ratio"]),
        "calmar": _float(stats["calmar_ratio"]),
        "omega": _float(stats["omega_ratio"]),
        "max_drawdown": _float(stats["max_drawdown"]),
        "max_drawdown_duration_bars": _float(stats["max_drawdown_duration"]),
        "max_drawdown_recovery_bars": _float(stats["max_drawdown_recovery"]),
        "hit_ratio": _float(stats["hit_ratio"]),
        "return_concentration": _float(stats["return_concentration"]),
        "beta_to_market": beta,
        "alpha_sharpe": _sharpe(alpha, bars_per_year),
        "annual_turnover": float(turnover.mean() * bars_per_year),
        "mean_gross_exposure": float(gross.mean()),
        "mean_net_exposure": float(weights.sum(axis=1).mean()),
        "holding_bars": float(gross.mean() / turnover.mean()) if turnover.mean() > 0 else float("nan"),
        "share_of_bars_in_market": float((gross > 0).mean()),
        "costs_over_gross_pnl": (
            float(-run.costs[rows].sum() / abs(gains - run.costs[rows].sum()))
            if gains - run.costs[rows].sum() != 0
            else float("nan")
        ),
        "funding_return": float(run.funding[rows].sum()),
        "instruments_traded": int(traded.sum()),
        "instruments_profitable": float((contributions[traded] > 0).mean()) if traded.any() else float("nan"),
        "yearly_return": {str(year): value for year, value in yearly.items()},
    }
    if run.portfolio is not None:
        found.update(trade_stats(run.portfolio.trades(), *_span(rows)))
    if btc is not None:
        found["beta_to_btc"] = _float(
            phasor.returns.metric(returns, "beta", benchmark=np.nan_to_num(run.asset_returns[rows, btc]))
        )
    return found


# -- robustness and diagnostics ----------------------------------------------------------------


def permutation_p_value(run: Backtest, rows: np.ndarray, n: int = 200, block: int = 6, seed: int = 0) -> float:
    """How often shuffled returns (blocks of bars, the same shuffle for every instrument, so their
    co-movement is kept) give a Sharpe at least as high as the strategy's. A low value says the
    timing of the positions matters."""
    from phasor import validation

    held = _shift(run.weights, 1)[rows]
    asset = np.nan_to_num(run.asset_returns[rows])
    if len(asset) < 2 * block:
        return float("nan")
    return float(validation.permutation_test(held, asset, statistic="sharpe", n=n, block=block, seed=seed)["p_value"])


def delay_curve(run: Backtest, panel: Panel, rows: np.ndarray, extra=(0, 1, 2, 3)) -> dict[str, float]:
    """The segment's Sharpe with the weights held ``d`` more bars late, before funding: the book's
    alpha-decay test. Costs are charged on the delayed turnover."""
    from phasor import validation

    held = _shift(run.weights, 1)[rows]
    asset = np.nan_to_num(run.asset_returns[rows])
    curve = validation.position_delay_test(
        held, asset, delays=extra, cost=run.cost_rate, ann_factor=panel.bars_per_year
    )
    return {str(delay): (value if math.isfinite(value) else float("nan")) for delay, value in curve.items()}


def diagnostics(raw: np.ndarray, panel: Panel, rows: np.ndarray, horizons=(1, 2, 4, 8, 16, 32)) -> dict[str, Any]:
    """What the data says about the strategy's prediction, on the pool rows of a segment where a
    position is held: the return that follows a position, by horizon (does the edge decay as
    stated?), and how much the position explains the next bar's return -- its correlation, R²,
    and the next return of each position quintile (is a larger position followed by a larger
    move, monotonically?)."""
    from phasor import predictions

    close = panel.fields["close"]
    held = np.nan_to_num(np.clip(raw, -1, 1)) * panel.pool
    taken = rows[:, None] & (np.abs(held) > TOLERANCE)
    decay = predictions.decay(np.where(taken, held, 0.0), close, horizons=tuple(horizons))
    forward = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        forward[:-1] = close[1:] / close[:-1] - 1
    found: dict[str, Any] = {"signed_forward_return_by_horizon": {str(h): value for h, value in decay.items()}}
    if taken.sum() >= 50:
        power = predictions.power(np.where(taken, held, np.nan), forward, quantiles=5)
        found["next_return_correlation"] = power["correlation"]
        found["r_squared"] = power["r_squared"]
        levels = power.get("quantile_predictions", {})
        found["next_return_by_position_quintile"] = {
            f"q{g} ({levels[g]:+.2f})" if g in levels else f"q{g}": value
            for g, value in power["quantile_returns"].items()
        }
        found["quintile_monotonicity"] = power["monotonicity"]
    return found


def cases(spec: FactorSpec) -> dict[str, dict[str, Any]]:
    """The parameter values of each case label (``{}`` for a strategy without parameters)."""
    if spec.mode != "expression" or not spec.params:
        return {"default": {}}
    import phasor

    grid = phasor.FactorGrid.compile({"position": spec.expression}, **spec.params)
    return {_label(case): dict(case) for case in grid.cases}


def neighbourhood(spec: FactorSpec, found: dict[str, Any], selected: str) -> dict[str, Any]:
    """The chosen case against its grid neighbours (one parameter one step away), on train and
    select: a peak with weak neighbours is a fit to noise more often than one on a plateau."""
    from phasor.params import ParamResults, neighbors

    labels = list(found)
    combos = cases(spec)
    if len(labels) < 2 or set(combos) != set(labels):
        return {}
    results = ParamResults(
        [combos[label] for label in labels],
        [
            {segment: _float(found[label].get(segment, {}).get("sharpe")) for segment in ("train", "select")}
            for label in labels
        ],
    )
    at = labels.index(selected)
    out = {}
    for segment in ("train", "select"):
        table = neighbors(results, key=segment)
        if int(table["count"][at]) == 0:
            continue
        out[segment] = {
            "sharpe": _float(table["value"][at]),
            "neighbour_mean": _float(table["mean"][at]),
            "neighbour_min": _float(table["min"][at]),
            "ratio": _float(table["ratio"][at]),
            "neighbours": int(table["count"][at]),
        }
    return out


def bootstrap(run: Backtest, panel: Panel, rows: np.ndarray, n: int = 1000, seed: int = 0) -> dict[str, Any]:
    """The segment's maximum drawdown and Sharpe over resampled paths (blocks of one day of bars):
    the history is one ordering of its returns, and its drawdown one draw."""
    from phasor import robust

    block = max(1, round(panel.bars_per_year / 365))
    found = robust.bootstrap(
        run.returns[rows], n=n, block=block, seed=seed, ann_factor=panel.bars_per_year,
        quantiles=(0.01, 0.05, 0.5, 0.95),
    )  # fmt: skip
    return {
        name: {"observed": values["observed"], **{f"q{round(q * 100):02d}": v for q, v in values["quantiles"].items()}}
        for name, values in found.items()
    }


@dataclass(frozen=True)
class Segments:
    train: tuple[str, str]
    select: tuple[str, str]
    lockbox: tuple[str, str | None]

    def rows(self, panel: Panel) -> dict[str, np.ndarray]:
        times = pd.to_datetime(panel.index, unit="ns", utc=True)
        found = {}
        for name, (start, end) in asdict(self).items():
            mask = times >= pd.Timestamp(start, tz="UTC")
            if end is not None:
                mask &= times < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            found[name] = np.asarray(mask)
        return found


def evaluate(
    spec: FactorSpec,
    panel: Panel,
    segments: Segments,
    settings: BacktestSettings = BacktestSettings(),
    trial_sharpes: list[float] | None = None,
    returns_out: Path | None = None,
) -> dict[str, Any]:
    """Every case on every segment; the case chosen on the train segment, with its robustness
    statistics and diagnostics on train and select. Lockbox numbers are only under ``lockbox``.

    ``trial_sharpes`` are the per-bar select Sharpe ratios of the earlier trials. With
    ``returns_out``, the chosen case's returns are written there as Parquet (``time``, ``returns``,
    ``segment``), for a report that deflates the lockbox Sharpe by every trial."""
    import phasor

    started = time.perf_counter()
    found = positions(spec, panel)
    rows = segments.rows(panel)
    seen = rows["train"] | rows["select"]
    cases: dict[str, Any] = {}
    runs: dict[str, Backtest] = {}
    for label, raw in found.items():
        run = backtest(raw, panel, settings)
        runs[label] = run
        cases[label] = {name: metrics(run, panel, mask) for name, mask in rows.items()}
    selected = max(cases, key=lambda label: _finite(cases[label]["train"].get("sharpe")))
    run, raw = runs[selected], found[selected]
    delayed = backtest(raw, panel, BacktestSettings(**{**asdict(settings), "delay": settings.delay + 1}))
    per_bar = [_finite(cases[label]["select"].get("sharpe")) / math.sqrt(panel.bars_per_year) for label in cases]
    trials = [*(trial_sharpes or []), *per_bar]
    select_returns = run.returns[rows["select"]]
    robustness = {
        "select_sharpe_delayed_one_more_bar": metrics(delayed, panel, rows["select"]).get("sharpe"),
        "select_sharpe_by_extra_delay_before_funding": delay_curve(run, panel, rows["select"]),
        "select_sharpe_of_each_case": {label: cases[label]["select"].get("sharpe") for label in cases},
        "parameter_neighbourhood": neighbourhood(spec, cases, selected),
        "bootstrap_train_and_select": bootstrap(run, panel, seen),
        "permutation_p_value_train_and_select": permutation_p_value(run, seen),
        "trials": len(trials),
        "select_deflated_sharpe_probability": (
            float(phasor.robust.deflated_sharpe(select_returns, np.array(trials)))
            if len(trials) >= 2 and select_returns.size > 2
            else float("nan")
        ),
    }
    if returns_out is not None:
        segment = np.full(len(panel.index), "", dtype=object)
        for name in ("train", "select", "lockbox"):
            segment[rows[name]] = name
        pd.DataFrame(
            {"returns": run.returns, "segment": segment},
            index=pd.to_datetime(panel.index, unit="ns", utc=True).rename("time"),
        ).to_parquet(returns_out)
    return {
        "selected_case": selected,
        "bars_per_year": panel.bars_per_year,
        "cases": cases,
        "robustness": robustness,
        "diagnostics": {name: diagnostics(raw, panel, rows[name]) for name in ("train", "select")},
        "lockbox_diagnostics": diagnostics(raw, panel, rows["lockbox"]),
        "seconds": round(time.perf_counter() - started, 2),
    }


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def without_lockbox(result: dict[str, Any]) -> dict[str, Any]:
    """The evaluation with every lockbox number removed: what the model may see."""
    hidden = json.loads(json.dumps(result, default=float))
    for case in hidden.get("cases", {}).values():
        case.pop("lockbox", None)
    hidden.pop("lockbox_diagnostics", None)
    return hidden
