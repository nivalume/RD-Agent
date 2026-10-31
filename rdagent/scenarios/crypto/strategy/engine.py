"""The strategy engine: a panel, the positions a strategy takes on it, and their backtest.

Everything that runs a strategy is here, as plain functions on numpy arrays, so the coder's checks,
the runner and the tests run the same code:

* :class:`Panel` -- the wide tables ``jarvis-history panel`` writes: ``(T, S)`` arrays of bars and
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
    weights: np.ndarray  # (T, S) the weights held over each bar (after delay)
    asset_returns: np.ndarray  # (T, S) close-to-close returns
    funding: np.ndarray  # (T,) funding paid (negative) or received over each bar, as a return
    costs: np.ndarray  # (T,) fees and slippage of each bar, as a return


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
    )
    returns = np.nan_to_num(np.asarray(portfolio.returns(), dtype=np.float64).reshape(-1))
    held = _shift(weights, 1)  # what is held over bar t is what was set at the close of t - 1
    paid = (held * funding).sum(axis=1)
    turnover = np.abs(np.diff(np.vstack([np.zeros((1, weights.shape[1])), weights]), axis=0)).sum(axis=1)
    return Backtest(returns, weights, asset_returns, -paid, -(settings.fee + settings.slippage) * turnover)


# -- metrics -----------------------------------------------------------------------------------


def _sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(bars_per_year))


def _max_drawdown(returns: np.ndarray) -> float:
    if not len(returns):
        return float("nan")
    wealth = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(wealth)
    return float((wealth / peaks - 1).min())


def _beta(returns: np.ndarray, market: np.ndarray) -> float:
    if len(returns) < 2 or np.var(market) == 0:
        return float("nan")
    return float(np.cov(returns, market)[0, 1] / np.var(market, ddof=1))


def market_returns(panel: Panel) -> np.ndarray:
    """The equally weighted return of the pool over each bar (the pool of the bar before)."""
    close = panel.fields["close"]
    asset = np.full(close.shape, np.nan)
    asset[1:] = close[1:] / close[:-1] - 1
    pool = _shift(panel.pool.astype(float), 1).astype(bool) & np.isfinite(asset)
    counts = pool.sum(axis=1)
    total = np.where(pool, np.nan_to_num(asset), 0.0).sum(axis=1)
    return np.where(counts > 0, total / np.maximum(counts, 1), 0.0)


def metrics(run: Backtest, panel: Panel, rows: np.ndarray) -> dict[str, Any]:
    """The statistics of a backtest over the rows of a segment (a boolean mask)."""
    bars_per_year = panel.bars_per_year
    returns = run.returns[rows]
    if returns.size < 2:
        return {"bars": int(returns.size)}
    market = market_returns(panel)[rows]
    beta = _beta(returns, market)
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
        "annual_return": float(np.mean(returns) * bars_per_year),
        "annual_volatility": float(np.std(returns, ddof=1) * math.sqrt(bars_per_year)),
        "sharpe": _sharpe(returns, bars_per_year),
        "max_drawdown": _max_drawdown(returns),
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
    if btc is not None:
        found["beta_to_btc"] = _beta(returns, np.nan_to_num(run.asset_returns[rows, btc]))
    return found


# -- robustness and diagnostics ----------------------------------------------------------------


def permutation_p_value(run: Backtest, rows: np.ndarray, n: int = 200, block: int = 6, seed: int = 0) -> float:
    """How often shuffled returns (blocks of bars, the same shuffle for every instrument, so their
    co-movement is kept) give a Sharpe at least as high as the strategy's. A low value says the
    timing of the positions matters."""
    weights = _shift(run.weights, 1)[rows]
    asset = np.nan_to_num(run.asset_returns[rows])
    if len(asset) < 2 * block:
        return float("nan")

    def sharpe(values: np.ndarray) -> float:
        pnl = (weights * values).sum(axis=1)
        deviation = pnl.std()
        return float(pnl.mean() / deviation) if deviation > 0 else 0.0

    observed = sharpe(asset)
    blocks = [asset[start : start + block] for start in range(0, len(asset), block)]
    generator = np.random.default_rng(seed)
    beaten = 0
    for _ in range(n):
        order = generator.permutation(len(blocks))
        shuffled = np.concatenate([blocks[index] for index in order])[: len(asset)]
        beaten += sharpe(shuffled) >= observed
    return (1 + beaten) / (n + 1)


def diagnostics(raw: np.ndarray, panel: Panel, rows: np.ndarray, horizons=(1, 2, 4, 8, 16, 32)) -> dict[str, Any]:
    """What the data says about the strategy's prediction, on the pool rows of a segment:
    the return that follows a position, by horizon (does the edge decay as stated?), and by the
    size of the position (is a larger position followed by a larger move?)."""
    close = panel.fields["close"]
    held = np.nan_to_num(np.clip(raw, -1, 1)) * panel.pool
    mask = rows[:, None] & (np.abs(held) > TOLERANCE)
    found: dict[str, Any] = {"signed_forward_return_by_horizon": {}}
    for horizon in horizons:
        forward = np.full(close.shape, np.nan)
        forward[:-horizon] = close[horizon:] / close[:-horizon] - 1
        values = (np.sign(held) * forward)[mask]
        values = values[np.isfinite(values)]
        found["signed_forward_return_by_horizon"][str(horizon)] = float(values.mean()) if values.size else float("nan")
    forward = np.full(close.shape, np.nan)
    forward[:-1] = close[1:] / close[:-1] - 1
    sizes, nexts = held[mask], forward[mask]
    keep = np.isfinite(nexts)
    sizes, nexts = sizes[keep], nexts[keep]
    if sizes.size >= 50:
        edges = np.quantile(sizes, [0, 0.2, 0.4, 0.6, 0.8, 1])
        buckets = np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, 4)
        found["next_return_by_position_quintile"] = {
            f"q{bucket + 1} ({edges[bucket]:+.2f}..{edges[bucket + 1]:+.2f})": float(nexts[buckets == bucket].mean())
            for bucket in range(5)
            if (buckets == bucket).any()
        }
    return found


# -- the whole evaluation ----------------------------------------------------------------------


@dataclass(frozen=True)
class Segments:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str | None]

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
    statistics and diagnostics on train and valid. Test numbers are only under ``test``.

    ``trial_sharpes`` are the per-bar valid Sharpe ratios of the earlier trials. With
    ``returns_out``, the chosen case's returns are written there as Parquet (``time``, ``returns``,
    ``segment``), for a report that deflates the test Sharpe by every trial."""
    import phasor

    started = time.perf_counter()
    found = positions(spec, panel)
    rows = segments.rows(panel)
    seen = rows["train"] | rows["valid"]
    cases: dict[str, Any] = {}
    runs: dict[str, Backtest] = {}
    for label, raw in found.items():
        run = backtest(raw, panel, settings)
        runs[label] = run
        cases[label] = {name: metrics(run, panel, mask) for name, mask in rows.items()}
    selected = max(cases, key=lambda label: _finite(cases[label]["train"].get("sharpe")))
    run, raw = runs[selected], found[selected]
    delayed = backtest(raw, panel, BacktestSettings(**{**asdict(settings), "delay": settings.delay + 1}))
    per_bar = [_finite(cases[label]["valid"].get("sharpe")) / math.sqrt(panel.bars_per_year) for label in cases]
    trials = [*(trial_sharpes or []), *per_bar]
    valid_returns = run.returns[rows["valid"]]
    robustness = {
        "valid_sharpe_delayed_one_more_bar": metrics(delayed, panel, rows["valid"]).get("sharpe"),
        "valid_sharpe_of_each_case": {label: cases[label]["valid"].get("sharpe") for label in cases},
        "permutation_p_value_train_and_valid": permutation_p_value(run, seen),
        "trials": len(trials),
        "valid_deflated_sharpe_probability": (
            float(phasor.robust.deflated_sharpe(valid_returns, np.array(trials)))
            if len(trials) >= 2 and valid_returns.size > 2
            else float("nan")
        ),
    }
    if returns_out is not None:
        segment = np.full(len(panel.index), "", dtype=object)
        for name in ("train", "valid", "test"):
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
        "diagnostics": {name: diagnostics(raw, panel, rows[name]) for name in ("train", "valid")},
        "test_diagnostics": diagnostics(raw, panel, rows["test"]),
        "seconds": round(time.perf_counter() - started, 2),
    }


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def without_test(result: dict[str, Any]) -> dict[str, Any]:
    """The evaluation with every test-segment number removed: what the model may see."""
    hidden = json.loads(json.dumps(result, default=float))
    for case in hidden.get("cases", {}).values():
        case.pop("test", None)
    hidden.pop("test_diagnostics", None)
    return hidden
