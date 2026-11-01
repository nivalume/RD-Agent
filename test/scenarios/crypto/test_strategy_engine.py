"""The strategy engine on synthetic panels whose answers are known."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from rdagent.components.coder.phasor_coder.spec import FactorSpecError, validate_factor_spec
from rdagent.scenarios.crypto.strategy import engine
from rdagent.scenarios.crypto.strategy.engine import (
    BacktestSettings,
    Panel,
    Segments,
    StrategyError,
    backtest,
    check,
    evaluate,
    positions,
    without_lockbox,
)

FIELDS = list(engine.FIELDS)
HOUR = 3_600 * 10**9


def make_panel(close: np.ndarray, pool: np.ndarray | None = None, funding: np.ndarray | None = None, step=4 * HOUR):
    rows, columns = close.shape
    index = np.int64(1_704_067_200 * 10**9) + np.arange(rows, dtype=np.int64) * step  # from 2024-01-01
    fields = {name: np.asfortranarray(close.copy()) for name in engine.FIELDS}
    fields["funding_rate"] = np.zeros_like(close) if funding is None else funding
    symbols = ("BTCUSDT", *(f"C{index}USDT" for index in range(1, columns)))
    return Panel(index, symbols, fields, np.ones_like(close, dtype=bool) if pool is None else pool)


def random_panel(rows=600, columns=3, seed=0):
    generator = np.random.default_rng(seed)
    returns = generator.normal(0, 0.01, size=(rows, columns))
    return make_panel(100 * np.exp(np.cumsum(returns, axis=0)))


def spec(**data):
    return validate_factor_spec({"fields": ["close"], **data}, available_fields=FIELDS)


TREND = spec(mode="expression", expression="Sign(Mean($close, 5) - Mean($close, 20))")
GRID = spec(mode="expression", expression="Sign(Mean($close, {n}) - Mean($close, 4*{n}))", params={"n": [3, 5]})
PYTHON = spec(
    mode="python",
    code=(
        "import numpy as np\n\n"
        "def calculate(fields, index, symbols):\n"
        "    close = fields['close']\n"
        "    previous = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])\n"
        "    return np.sign(close - previous)\n"
    ),
)
PEEKING = spec(
    mode="python",
    code=(
        "import numpy as np\n\n"
        "def calculate(fields, index, symbols):\n"
        "    close = fields['close']\n"
        "    following = np.vstack([close[1:], np.full((1, close.shape[1]), np.nan)])\n"
        "    return np.sign(following - close)\n"
    ),
)


# -- positions and checks ----------------------------------------------------------------------


def test_positions_of_both_forms_and_every_case():
    panel = random_panel()
    assert set(positions(TREND, panel)) == {"default"}
    assert set(positions(GRID, panel)) == {"n=3", "n=5"}
    assert positions(GRID, panel)["n=5"].shape == panel.shape
    python = positions(PYTHON, panel)["default"]
    assert np.isnan(python[0]).all() and set(np.unique(python[1:])) <= {-1.0, 0.0, 1.0}


def test_a_good_strategy_passes_the_checks():
    result = check(TREND, random_panel())
    assert result.ok, result.problems
    assert result.cases == 1 and 0.9 < result.in_market <= 1.0


def test_a_peeking_strategy_is_caught_by_truncation():
    result = check(PEEKING, random_panel())
    assert not result.ok
    assert any("Look-ahead" in problem for problem in result.problems)


def test_an_expression_that_reads_ahead_is_refused_when_parsed():
    with pytest.raises(FactorSpecError, match="reads the future"):
        spec(mode="expression", expression="Sign(Ref($close, -1) - $close)")


@pytest.mark.parametrize(
    "expression,message",
    [
        ("$close / Mean($close, 5)", "range from"),
        ("$close * 0", "flat"),
        ("Log($close - $close)", "infinite"),
    ],
)
def test_positions_must_be_bounded_finite_and_not_flat(expression, message):
    result = check(spec(mode="expression", expression=expression), random_panel())
    assert any(message in problem for problem in result.problems), result.problems


def test_code_that_fails_or_returns_the_wrong_shape():
    broken = spec(mode="python", code="def calculate(fields, index, symbols):\n    return 1 / 0\n")
    assert "ZeroDivisionError" in check(broken, random_panel()).problems[0]
    wrong = spec(
        mode="python", code="import numpy as np\n\ndef calculate(fields, index, symbols):\n    return np.zeros(3)\n"
    )
    with pytest.raises(StrategyError, match="shape"):
        positions(wrong, random_panel())


# -- the backtest ------------------------------------------------------------------------------


UNSCALED = BacktestSettings(target_vol=1e9, max_leverage=1.0, vol_lookback=4, fee=0.0, slippage=0.0)


def test_holding_one_instrument_earns_its_returns_one_bar_late():
    panel = random_panel(rows=60, columns=1)
    run = backtest(np.ones(panel.shape), panel, UNSCALED)
    asset = run.asset_returns[:, 0]
    # The volatility needs two returns (rows 1 and 2): the position decided at the close of row 2
    # is set at the close of row 3, so the account moves with the instrument from row 4.
    first = int(np.argmax(run.weights[:, 0] > 0))
    assert first == 3 and run.weights[first:, 0] == pytest.approx(1.0)
    assert np.allclose(run.returns[first + 1 :], asset[first + 1 :], atol=1e-12)
    assert np.allclose(run.returns[: first + 1], 0.0)


def test_a_long_pays_positive_funding_and_a_short_receives_it():
    close = random_panel(rows=30, columns=2).fields["close"]
    funding = np.zeros_like(close)
    funding[20] = 0.001
    unpaid, paid = make_panel(close), make_panel(close, funding=funding)

    def effect(position):
        raw = np.tile(position, (30, 1))
        return backtest(raw, paid, UNSCALED).returns[20] - backtest(raw, unpaid, UNSCALED).returns[20]

    # Half the account in each instrument: a long pays half the rate, a short receives it.
    assert effect([1.0, 0.0]) == pytest.approx(-0.5 * 0.001, rel=0.05)
    assert effect([0.0, -1.0]) == pytest.approx(0.5 * 0.001, rel=0.05)
    assert effect([1.0, -1.0]) == pytest.approx(0.0, abs=0.0001)


def test_nothing_is_held_outside_the_pool_or_after_a_delisting():
    close = random_panel(rows=40, columns=2).fields["close"].copy()
    close[30:, 1] = np.nan  # delisted after row 29
    pool = np.ones_like(close, dtype=bool)
    pool[:15, 0] = False
    run = backtest(np.ones_like(close), make_panel(close, pool=pool), UNSCALED)
    assert not run.weights[:16, 0].any()  # out of the pool until row 15, set a bar later
    assert not run.weights[30:, 1].any()
    assert np.isfinite(run.returns).all()


def test_each_instrument_is_scaled_to_the_same_volatility():
    generator = np.random.default_rng(1)
    returns = generator.normal(0, 1, size=(2000, 2)) * np.array([0.01, 0.04])
    panel = make_panel(100 * np.exp(np.cumsum(returns, axis=0)))
    run = backtest(np.ones(panel.shape), panel, BacktestSettings(target_vol=0.2, vol_lookback=180, fee=0, slippage=0))
    ratio = run.weights[500:, 0] / run.weights[500:, 1]
    assert np.median(ratio) == pytest.approx(4.0, rel=0.15)  # four times the weight for a quarter of the risk


def test_costs_are_charged_on_turnover():
    panel = random_panel(rows=50, columns=1)
    flipping = np.where(np.arange(50) % 2 == 0, 1.0, -1.0)[:, None]
    free = backtest(flipping, panel, UNSCALED)
    paid = backtest(flipping, panel, BacktestSettings(target_vol=1e9, max_leverage=1.0, vol_lookback=4, fee=0.001))
    assert paid.returns[5:].sum() < free.returns[5:].sum() - 0.001 * 40


# -- the evaluation ----------------------------------------------------------------------------


SEGMENTS = Segments(("2024-01-01", "2024-02-15"), ("2024-02-16", "2024-03-15"), ("2024-03-16", None))


def test_an_evaluation_reports_every_case_and_hides_nothing_until_asked():
    panel = random_panel(rows=600)
    result = evaluate(GRID, panel, SEGMENTS, BacktestSettings(vol_lookback=30), trial_sharpes=[0.01, 0.02])
    assert result["selected_case"] in {"n=3", "n=5"}
    assert set(result["cases"]["n=3"]) == {"train", "select", "lockbox"}
    assert result["cases"]["n=3"]["train"]["bars"] > 200
    robustness = result["robustness"]
    assert robustness["trials"] == 4 and 0 <= robustness["permutation_p_value_train_and_select"] <= 1
    assert set(result["diagnostics"]["select"]["signed_forward_return_by_horizon"]) == {"1", "2", "4", "8", "16", "32"}
    train = result["cases"]["n=3"]["train"]
    assert train["trades"] > 0 and 0 <= train["trade_win_rate"] <= 1 and 0 <= train["hit_ratio"] <= 1
    assert {"sortino", "calmar", "omega", "return_concentration", "max_drawdown_recovery_bars"} <= set(train)
    assert set(robustness["select_sharpe_by_extra_delay_before_funding"]) == {"0", "1", "2", "3"}
    neighbourhood = robustness["parameter_neighbourhood"]
    assert set(neighbourhood) == {"train", "select"} and neighbourhood["train"]["neighbours"] == 1
    drawdown = robustness["bootstrap_train_and_select"]["max_drawdown"]
    assert drawdown["q01"] <= drawdown["q05"] <= drawdown["q50"] <= 0
    assert 0 <= result["diagnostics"]["select"]["r_squared"] < 1
    hidden = without_lockbox(result)
    text = json.dumps(hidden)
    assert '"lockbox"' not in text and "lockbox_diagnostics" not in text
    assert set(hidden["cases"]["n=3"]) == {"train", "select"}


def test_a_strategy_that_knows_the_future_has_a_tiny_p_value():
    panel = random_panel(rows=600, columns=3, seed=4)
    close = panel.fields["close"]
    future = np.zeros_like(close)
    future[:-2] = np.sign(close[2:] / close[1:-1] - 1)  # the return it will hold, two rows on
    run = backtest(future, panel, BacktestSettings(vol_lookback=30, fee=0, slippage=0))
    seen = np.ones(len(close), dtype=bool)
    assert engine.permutation_p_value(run, seen, n=100) < 0.05
    assert engine.metrics(run, panel, seen)["sharpe"] > 5


def test_the_panel_loads_what_jarvis_history_writes(tmp_path):
    panel = random_panel(rows=10, columns=2)
    times = pd.to_datetime(panel.index, unit="ns", utc=True).rename("time")
    for name, values in panel.fields.items():
        pd.DataFrame(values, index=times, columns=list(panel.symbols)).to_parquet(tmp_path / f"{name}.parquet")
    pd.DataFrame(panel.pool, index=times, columns=list(panel.symbols)).to_parquet(tmp_path / "pool.parquet")
    loaded = Panel.load(tmp_path)
    assert loaded.symbols == panel.symbols and (loaded.index == panel.index).all()
    assert np.allclose(loaded.fields["close"], panel.fields["close"])
    assert loaded.bars_per_year == pytest.approx(365 * 6)
    assert math.isclose(loaded.head(4).fields["close"][-1, 0], panel.fields["close"][3, 0])


def test_trade_stats_count_round_trips_closed_in_the_segment():
    trades = {
        "exit_idx": np.array([5, 15, 25, 40]),
        "status": np.array([1, 1, 1, 0]),  # the last one is still open
        "pnl": np.array([1.0, -0.5, 2.0, 3.0]),
        "duration": np.array([4.0, 6.0, 10.0, 8.0]),
    }
    found = engine.trade_stats(trades, 10, 30)
    assert found["trades"] == 2 and found["trade_win_rate"] == 0.5
    assert found["profit_factor"] == pytest.approx(4.0)
    assert found["mean_winning_trade_bars"] == 10 and found["mean_losing_trade_bars"] == 6
