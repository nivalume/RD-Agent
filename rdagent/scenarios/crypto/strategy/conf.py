"""The crypto strategy settings, from ``CRYPTO_STRATEGY_`` environment variables.

The panel is written by ``jarvis-history panel --interval 4h --top 100 --out <panel_dir>``; its
``debug/`` slice is what the coder checks code on.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.scenarios.crypto.strategy.engine import BacktestSettings, Segments


class CryptoStrategySettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_STRATEGY_", env_parse_none_str="None")

    panel_dir: str = "~/jarvis-data/strategy/4h"
    """The panel: one Parquet table a field, and ``pool.parquet``; the bar length is the panel's."""

    target_vol: float = 0.20
    """Annualised volatility each instrument's position is scaled to before they are averaged."""

    max_leverage: float = 3.0
    """The most an instrument's volatility scaling may multiply its position by."""

    vol_lookback: int = 180
    """Bars of returns that estimate an instrument's volatility (180 bars of 4h: 30 days)."""

    fee: float = 0.0005
    """Fee on each trade, as a fraction of its value: a taker fee on USD-M perpetuals."""

    slippage: float = 0.0002
    """Price slippage on each trade, as a fraction."""

    delay: int = 1
    """Bars between the close a position is decided on and the close it trades at."""

    train_start: str = "2021-01-01"
    train_end: str = "2023-06-30"
    valid_start: str = "2023-07-01"
    valid_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: Optional[str] = None
    """The segments; test_end None runs to the end of the panel."""

    hide_test: bool = True
    """Keep the test segment out of everything the model sees: feedback, the trace, SOTA choice."""

    min_valid_sharpe: float = 0.0
    """A strategy whose valid Sharpe is not above this is rejected before the model judges it."""

    max_p_value: float = 0.05
    """A strategy whose permutation p-value (train and valid) is above this is rejected."""

    max_delay_loss: float = 0.5
    """Rejected when one more bar of delay loses more than this share of the valid Sharpe."""

    min_neighbour_ratio: Optional[float] = None
    """When set, rejected when the chosen case's grid neighbours average less than this share of its
    train Sharpe (a peak rather than a plateau). Off by default: it is reported either way."""

    max_correlation: float = 0.7
    """A strategy whose valid returns correlate more than this with an accepted one is not new."""

    min_portfolio_gain: float = 0.05
    """A strategy is accepted only when adding it raises the book's valid Sharpe by at least this."""

    python_bin: str = "python"
    """The interpreter that runs a strategy (it needs rdagent and phasor)."""

    timeout: int = 1800
    """Seconds a backtest of the full panel may take."""

    @property
    def panel_path(self) -> Path:
        return Path(os.path.expandvars(self.panel_dir)).expanduser()

    @property
    def debug_panel_path(self) -> Path:
        return self.panel_path / "debug"

    def backtest(self) -> BacktestSettings:
        return BacktestSettings(
            target_vol=self.target_vol,
            max_leverage=self.max_leverage,
            vol_lookback=self.vol_lookback,
            fee=self.fee,
            slippage=self.slippage,
            delay=self.delay,
        )

    def segments(self) -> Segments:
        return Segments(
            (self.train_start, self.train_end),
            (self.valid_start, self.valid_end),
            (self.test_start, self.test_end),
        )


CRYPTO_STRATEGY_SETTINGS = CryptoStrategySettings()

__all__ = ["CRYPTO_STRATEGY_SETTINGS", "CryptoStrategySettings"]
