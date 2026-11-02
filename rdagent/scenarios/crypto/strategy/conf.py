"""The crypto strategy settings, from ``CRYPTO_STRATEGY_`` environment variables.

The panel is written by ``jarvis-data panel --interval 4h --top 100 --out <panel_dir>``; its
the coder checks code on a slice of it cut from the fitting period (see ``debug.py``).
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.scenarios.crypto.strategy.engine import BacktestSettings, Segments


class CryptoStrategySettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_STRATEGY_", env_parse_none_str="None", populate_by_name=True)

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
    """Fitting: the parameter case of each strategy is chosen on this period."""

    select_start: str = Field(
        "2023-07-01",
        validation_alias=AliasChoices("select_start", "CRYPTO_STRATEGY_SELECT_START", "CRYPTO_STRATEGY_VALID_START"),
    )
    select_end: str = Field(
        "2024-06-30",
        validation_alias=AliasChoices("select_end", "CRYPTO_STRATEGY_SELECT_END", "CRYPTO_STRATEGY_VALID_END"),
    )
    """Selection: the gates, the book and every number the model sees are on this period."""

    lockbox_start: str = Field(
        "2024-07-01",
        validation_alias=AliasChoices("lockbox_start", "CRYPTO_STRATEGY_LOCKBOX_START", "CRYPTO_STRATEGY_TEST_START"),
    )
    lockbox_end: Optional[str] = Field(
        None, validation_alias=AliasChoices("lockbox_end", "CRYPTO_STRATEGY_LOCKBOX_END", "CRYPTO_STRATEGY_TEST_END")
    )
    """The lockbox: only the report shows it; None runs to the end of the panel. The old
    ``CRYPTO_STRATEGY_VALID_*`` / ``TEST_*`` names still set the selection and lockbox periods."""

    hide_lockbox: bool = Field(
        True, validation_alias=AliasChoices("hide_lockbox", "CRYPTO_STRATEGY_HIDE_LOCKBOX", "CRYPTO_STRATEGY_HIDE_TEST")
    )
    """Keep the lockbox out of everything the model sees: feedback, the trace, the book."""

    debug_instruments: int = 30
    """Instruments of the coder's debug panel: the most traded over the debug period."""

    debug_days: int = 365
    """Days of the coder's debug panel, ending on ``train_end``: inside the fitting period."""

    min_select_sharpe: float = Field(
        0.0,
        validation_alias=AliasChoices(
            "min_select_sharpe", "CRYPTO_STRATEGY_MIN_SELECT_SHARPE", "CRYPTO_STRATEGY_MIN_VALID_SHARPE"
        ),
    )
    """A strategy whose select Sharpe is not above this is rejected before the model judges it."""

    max_p_value: float = 0.05
    """A strategy whose permutation p-value (train and select) is above this is rejected."""

    max_delay_loss: float = 0.5
    """Rejected when one more bar of delay loses more than this share of the select Sharpe."""

    min_neighbour_ratio: Optional[float] = None
    """When set, rejected when the chosen case's grid neighbours average less than this share of its
    train Sharpe (a peak rather than a plateau). Off by default: it is reported either way."""

    max_correlation: float = 0.7
    """A strategy whose select returns correlate more than this with an accepted one is not new."""

    min_portfolio_gain: float = 0.05
    """A strategy is accepted only when adding it raises the book's select Sharpe by at least this."""

    python_bin: str = "python"
    """The interpreter that runs a strategy (it needs rdagent and phasor)."""

    timeout: int = 1800
    """Seconds a backtest of the full panel may take."""

    @property
    def panel_path(self) -> Path:
        return Path(os.path.expandvars(self.panel_dir)).expanduser()

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
            (self.select_start, self.select_end),
            (self.lockbox_start, self.lockbox_end),
        )


CRYPTO_STRATEGY_SETTINGS = CryptoStrategySettings()

__all__ = ["CRYPTO_STRATEGY_SETTINGS", "CryptoStrategySettings"]
