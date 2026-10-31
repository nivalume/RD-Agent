"""The crypto qlib settings: where the data is, the pool, the backtest, and the time segments.

Every field can be set with a ``CRYPTO_QLIB_`` environment variable (``CRYPTO_QLIB_MARKET=top50``, ...).
The loops' own settings (which classes they wire) are in :mod:`rdagent.app.crypto_rd_loop.qlib_conf`.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class CryptoQlibSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_QLIB_", env_parse_none_str="None")

    provider_uri: str = "~/.qlib/qlib_data/crypto_data"
    """The qlib data directory ``jarvis-history qlib`` wrote. It must be under ``~/.qlib``: that is
    what the docker environment mounts (at ``/root/.qlib``)."""

    market: str = "top100"
    """The instrument pool: a file of the data directory's ``instruments/``."""

    benchmark: str = "BTCUSDT"
    """The instrument the backtest compares the portfolio with."""

    topk: int = 20
    """Instruments held by the backtest strategy."""

    n_drop: int = 4
    """Instruments the strategy may replace each day."""

    open_cost: float = 0.0005
    """Cost of buying, as a fraction of the amount: a taker fee on USD-M perpetuals."""

    close_cost: float = 0.0005
    """Cost of selling, as a fraction of the amount."""

    train_start: str = "2021-01-01"
    train_end: str = "2023-06-30"
    valid_start: str = "2023-07-01"
    valid_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: Optional[str] = None
    """The time segments. ``None`` for test_end: up to the last day of the data."""

    factor_data_folder: str = "git_ignore_folder/crypto_factor_source_data"
    """Where the factor coder reads ``daily_pv.h5`` from: the whole history of the pool."""

    factor_data_folder_debug: str = "git_ignore_folder/crypto_factor_source_data_debug"
    """A small ``daily_pv.h5`` the factor coder checks its code on first."""

    debug_instruments: int = 30
    """Instruments in the debug data."""

    debug_start: str = "2024-01-01"
    """First day of the debug data."""

    @property
    def data_dir(self) -> Path:
        """The data directory on this machine."""
        return Path(self.provider_uri).expanduser()

    def run_env(self) -> dict[str, str]:
        """The variables the workflow configs read, as qlib renders them (``{{ market }}``, ...)."""
        found = {
            "crypto_provider_uri": self.provider_uri,
            "crypto_market": self.market,
            "crypto_benchmark": self.benchmark,
            "crypto_topk": str(self.topk),
            "crypto_n_drop": str(self.n_drop),
            "crypto_open_cost": str(self.open_cost),
            "crypto_close_cost": str(self.close_cost),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "test_start": self.test_start,
        }
        if self.test_end is not None:
            found["test_end"] = self.test_end
        return found


CRYPTO_QLIB_SETTINGS = CryptoQlibSettings()

__all__ = ["CRYPTO_QLIB_SETTINGS", "CryptoQlibSettings"]
