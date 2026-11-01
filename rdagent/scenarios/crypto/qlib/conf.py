"""The crypto qlib settings: where the data is, the pool, the backtest, and the time segments.

Every field can be set with a ``CRYPTO_QLIB_`` environment variable (``CRYPTO_QLIB_MARKET=top50``, ...).
The loops' own settings (which classes they wire) are in :mod:`rdagent.app.crypto_rd_loop.qlib_conf`.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class CryptoQlibSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_QLIB_", env_parse_none_str="None", populate_by_name=True)

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
    train_end: str = "2022-12-31"
    """Fitting: the model is trained and the features normalised on this period."""

    valid_start: str = "2023-01-01"
    valid_end: str = "2023-06-30"
    """Early stopping of the model; nothing is compared on it."""

    select_start: str = "2023-07-01"
    select_end: str = "2024-06-30"
    """Selection: the IC and backtest the feedback and the SOTA choice read. The loop's workflows run
    with this as qlib's ``test`` segment, and load no data after it."""

    lockbox_start: str = Field(
        "2024-07-01",
        validation_alias=AliasChoices("lockbox_start", "CRYPTO_QLIB_LOCKBOX_START", "CRYPTO_QLIB_TEST_START"),
    )
    lockbox_end: Optional[str] = Field(
        None, validation_alias=AliasChoices("lockbox_end", "CRYPTO_QLIB_LOCKBOX_END", "CRYPTO_QLIB_TEST_END")
    )
    """The lockbox: run once, by ``python -m rdagent.scenarios.crypto.qlib.lockbox``. ``None`` runs to
    the last day of the data. The old ``CRYPTO_QLIB_TEST_*`` names set it."""

    embargo_days: int = 2
    """Days cut from the end of the train, valid and select segments: the label of a day reads the
    next two closes, so without the cut a segment's last labels would be prices of the next one."""

    factor_data_folder: str = "git_ignore_folder/crypto_factor_source_data"
    """Where the factor coder reads ``daily_pv.h5`` from: the whole history of the pool."""

    factor_data_folder_debug: str = "git_ignore_folder/crypto_factor_source_data_debug"
    """A small ``daily_pv.h5`` the factor coder checks its code on first."""

    debug_instruments: int = 30
    """Instruments in the debug data."""

    debug_days: int = 365
    """Days of the debug data, ending on ``train_end``: inside the fitting period."""

    @property
    def data_dir(self) -> Path:
        """The data directory on this machine."""
        return Path(self.provider_uri).expanduser()

    def _cut(self, day: str) -> str:
        return (date.fromisoformat(day) - timedelta(days=self.embargo_days)).isoformat()

    @property
    def debug_start(self) -> str:
        return (date.fromisoformat(self.train_end) - timedelta(days=self.debug_days - 1)).isoformat()

    def run_env(self, lockbox: bool = False, last_day: Optional[str] = None) -> dict[str, str]:
        """The variables the workflow configs read, as qlib renders them (``{{ market }}``, ...).

        qlib's ``train`` and ``valid`` are the fitting and early-stopping periods; its ``test`` --
        what the IC, the backtest and so the feedback are computed on -- is the selection period,
        or with ``lockbox`` the lockbox (to ``lockbox_end``, or ``last_day``). Train, valid and select
        end ``embargo_days`` early, so no label of one period reads the prices of the next."""
        if lockbox:
            test_start, test_end = self.lockbox_start, self.lockbox_end or last_day
        else:
            test_start, test_end = self.select_start, self._cut(self.select_end)
        found = {
            "crypto_provider_uri": self.provider_uri,
            "crypto_market": self.market,
            "crypto_benchmark": self.benchmark,
            "crypto_topk": str(self.topk),
            "crypto_n_drop": str(self.n_drop),
            "crypto_open_cost": str(self.open_cost),
            "crypto_close_cost": str(self.close_cost),
            "train_start": self.train_start,
            "train_end": self._cut(self.train_end),
            "valid_start": self.valid_start,
            "valid_end": self._cut(self.valid_end),
            "test_start": test_start,
        }
        if test_end is not None:
            found["test_end"] = test_end
        return found


CRYPTO_QLIB_SETTINGS = CryptoQlibSettings()

__all__ = ["CRYPTO_QLIB_SETTINGS", "CryptoQlibSettings"]
