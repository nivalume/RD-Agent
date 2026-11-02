"""The tables the factor coder runs factor code on, written from the crypto qlib data.

The factor coder executes a factor's code next to ``daily_pv.h5`` from
``FACTOR_COSTEER_SETTINGS.data_folder`` (and first on the small one in ``data_folder_debug``).
:func:`use_crypto_factor_data` points those two settings at the crypto folders and writes them when
they are missing, by running ``factor_data_template/generate.py`` in the qlib environment.
"""

import shutil
from pathlib import Path

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS, CryptoQlibSettings
from rdagent.scenarios.crypto.qlib.workspace import (
    FACTOR_DATA_TEMPLATE,
    crypto_qlib_env,
    require_data,
)


def generate_crypto_factor_data(settings: CryptoQlibSettings = CRYPTO_QLIB_SETTINGS) -> None:
    """Run the data template in the qlib environment and copy its tables to the factor folders."""
    require_data(settings)
    env = crypto_qlib_env()
    if env is None:
        raise RuntimeError("MODEL_CoSTEER_ENV_TYPE must be docker or conda")
    env.prepare()
    log = env.check_output(
        local_path=str(FACTOR_DATA_TEMPLATE),
        entry="python generate.py",
        env={
            "crypto_provider_uri": settings.provider_uri,
            "crypto_market": settings.market,
            "debug_instruments": str(settings.debug_instruments),
            "debug_start": settings.debug_start,
            "debug_end": settings.train_end,
        },
    )
    for name in ("daily_pv_all.h5", "daily_pv_debug.h5"):
        if not (FACTOR_DATA_TEMPLATE / name).exists():
            raise RuntimeError(f"generate.py did not write {name}:\n{log}")
    for source, folder in (
        ("daily_pv_all.h5", settings.factor_data_folder),
        ("daily_pv_debug.h5", settings.factor_data_folder_debug),
    ):
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(FACTOR_DATA_TEMPLATE / source, target / "daily_pv.h5")
        shutil.copy(FACTOR_DATA_TEMPLATE / "README.md", target / "README.md")
    _stamp_path(settings).write_text(_debug_stamp(settings))
    logger.info(f"Wrote the crypto factor data to {settings.factor_data_folder}")


DEBUG_STAMP = ".period"


def _stamp_path(settings: CryptoQlibSettings) -> Path:
    """Beside the debug folder, not in it: the coder describes and copies every file of the folder."""
    folder = Path(settings.factor_data_folder_debug)
    return folder.with_name(folder.name + DEBUG_STAMP)


def _debug_stamp(settings: CryptoQlibSettings) -> str:
    return f"{settings.market} {settings.debug_instruments} {settings.debug_start} {settings.train_end}"


def use_crypto_factor_data(settings: CryptoQlibSettings = CRYPTO_QLIB_SETTINGS) -> None:
    """Make the factor coder read the crypto tables, writing them first when they are missing or the
    debug table was cut for another period (it must lie inside the fitting period)."""
    FACTOR_COSTEER_SETTINGS.data_folder = settings.factor_data_folder
    FACTOR_COSTEER_SETTINGS.data_folder_debug = settings.factor_data_folder_debug
    stamp = _stamp_path(settings)
    if (
        not (Path(settings.factor_data_folder) / "daily_pv.h5").exists()
        or not (Path(settings.factor_data_folder_debug) / "daily_pv.h5").exists()
        or not stamp.exists()
        or stamp.read_text() != _debug_stamp(settings)
    ):
        generate_crypto_factor_data(settings)


__all__ = ["generate_crypto_factor_data", "use_crypto_factor_data"]
