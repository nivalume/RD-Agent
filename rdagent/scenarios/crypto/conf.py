"""The crypto scenario's configuration, taken from ``CRYPTO_*`` environment variables.

A scenario keeps its own settings class so that its data root (and anything added later) stays
independent of the other scenarios and of the global :class:`RDAgentSettings`.

Set the root in `.env` (or in the process environment)::

    CRYPTO_DATA_ROOT_DIR=/path/to/data

The schemas are read from ``<data_root_dir>/schemas`` and the rows from
``<data_root_dir>/crypto/data``; see :mod:`rdagent.scenarios.crypto.experiment.data_schema`.
"""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class CryptoSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_")

    data_root_dir: Path = Path.home() / "jarvis-data"
    """Where the data lives: the schemas under ``schemas/``, the rows under ``crypto/data/``."""


__all__ = ["CryptoSettings"]