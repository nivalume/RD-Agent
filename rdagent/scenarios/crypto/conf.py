"""The crypto scenario's configuration, taken from ``CRYPTO_*`` environment variables.

A scenario keeps its own settings class so that its data root (and anything added later) stays
independent of the other scenarios and of the global :class:`RDAgentSettings`.

Set the root in `.env` (or in the process environment)::

    CRYPTO_DATA_ROOT_DIR=/path/to/data

A leading ``~`` (and ``$VAR``) is expanded here: `.env` files keep both literal, and a literal ``~``
would otherwise point at ``<cwd>/~/<...>`` instead of the home directory.

The schemas are read from ``<data_root_dir>/schemas`` and the rows from
``<data_root_dir>/crypto/data``; see :mod:`rdagent.scenarios.crypto.experiment.data_schema`.
"""

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class CryptoSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_")

    data_root_dir: Path = Path.home() / "jarvis-data"
    """Where the data lives: the schemas under ``schemas/``, the rows under ``crypto/data/``."""

    @field_validator("data_root_dir", mode="after")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        """Turn ``~/jarvis-data`` and ``$HOME/jarvis-data`` into a path that exists on disk."""
        return Path(os.path.expandvars(str(value))).expanduser()


__all__ = ["CryptoSettings"]
