from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.phasor_coder.config import PhasorCoSTEERSettings
from rdagent.scenarios.crypto.strategy.engine import FIELDS


class StrategyCoSTEERSettings(PhasorCoSTEERSettings):
    """Settings of the strategy coder: the phasor coder's, with the panel's fields."""

    model_config = SettingsConfigDict(env_prefix="STRATEGY_CoSTEER_")

    fields: list[str] = list(FIELDS)
    """Panel columns a strategy may read, without the leading dollar sign."""

    check_timeout: int = 300
    """Seconds the checks on the debug panel may take."""


STRATEGY_COSTEER_SETTINGS = StrategyCoSTEERSettings()
