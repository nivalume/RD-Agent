from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings


class PhasorCoSTEERSettings(CoSTEERSettings):
    """Settings of the phasor factor coder.

    ``fields`` is the set of panel columns a generated factor may read; the
    scenario can narrow it per task by setting ``PhasorFactorTask.fields``.
    """

    model_config = SettingsConfigDict(env_prefix="PHASOR_CoSTEER_")

    fields: list[str] = ["open", "high", "low", "close", "volume"]
    """Panel columns a generated factor may read, without the leading dollar sign."""

    max_params: int = 2
    """Maximum number of parameter sweeps in one expression-mode factor."""

    max_cases: int = 6
    """Maximum number of grid cases in total (the product of the parameter lengths)."""

    llm_retries: int = 5
    """How many times a malformed response is sent back to the model with the parse error."""

    v2_error_summary: bool = True
    """Before rewriting a factor that has already failed, ask for short critics.

    The critics cover that failure and similar failures that were later fixed.
    """


PHASOR_COSTEER_SETTINGS = PhasorCoSTEERSettings()
