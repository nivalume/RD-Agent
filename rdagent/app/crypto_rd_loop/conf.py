"""The crypto RD loop's settings: the classes the loop wires and how the scenario is rendered.

Only the proposal side is wired: the crypto scenario has no coder, runner or summarizer yet, so those
three settings stay ``None`` and :class:`rdagent.app.crypto_rd_loop.factor.CryptoFactorRDLoop` runs
hypothesis generation and hypothesis-to-experiment conversion only.

Every field can be overridden with a ``CRYPTO_`` prefixed environment variable (``CRYPTO_SCEN``,
``CRYPTO_HYPOTHESIS_GEN``, ...). The data root keeps its own setting, ``CRYPTO_DATA_ROOT_DIR``, read by
:class:`rdagent.scenarios.crypto.conf.CryptoSettings`.
"""

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class CryptoPropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_", protected_namespaces=())

    scen: str = "rdagent.scenarios.crypto.experiment.factor_experiment.CryptoFactorScenario"
    """Scenario class for the crypto perpetual futures market."""

    hypothesis_gen: str = "rdagent.scenarios.crypto.proposal.factor_proposal.CryptoFactorHypothesisGen"
    """Hypothesis generation class."""

    hypothesis2experiment: str = "rdagent.scenarios.crypto.proposal.factor_proposal.CryptoFactorHypothesis2Experiment"
    """Hypothesis to experiment class."""

    coder: str | None = None
    """Not wired yet: the loop stops before any factor code is written."""

    runner: str | None = None
    """Not wired yet: there is no crypto evaluation to run the factors on."""

    summarizer: str | None = None
    """Not wired yet: without an evaluation there is nothing to summarise."""

    with_runtime_environment: bool = False
    """Execute the factor environment to list its packages when rendering the scenario.

    Off by default, like the scenario's own CLI: rendering the background should not need a configured
    environment.
    """

class StrategyBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    available_schemas: Annotated[list[str], NoDecode] = ["binance.market.crypto.derivative_bar.v1.json"]
    """Schema files under ``<data root>/schemas`` that a strategy prompt may describe.

    Override with a comma-separated list in ``QLIB_FACTOR_AVAILABLE_SCHEMAS``. The scenario renders
    these files and no other schema file.
    """

    @field_validator("available_schemas", mode="before")
    @classmethod
    def _split_available_schemas(cls, value: object) -> object:
        """``a.json, b.json`` is the two schema files ``a.json`` and ``b.json``."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # 1) override base settings
    scen: str = "rdagent.scenarios.crypto.experiment.strategy_experiment.CryptoStrategyScenario"
    """Scenario class for Qlib Factor"""

    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""

    train_start: str = "2008-01-01"
    """Start date of the training segment"""

    train_end: str = "2014-12-31"
    """End date of the training segment"""

    valid_start: str = "2015-01-01"
    """Start date of the validation segment"""

    valid_end: str = "2016-12-31"
    """End date of the validation segment"""

    test_start: str = "2017-01-01"
    """Start date of the test / backtest segment"""

    test_end: str | None = "2020-08-01"
    """End date of the test / backtest segment"""


CRYPTO_PROP_SETTING = CryptoPropSetting()
