"""The crypto RD loop's settings: the classes the loop wires and how the scenario is rendered.

Only the proposal side is wired: the crypto scenario has no coder, runner or summarizer yet, so those
three settings stay ``None`` and :class:`rdagent.app.crypto_rd_loop.factor.CryptoFactorRDLoop` runs
hypothesis generation and hypothesis-to-experiment conversion only.

Every field can be overridden with a ``CRYPTO_`` prefixed environment variable (``CRYPTO_SCEN``,
``CRYPTO_HYPOTHESIS_GEN``, ...). The data root keeps its own setting, ``CRYPTO_DATA_ROOT_DIR``, read by
:class:`rdagent.scenarios.crypto.conf.CryptoSettings`.
"""

from pydantic_settings import SettingsConfigDict

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


CRYPTO_PROP_SETTING = CryptoPropSetting()
