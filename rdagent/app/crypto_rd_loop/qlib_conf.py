"""The crypto qlib loops' settings: the classes each loop wires.

They are the qlib loops' settings with the crypto scenario, proposal and experiments; the coders,
runners and feedback are the qlib ones. Each field can be overridden with an environment variable:
``CRYPTO_QLIB_FACTOR_`` for the factor loop, ``CRYPTO_QLIB_MODEL_`` for the model loop and
``CRYPTO_QLIB_QUANT_`` for the loop that picks between the two. The data, pool, backtest and time
segments are in :mod:`rdagent.scenarios.crypto.qlib.conf` (``CRYPTO_QLIB_``); the segments here only
mirror them, since the workspace passes those to every run.
"""

from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.app.qlib_rd_loop.conf import (
    FactorBasePropSetting,
    ModelBasePropSetting,
    QuantBasePropSetting,
)
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS

CRYPTO_QLIB = "rdagent.scenarios.crypto.qlib"
QLIB = "rdagent.scenarios.qlib"


_RENDERED = CRYPTO_QLIB_SETTINGS.run_env()


class _Segments:
    """The segments the workflows run with (qlib's test is the selection period); the workspace
    passes these itself, so these only mirror them."""

    train_start: str = _RENDERED["train_start"]
    train_end: str = _RENDERED["train_end"]
    valid_start: str = _RENDERED["valid_start"]
    valid_end: str = _RENDERED["valid_end"]
    test_start: str = _RENDERED["test_start"]
    test_end: Optional[str] = _RENDERED.get("test_end")


class CryptoQlibFactorPropSetting(FactorBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_QLIB_FACTOR_", protected_namespaces=())

    scen: str = f"{CRYPTO_QLIB}.experiment.CryptoQlibFactorScenario"
    hypothesis_gen: str = "rdagent.scenarios.crypto.proposal.factor_proposal.CryptoFactorHypothesisGen"
    hypothesis2experiment: str = f"{CRYPTO_QLIB}.proposal.CryptoQlibFactorHypothesis2Experiment"
    coder: str = f"{QLIB}.developer.factor_coder.QlibFactorCoSTEER"
    runner: str = f"{QLIB}.developer.factor_runner.QlibFactorRunner"
    summarizer: str = f"{QLIB}.developer.feedback.QlibFactorExperiment2Feedback"

    train_start: str = _Segments.train_start
    train_end: str = _Segments.train_end
    valid_start: str = _Segments.valid_start
    valid_end: str = _Segments.valid_end
    test_start: str = _Segments.test_start
    test_end: Optional[str] = _Segments.test_end


class CryptoQlibModelPropSetting(ModelBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_QLIB_MODEL_", protected_namespaces=())

    scen: str = f"{CRYPTO_QLIB}.experiment.CryptoQlibModelScenario"
    hypothesis_gen: str = f"{QLIB}.proposal.model_proposal.QlibModelHypothesisGen"
    hypothesis2experiment: str = f"{CRYPTO_QLIB}.proposal.CryptoQlibModelHypothesis2Experiment"
    coder: str = f"{QLIB}.developer.model_coder.QlibModelCoSTEER"
    runner: str = f"{QLIB}.developer.model_runner.QlibModelRunner"
    summarizer: str = f"{QLIB}.developer.feedback.QlibModelExperiment2Feedback"

    train_start: str = _Segments.train_start
    train_end: str = _Segments.train_end
    valid_start: str = _Segments.valid_start
    valid_end: str = _Segments.valid_end
    test_start: str = _Segments.test_start
    test_end: Optional[str] = _Segments.test_end


class CryptoQlibQuantPropSetting(QuantBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_QLIB_QUANT_", protected_namespaces=())

    scen: str = f"{CRYPTO_QLIB}.experiment.CryptoQlibQuantScenario"
    quant_hypothesis_gen: str = f"{QLIB}.proposal.quant_proposal.QlibQuantHypothesisGen"
    factor_hypothesis2experiment: str = f"{CRYPTO_QLIB}.proposal.CryptoQlibFactorHypothesis2Experiment"
    model_hypothesis2experiment: str = f"{CRYPTO_QLIB}.proposal.CryptoQlibModelHypothesis2Experiment"

    train_start: str = _Segments.train_start
    train_end: str = _Segments.train_end
    valid_start: str = _Segments.valid_start
    valid_end: str = _Segments.valid_end
    test_start: str = _Segments.test_start
    test_end: Optional[str] = _Segments.test_end


__all__ = ["CryptoQlibFactorPropSetting", "CryptoQlibModelPropSetting", "CryptoQlibQuantPropSetting"]
