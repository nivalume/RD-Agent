"""The crypto RD loop: the proposal-side steps, the settings behind them and the record step."""

from rdagent.app.crypto_rd_loop.conf import CRYPTO_PROP_SETTING
from rdagent.app.crypto_rd_loop.factor import NOT_EVALUATED, CryptoFactorRDLoop
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.proposal import Trace
from rdagent.core.utils import import_class
from rdagent.scenarios.crypto.experiment.factor_experiment import CryptoFactorExperiment


def test_the_loop_runs_the_proposal_side_only():
    assert CryptoFactorRDLoop.steps == ["propose", "exp_gen", "record"]


def test_the_settings_wire_the_crypto_proposal_classes():
    assert import_class(CRYPTO_PROP_SETTING.scen).__name__ == "CryptoFactorScenario"
    assert import_class(CRYPTO_PROP_SETTING.hypothesis_gen).__name__ == "CryptoFactorHypothesisGen"
    assert import_class(CRYPTO_PROP_SETTING.hypothesis2experiment).__name__ == "CryptoFactorHypothesis2Experiment"


def test_the_settings_leave_the_development_side_unwired():
    # There is no crypto coder, runner or summarizer yet, so the loop must not claim one.
    assert CRYPTO_PROP_SETTING.coder is None
    assert CRYPTO_PROP_SETTING.runner is None
    assert CRYPTO_PROP_SETTING.summarizer is None


def test_the_scenario_is_rendered_without_the_factor_environment_by_default():
    assert CRYPTO_PROP_SETTING.with_runtime_environment is False


def test_record_keeps_the_experiment_as_unevaluated():
    loop = CryptoFactorRDLoop.__new__(CryptoFactorRDLoop)
    loop.trace = Trace(scen=None)
    experiment = CryptoFactorExperiment([FactorTask("Momentum_1D", "d", "f")])

    feedback = loop.record({"exp_gen": experiment, loop.LOOP_IDX_KEY: 0})

    assert feedback.reason == NOT_EVALUATED
    assert loop.trace.hist == [(experiment, feedback)]
    assert loop.trace.dag_parent == [()]
