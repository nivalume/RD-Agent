"""Hypotheses to crypto qlib experiments.

Factors keep the crypto factor proposal -- its prompts and its rule that a factor already in the
library is not proposed again -- and only build qlib experiments. Models keep the qlib model proposal
and build crypto qlib experiments.
"""

from rdagent.components.coder.model_coder.model import ModelExperiment
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.crypto.proposal.factor_proposal import (
    CryptoFactorHypothesis2Experiment,
)
from rdagent.scenarios.crypto.qlib.experiment import (
    CryptoQlibFactorExperiment,
    CryptoQlibModelExperiment,
)
from rdagent.scenarios.qlib.proposal.model_proposal import (
    QlibModelHypothesis2Experiment,
)


class CryptoQlibFactorHypothesis2Experiment(CryptoFactorHypothesis2Experiment):
    experiment_cls = CryptoQlibFactorExperiment


class CryptoQlibModelHypothesis2Experiment(QlibModelHypothesis2Experiment):
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        built = super().convert_response(response, hypothesis, trace)
        experiment = CryptoQlibModelExperiment(built.sub_tasks, hypothesis=hypothesis)
        experiment.based_experiments = built.based_experiments
        return experiment


__all__ = ["CryptoQlibFactorHypothesis2Experiment", "CryptoQlibModelHypothesis2Experiment"]
