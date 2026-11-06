"""Hypotheses to crypto qlib experiments.

Factors keep the crypto factor proposal -- its prompts and its rule that a factor already in the
library is not proposed again -- and only build qlib experiments. Models are converted here, on the
shared :class:`ModelHypothesis2Experiment`, with the qlib model prompts: what the model may be, the
trace and the answer format do not depend on the market.
"""

import json
from typing import Tuple

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.components.proposal import ModelHypothesis2Experiment
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.crypto.proposal.factor_proposal import (
    CryptoFactorHypothesis2Experiment,
)
from rdagent.scenarios.crypto.qlib.experiment import (
    CryptoQlibFactorExperiment,
    CryptoQlibModelExperiment,
    CryptoQlibQuantScenario,
)
from rdagent.utils.agent.tpl import T

QLIB_PROMPTS = "scenarios.qlib.prompts"
"""The qlib proposal prompts: the trace, the SOTA round and the model experiment format."""

MODEL_RAG = (
    "Note, the training data consists of less than 1 million samples for the training set and approximately "
    "250,000 samples for the validation set. Please design the hyperparameters accordingly and control the model "
    "size. This has a significant impact on the training results. If you believe that the previous model itself "
    "is good but the training hyperparameters or model hyperparameters are not optimal, you can return the same "
    "model and adjust these parameters instead."
)


class CryptoQlibFactorHypothesis2Experiment(CryptoFactorHypothesis2Experiment):
    experiment_cls = CryptoQlibFactorExperiment


class CryptoQlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    """A model hypothesis to one model task, run on the base features and the SOTA factors."""

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        if isinstance(trace.scen, CryptoQlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="model")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        # The model rounds of the trace (a quant trace holds factor rounds too), oldest first; the prompt
        # lists them newest first, as the qlib model proposal does.
        rounds = [
            (experiment, feedback)
            for experiment, feedback in trace.hist
            if getattr(experiment.hypothesis, "action", "model") == "model"
        ]
        last = rounds[-1] if rounds else None
        sota = next(
            ((experiment, feedback) for experiment, feedback in reversed(rounds) if feedback.decision is True), None
        )

        if not trace.hist:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        elif rounds:
            model_trace = Trace(trace.scen)
            model_trace.hist = rounds[::-1]
            hypothesis_and_feedback = T(f"{QLIB_PROMPTS}:hypothesis_and_feedback").r(trace=model_trace)
        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available."

        last_hypothesis_and_feedback = (
            T(f"{QLIB_PROMPTS}:last_hypothesis_and_feedback").r(experiment=last[0], feedback=last[1])
            if last is not None
            else "No previous hypothesis and feedback available since it's the first round."
        )
        sota_hypothesis_and_feedback = (
            T(f"{QLIB_PROMPTS}:sota_hypothesis_and_feedback").r(experiment=sota[0], feedback=sota[1])
            if sota is not None
            else "No SOTA hypothesis and feedback available since previous experiments were not accepted."
        )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "experiment_output_format": T(f"{QLIB_PROMPTS}:model_experiment_output_format").r(),
            "target_list": [],
            "RAG": MODEL_RAG,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        tasks = [
            ModelTask(
                name=name,
                description=answer["description"],
                formulation=answer["formulation"],
                architecture=answer["architecture"],
                variables=answer["variables"],
                hyperparameters=answer["hyperparameters"],
                training_hyperparameters=answer["training_hyperparameters"],
                model_type=answer["model_type"],
            )
            for name, answer in json.loads(response).items()
        ]
        experiment = CryptoQlibModelExperiment(tasks, hypothesis=hypothesis)
        # Every accepted round, the factor ones of a quant trace included: the model runner takes the SOTA
        # factors from them.
        experiment.based_experiments = [done for done, feedback in trace.hist if feedback]
        return experiment


__all__ = ["CryptoQlibFactorHypothesis2Experiment", "CryptoQlibModelHypothesis2Experiment"]
