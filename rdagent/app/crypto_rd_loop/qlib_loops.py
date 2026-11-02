"""The loops of the crypto qlib app: factor, model, and the one that picks between the two.

They are this app's own classes (the qlib app's loops are not imported), built on the shared
:class:`~rdagent.components.workflow.rd_loop.RDLoop` steps. They live in a module of their own, not
in the ``python -m`` entry point, so that a dumped session can be loaded again by class path.
"""

import asyncio
from typing import Any

from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentPlan,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace
from rdagent.utils.qlib import ALPHA20

from .qlib_conf import CryptoQlibQuantPropSetting


class CryptoQlibFactorRDLoop(RDLoop):
    """Propose factors, code them, run them with LightGBM on qlib, and get the feedback."""

    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error("Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp


class CryptoQlibModelRDLoop(RDLoop):
    """Propose a model, code it, and train it on the base features on qlib."""

    skip_loop_error = (ModelEmptyError,)


class CryptoQlibQuantRDLoop(RDLoop):
    """Each round picks a factor or a model step, then runs that step's coder, runner and feedback."""

    skip_loop_error = (FactorEmptyError, ModelEmptyError)

    def __init__(self, prop_setting: CryptoQlibQuantPropSetting) -> None:
        scen: Scenario = import_class(prop_setting.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(prop_setting.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        self.hypothesis_gen: HypothesisGen = import_class(prop_setting.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator")

        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            prop_setting.factor_hypothesis2experiment
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            prop_setting.model_hypothesis2experiment
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")

        self.factor_coder: Developer = import_class(prop_setting.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="factor coder")
        self.model_coder: Developer = import_class(prop_setting.model_coder)(scen)
        logger.log_object(self.model_coder, tag="model coder")

        self.factor_runner: Developer = import_class(prop_setting.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="factor runner")
        self.model_runner: Developer = import_class(prop_setting.model_runner)(scen)
        logger.log_object(self.model_runner, tag="model runner")

        self.factor_summarizer: Experiment2Feedback = import_class(prop_setting.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="factor summarizer")
        self.model_summarizer: Experiment2Feedback = import_class(prop_setting.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="model summarizer")

        self.plan: ExperimentPlan = {"features": ALPHA20, "feature_codes": {}}  # for user interaction
        self.trace = QuantTrace(scen=scen)
        # RDLoop.__init__ builds the single-action wiring above; only the base loop is set up here.
        super(RDLoop, self).__init__()

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                assert hypo.action in ["factor", "model"]
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                exp.base_features = self.plan["features"]
                exp.base_feature_codes = self.plan["feature_codes"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                    exp.based_experiments[-1].base_feature_codes = self.plan["feature_codes"]
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        coder = self.factor_coder if prev_out["direct_exp_gen"]["propose"].action == "factor" else self.model_coder
        exp = coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_runner.develop(prev_out["coding"])
            if exp is None:
                logger.error("Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
        else:
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
        else:
            summarizer = (
                self.factor_summarizer
                if prev_out["direct_exp_gen"]["propose"].action == "factor"
                else self.model_summarizer
            )
            feedback = summarizer.generate_feedback(prev_out["running"], self.trace)
        feedback = self._interact_feedback(feedback)
        logger.log_object(feedback, tag="feedback")
        return feedback
