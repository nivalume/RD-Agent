"""The strategy coder: it writes one strategy, a phasor expression or numpy code, per task.

The phasor coder's machinery (the spec, its parser, the CoSTEER loop and its knowledge) with the
strategy prompts, where the value is a target position in [-1, 1], and an evaluator that runs the
strategy on the debug panel (bounds, not flat, no look-ahead)::

    coder = StrategyCoSTEER(scen=CryptoStrategyBacktestScenario())
    exp = coder.develop(exp)
"""

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.strategy_coder.config import (
    STRATEGY_COSTEER_SETTINGS,
    StrategyCoSTEERSettings,
)
from rdagent.components.coder.strategy_coder.evaluators import StrategyEvaluatorForCoder
from rdagent.components.coder.strategy_coder.evolving_strategy import StrategyEvolvingStrategy
from rdagent.components.coder.strategy_coder.workspace import StrategyWorkspace
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario


class StrategyCoSTEER(CoSTEER):
    def __init__(self, scen: Scenario, *args, settings: StrategyCoSTEERSettings | None = None, **kwargs) -> None:
        settings = settings or STRATEGY_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(StrategyEvaluatorForCoder(scen=scen, settings=settings), scen=scen)
        es = StrategyEvolvingStrategy(scen=scen, settings=settings)
        super().__init__(*args, settings=settings, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                exp.prop_dev_feedback = self.evolve_agent.evolving_trace[-1].feedback
        return exp


__all__ = [
    "STRATEGY_COSTEER_SETTINGS",
    "StrategyCoSTEER",
    "StrategyCoSTEERSettings",
    "StrategyEvaluatorForCoder",
    "StrategyEvolvingStrategy",
    "StrategyWorkspace",
]
