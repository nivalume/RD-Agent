"""The phasor coder's CoSTEER loop."""

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.phasor_coder.config import (
    PHASOR_COSTEER_SETTINGS,
    PhasorCoSTEERSettings,
)
from rdagent.components.coder.phasor_coder.evaluators import PhasorFactorEvaluatorForCoder
from rdagent.components.coder.phasor_coder.evolving_strategy import (
    PhasorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario


class PhasorCoSTEER(CoSTEER):
    """Implement factors for phasor. One task is one factor at one bar frequency."""

    def __init__(
        self,
        scen: Scenario,
        *args,
        settings: PhasorCoSTEERSettings | None = None,
        **kwargs,
    ) -> None:
        settings = settings or PHASOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(PhasorFactorEvaluatorForCoder(scen=scen, settings=settings), scen=scen)
        es = PhasorMultiProcessEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            **kwargs,
        )

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                exp.prop_dev_feedback = self.evolve_agent.evolving_trace[-1].feedback
        return exp
