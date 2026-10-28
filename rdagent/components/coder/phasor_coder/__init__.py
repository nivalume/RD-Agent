"""The phasor factor coder: it writes one factor, expression or numpy, per task.

The public entry point is :class:`PhasorCoSTEER`, which plugs the phasor prompts,
the spec validator and the code-generation evaluator into the CoSTEER loop::

    coder = PhasorCoSTEER(scen=CryptoFactorScenario())
    exp = coder.develop(exp)

A task is a :class:`PhasorFactorTask`; its workspace holds ``factor.json``.

Render the implementation prompt and call the model once, for debugging::

    python -m rdagent.components.coder.phasor_coder
    python -m rdagent.components.coder.phasor_coder --print-only
"""

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
from rdagent.components.coder.phasor_coder.spec import (
    FactorSpec,
    FactorSpecError,
    parse_factor_spec,
    validate_factor_spec,
)
from rdagent.components.coder.phasor_coder.task import PhasorFactorTask
from rdagent.components.coder.phasor_coder.workspace import PhasorFactorWorkspace
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
        eva = CoSTEERMultiEvaluator(
            PhasorFactorEvaluatorForCoder(scen=scen, settings=settings), scen=scen
        )
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


__all__ = [
    "PhasorCoSTEER",
    "PhasorCoSTEERSettings",
    "PHASOR_COSTEER_SETTINGS",
    "PhasorFactorTask",
    "PhasorFactorWorkspace",
    "PhasorFactorEvaluatorForCoder",
    "PhasorMultiProcessEvolvingStrategy",
    "FactorSpec",
    "FactorSpecError",
    "parse_factor_spec",
    "validate_factor_spec",
]
