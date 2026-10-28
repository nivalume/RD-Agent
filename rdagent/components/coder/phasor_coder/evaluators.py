from __future__ import annotations

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.phasor_coder.config import (
    PHASOR_COSTEER_SETTINGS,
    PhasorCoSTEERSettings,
)
from rdagent.components.coder.phasor_coder.spec import (
    FactorSpec,
    FactorSpecError,
    parse_factor_spec,
)
from rdagent.core.experiment import Task, Workspace


class PhasorFactorEvaluatorForCoder(CoSTEEREvaluator):
    """Checks that what the coder wrote is a usable phasor factor.

    Parsing ``factor.json`` compiles an expression with phasor
    (:meth:`rdagent.components.coder.phasor_coder.spec.parse_factor_spec`).
    Executing the factor on data and judging its quality is a later stage; it
    stays out of this evaluator so the loop that writes code is not slowed
    down by a backtest.
    """

    def __init__(
        self,
        scen,
        settings: PhasorCoSTEERSettings | None = None,
    ) -> None:
        super().__init__(scen)
        self.settings = settings or PHASOR_COSTEER_SETTINGS

    def _available_fields(self, target_task: Task) -> list[str]:
        fields = getattr(target_task, "fields", None) or self.settings.fields
        return list(fields)

    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace | None = None,
        queried_knowledge=None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        if implementation is None:
            return self._fail("the task has no workspace; nothing was written")

        raw = implementation.file_dict.get("factor.json", "")
        if not raw:
            return self._fail("factor.json is missing; the answer must be one JSON object")

        try:
            spec = parse_factor_spec(
                raw,
                available_fields=self._available_fields(target_task),
                max_params=self.settings.max_params,
                max_cases=self.settings.max_cases,
            )
        except FactorSpecError as error:
            return self._fail(str(error))

        return self._pass(spec)

    @staticmethod
    def _fail(message: str) -> CoSTEERSingleFeedback:
        return CoSTEERSingleFeedback(
            execution=message,
            return_checking=None,
            code="",
            final_decision=False,
        )

    @staticmethod
    def _pass(spec: FactorSpec) -> CoSTEERSingleFeedback:
        cases = f", {spec.cases} cases" if spec.params else ""
        return CoSTEERSingleFeedback(
            execution=f"factor.json is valid ({spec.mode} mode{cases}).",
            return_checking=None,
            code="",
            final_decision=True,
        )
