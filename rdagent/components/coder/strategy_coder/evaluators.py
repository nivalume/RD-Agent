"""Checks that what the coder wrote is a usable strategy, on the debug panel.

The spec is parsed (and an expression compiled) first; then the strategy runs in a child process
on the debug slice of the panel: its positions must have the panel's shape, lie in [-1, 1], not be
flat everywhere, and not move when later rows are removed. Its returns are not judged here.
"""

from __future__ import annotations

from rdagent.components.coder.CoSTEER.evaluators import CoSTEEREvaluator, CoSTEERSingleFeedback
from rdagent.components.coder.phasor_coder.spec import FactorSpecError, parse_factor_spec
from rdagent.components.coder.strategy_coder.config import (
    STRATEGY_COSTEER_SETTINGS,
    StrategyCoSTEERSettings,
)
from rdagent.core.experiment import Task, Workspace
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.run import SPEC_FILE, run_in_process


class StrategyEvaluatorForCoder(CoSTEEREvaluator):
    def __init__(self, scen, settings: StrategyCoSTEERSettings | None = None) -> None:
        super().__init__(scen)
        self.settings = settings or STRATEGY_COSTEER_SETTINGS

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
        raw = implementation.file_dict.get(SPEC_FILE, "")
        if not raw:
            return self._fail(f"{SPEC_FILE} is missing; the answer must be one JSON object")
        try:
            spec = parse_factor_spec(
                raw,
                available_fields=list(self.settings.fields),
                max_params=self.settings.max_params,
                max_cases=self.settings.max_cases,
            )
        except FactorSpecError as error:
            return self._fail(str(error))
        strategy = CryptoStrategySettings()
        result = run_in_process(
            "check",
            implementation.workspace_path,
            strategy.debug_panel_path,
            strategy.python_bin,
            self.settings.check_timeout,
        )
        if not result.get("ok"):
            problems = "\n".join(f"- {problem}" for problem in result.get("problems", ["unknown failure"]))
            return self._fail(f"The strategy failed its checks on the debug panel:\n{problems}")
        cases = f", {spec.cases} cases" if spec.params else ""
        return CoSTEERSingleFeedback(
            execution=(
                f"{SPEC_FILE} runs ({spec.mode} mode{cases}) in {result.get('seconds', 0):.1f}s on the debug "
                f"panel without look-ahead; in the market on {result.get('in_market', 0):.0%} of the pool's "
                f"bars, mean |position| {result.get('mean_abs_position', 0):.2f}."
            ),
            return_checking=None,
            code="",
            final_decision=True,
        )

    @staticmethod
    def _fail(message: str) -> CoSTEERSingleFeedback:
        return CoSTEERSingleFeedback(execution=message, return_checking=None, code="", final_decision=False)
