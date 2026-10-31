"""The strategy runner and the feedback on its results.

The runner backtests each coded strategy on the full panel, in a child process (python-mode code was
written by the model). The feedback applies the hard gates, picks the round's best strategy by valid
alpha Sharpe, compares it with the best so far, and asks the model to explain the outcome. The
model never sees the test segment while ``hide_test`` is on.
"""

from __future__ import annotations

import json
import math
from typing import Any

from rdagent.core.developer import Developer
from rdagent.core.exception import RunnerError
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.crypto.experiment.strategy_experiment import CryptoStrategyExperiment
from rdagent.scenarios.crypto.strategy import results as R
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.engine import without_test
from rdagent.scenarios.crypto.strategy.run import SPEC_FILE, run_in_process
from rdagent.utils.agent.tpl import T


def _coded(exp: CryptoStrategyExperiment, index: int) -> bool:
    feedback = getattr(exp, "prop_dev_feedback", None)
    if feedback is None:
        return True
    try:
        single = feedback[index]
    except (IndexError, TypeError):
        return True
    return single is None or bool(single.final_decision)


class StrategyRunner(Developer[CryptoStrategyExperiment]):
    """Backtest every strategy the coder wrote; the results go on the experiment:

    - ``strategy_results``: name -> the whole evaluation, test segment included (for the report);
    - ``result``: name -> its summary, without the test segment while ``hide_test`` is on (the trace
      prints it into every later prompt);
    - ``trial_sharpes``: the per-bar valid Sharpe of every case, for later deflated Sharpes.

    ``exp.prior_trial_sharpes``, set by the loop from the whole trace, are the earlier trials.
    """

    def __init__(self, scen: Scenario, settings: CryptoStrategySettings | None = None) -> None:
        super().__init__(scen)
        self.settings = settings

    def develop(self, exp: CryptoStrategyExperiment) -> CryptoStrategyExperiment:
        settings = self.settings or CryptoStrategySettings()
        trials = list(getattr(exp, "prior_trial_sharpes", None) or [])
        found: dict[str, dict[str, Any]] = {}
        own_trials: list[float] = []
        for index, (task, workspace) in enumerate(zip(exp.sub_tasks, exp.sub_workspace_list)):
            spec = workspace.file_dict.get(SPEC_FILE) if workspace is not None else None
            if not spec or not _coded(exp, index):
                found[task.strategy_name] = {"ok": False, "problems": ["the coder did not produce a working strategy"]}
                continue
            workspace.workspace_path.mkdir(parents=True, exist_ok=True)
            (workspace.workspace_path / SPEC_FILE).write_text(spec)
            result = run_in_process(
                "backtest",
                workspace.workspace_path,
                settings.panel_path,
                settings.python_bin,
                settings.timeout,
                trials=trials + own_trials,
            )
            logger.info(f"Backtest of {task.strategy_name}: {json.dumps(R.summary(result, settings), default=str)}")
            found[task.strategy_name] = result
            own_trials += R.per_bar_trials(result)
        if not any(result.get("ok") for result in found.values()):
            problems = "\n".join(f"- {name}: {result.get('problems')}" for name, result in found.items())
            raise RunnerError(f"No strategy could be backtested:\n{problems}")
        exp.strategy_results = found
        exp.trial_sharpes = own_trials
        exp.result = {name: R.summary(result, settings) for name, result in found.items()}
        return exp


def best_of(exp: Any, settings: CryptoStrategySettings) -> tuple[str, float] | None:
    """The experiment's strategy with the highest valid alpha Sharpe among those that passed the gates."""
    passing = {
        name: R.score(result)
        for name, result in (getattr(exp, "strategy_results", None) or {}).items()
        if not R.gate_failures(result, settings)
    }
    if not passing:
        return None
    name = max(passing, key=passing.get)
    return name, passing[name]


def sota_of(trace: Trace, settings: CryptoStrategySettings) -> tuple[Any, str, float] | None:
    """The accepted experiment holding the best strategy so far: (experiment, name, score)."""
    found = None
    for exp, feedback in trace.hist:
        if not (feedback and feedback.decision):
            continue
        best = best_of(exp, settings)
        if best is not None and (found is None or best[1] > found[2]):
            found = (exp, *best)
    return found


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


class StrategyExperiment2Feedback(Experiment2Feedback):
    """The gates and the comparison decide; the model explains the outcome and proposes the next step."""

    def __init__(self, scen: Scenario, settings: CryptoStrategySettings | None = None) -> None:
        super().__init__(scen)
        self.settings = settings

    def shown(self, result: dict[str, Any]) -> dict[str, Any]:
        """The evaluation the model may see."""
        settings = self.settings or CryptoStrategySettings()
        if not result.get("ok"):
            return result
        visible = without_test(result) if settings.hide_test else result
        return _rounded({key: value for key, value in visible.items() if key not in ("ok", "seconds")})

    def prompts(self, exp: CryptoStrategyExperiment, trace: Trace) -> tuple[str, str, bool]:
        settings = self.settings or CryptoStrategySettings()
        found = getattr(exp, "strategy_results", None) or {}
        strategies = []
        for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
            result = found.get(task.strategy_name, {"ok": False, "problems": ["not backtested"]})
            strategies.append(
                {
                    "name": task.strategy_name,
                    "information": task.get_task_information(),
                    "code": (workspace.file_dict.get(SPEC_FILE, "") if workspace is not None else "") or "(none)",
                    "evaluation": json.dumps(self.shown(result), indent=1),
                    "failures": R.gate_failures(result, settings),
                }
            )
        best = best_of(exp, settings)
        sota = sota_of(trace, settings)
        decision = best is not None and (sota is None or best[1] > sota[2])
        if best is None:
            outcome = "Rejected: no strategy of this round passed the hard gates."
        elif decision:
            outcome = f"Accepted: {best[0]} passed the gates with a valid alpha Sharpe of {best[1]:.2f}" + (
                f", above the best so far ({sota[1]}, {sota[2]:.2f})." if sota else "; it is the first accepted."
            )
        else:
            outcome = (
                f"Rejected: {best[0]} passed the gates, but its valid alpha Sharpe {best[1]:.2f} does not beat "
                f"the best so far ({sota[1]}, {sota[2]:.2f})."
            )
        sota_text = None
        if sota is not None:
            sota_exp, name, _ = sota
            task = next((task for task in sota_exp.sub_tasks if task.strategy_name == name), None)
            sota_text = (
                (task.get_task_information() if task is not None else name)
                + "\n"
                + json.dumps(R.summary(sota_exp.strategy_results[name], settings), indent=1)
            )
        system = T(".prompts:strategy_feedback_system").r(scenario=self.scen.get_scenario_all_desc())
        user = T(".prompts:strategy_feedback_user").r(
            hypothesis=exp.hypothesis, strategies=strategies, outcome=outcome, sota=sota_text
        )
        return system, user + "", decision

    def generate_feedback(self, exp: CryptoStrategyExperiment, trace: Trace, exception=None) -> HypothesisFeedback:
        settings = self.settings or CryptoStrategySettings()
        system, user, decision = self.prompts(exp, trace)
        try:
            answer = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user, system_prompt=system, json_mode=True, json_target_type=dict[str, str]
                )
            )
        except Exception as error:  # the decision stands without the model's explanation
            logger.warning(f"The strategy feedback could not be written by the model: {error}")
            answer = {}
        passing = best_of(exp, settings) is not None
        return HypothesisFeedback(
            observations=str(answer.get("observations", "")),
            hypothesis_evaluation=str(answer.get("hypothesis_evaluation", "")),
            new_hypothesis=str(answer.get("new_hypothesis", "")),
            reason=str(answer.get("reason", "")) or user.rsplit("# Outcome", 1)[-1].strip().splitlines()[0],
            decision=decision,
            acceptable=passing,
        )
