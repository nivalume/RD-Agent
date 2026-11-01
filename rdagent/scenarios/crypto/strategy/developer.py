"""The strategy runner and the feedback on its results.

The runner backtests each coded strategy on the full panel, in a child process (python-mode code was
written by the model). The feedback applies the hard gates, measures what each strategy that passed
adds to the book of accepted strategies (its correlation with them and the book's valid Sharpe with
it), accepts the one that adds most if it adds enough, and asks the model to explain the outcome.
The model never sees the test segment while ``hide_test`` is on.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

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
from rdagent.scenarios.crypto.strategy.run import RETURNS_FILE, SPEC_FILE, run_in_process
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


def seen_returns(path) -> pd.DataFrame | None:
    """The train and valid rows of a backtest's returns file, as float32."""
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    frame = frame[frame["segment"].isin(["train", "valid"])]
    return pd.DataFrame(
        {"returns": frame["returns"].astype(np.float32), "segment": frame["segment"].astype("category")},
        index=frame.index,
    )


class StrategyRunner(Developer[CryptoStrategyExperiment]):
    """Backtest every strategy the coder wrote; the results go on the experiment:

    - ``strategy_results``: name -> the whole evaluation, test segment included (for the report);
    - ``result``: name -> its summary, without the test segment while ``hide_test`` is on (the trace
      prints it into every later prompt);
    - ``trial_sharpes``: the per-bar valid Sharpe of every case, for later deflated Sharpes;
    - ``strategy_returns``: name -> the chosen case's returns on the train and valid segments
      (``returns``, ``segment``), what the book is built from. Test bars are not kept here.

    ``exp.prior_trial_sharpes``, set by the loop from the whole trace, are the earlier trials.
    """

    def __init__(self, scen: Scenario, settings: CryptoStrategySettings | None = None) -> None:
        super().__init__(scen)
        self.settings = settings

    def develop(self, exp: CryptoStrategyExperiment) -> CryptoStrategyExperiment:
        settings = self.settings or CryptoStrategySettings()
        trials = list(getattr(exp, "prior_trial_sharpes", None) or [])
        found: dict[str, dict[str, Any]] = {}
        series: dict[str, pd.DataFrame] = {}
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
            kept = seen_returns(workspace.workspace_path / RETURNS_FILE) if result.get("ok") else None
            if kept is not None:
                series[task.strategy_name] = kept
        if not any(result.get("ok") for result in found.values()):
            problems = "\n".join(f"- {name}: {result.get('problems')}" for name, result in found.items())
            raise RunnerError(f"No strategy could be backtested:\n{problems}")
        exp.strategy_results = found
        exp.strategy_returns = series
        exp.trial_sharpes = own_trials
        exp.result = {name: R.summary(result, settings) for name, result in found.items()}
        return exp


def book_of(trace: Trace) -> list[tuple[Any, str]]:
    """The accepted strategies, in the order they joined: (experiment, strategy name)."""
    found = []
    for exp, feedback in trace.hist:
        name = getattr(exp, "accepted_strategy", None)
        if feedback and feedback.decision and name and name in (getattr(exp, "strategy_returns", None) or {}):
            found.append((exp, name))
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

    def judge(self, exp: CryptoStrategyExperiment, trace: Trace) -> dict[str, Any]:
        """The gates, each passing strategy against the book, and the one accepted (or None)."""
        settings = self.settings or CryptoStrategySettings()
        found = getattr(exp, "strategy_results", None) or {}
        series = getattr(exp, "strategy_returns", None) or {}
        book = book_of(trace)
        members = [member.strategy_returns[name] for member, name in book]
        judged: dict[str, dict[str, Any]] = {}
        for name, result in found.items():
            entry: dict[str, Any] = {"failures": R.gate_failures(result, settings), "warnings": R.warnings(result)}
            if not entry["failures"] and name in series:
                entry["contribution"] = R.contribution(members, series[name], float(result["bars_per_year"]))
                entry["not_new"] = R.accepts(entry["contribution"], len(members), settings)
            judged[name] = entry
        eligible = {
            name: float(entry["contribution"].get("gain") or float("-inf"))
            for name, entry in judged.items()
            if "contribution" in entry and not entry["not_new"]
        }
        accepted = max(eligible, key=eligible.get) if eligible else None
        return {"strategies": judged, "accepted": accepted, "book": book}

    def prompts(self, exp: CryptoStrategyExperiment, trace: Trace) -> tuple[str, str, str | None]:
        settings = self.settings or CryptoStrategySettings()
        found = getattr(exp, "strategy_results", None) or {}
        verdict = self.judge(exp, trace)
        strategies = []
        for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
            result = found.get(task.strategy_name, {"ok": False, "problems": ["not backtested"]})
            entry = verdict["strategies"].get(task.strategy_name, {"failures": R.gate_failures(result, settings)})
            strategies.append(
                {
                    "name": task.strategy_name,
                    "information": task.get_task_information(),
                    "code": (workspace.file_dict.get(SPEC_FILE, "") if workspace is not None else "") or "(none)",
                    "evaluation": json.dumps(self.shown(result), indent=1),
                    "failures": entry.get("failures", []),
                    "warnings": entry.get("warnings", []),
                    "contribution": json.dumps(entry["contribution"]) if "contribution" in entry else "",
                    "not_new": entry.get("not_new", []),
                }
            )
        accepted, book = verdict["accepted"], verdict["book"]
        passed = [name for name, entry in verdict["strategies"].items() if not entry["failures"]]
        if accepted is not None:
            gain = verdict["strategies"][accepted]["contribution"].get("gain")
            outcome = f"Accepted: {accepted} passed the gates and joins the book" + (
                f", raising its valid Sharpe by {gain:+.2f}." if book else " as its first strategy."
            )
        elif passed:
            outcome = f"Rejected: {', '.join(passed)} passed the gates but add too little to the book."
        else:
            outcome = "Rejected: no strategy of this round passed the hard gates."
        book_text = None
        if book:
            lines = []
            for member, name in book:
                task = next((task for task in member.sub_tasks if task.strategy_name == name), None)
                summary = R.summary(member.strategy_results[name], settings)
                valid = summary.get("valid", {})
                lines.append(
                    f"- {name}: {task.description if task is not None else ''} (valid Sharpe {valid.get('sharpe')}, "
                    f"alpha Sharpe {valid.get('alpha_sharpe')})"
                )
            book_text = "\n".join(lines)
        system = T(".prompts:strategy_feedback_system").r(scenario=self.scen.get_scenario_all_desc())
        user = T(".prompts:strategy_feedback_user").r(
            hypothesis=exp.hypothesis, strategies=strategies, outcome=outcome, book=book_text
        )
        return system, user, accepted

    def generate_feedback(self, exp: CryptoStrategyExperiment, trace: Trace, exception=None) -> HypothesisFeedback:
        system, user, accepted = self.prompts(exp, trace)
        exp.accepted_strategy = accepted
        exp.judged = self.judge(exp, trace)["strategies"]
        try:
            answer = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user, system_prompt=system, json_mode=True, json_target_type=dict[str, str]
                )
            )
        except Exception as error:  # the decision stands without the model's explanation
            logger.warning(f"The strategy feedback could not be written by the model: {error}")
            answer = {}
        settings = self.settings or CryptoStrategySettings()
        passing = any(not R.gate_failures(result, settings) for result in (exp.strategy_results or {}).values())
        return HypothesisFeedback(
            observations=str(answer.get("observations", "")),
            hypothesis_evaluation=str(answer.get("hypothesis_evaluation", "")),
            new_hypothesis=str(answer.get("new_hypothesis", "")),
            reason=str(answer.get("reason", "")) or user.rsplit("# Outcome", 1)[-1].strip().splitlines()[0],
            decision=accepted is not None,
            acceptable=passing,
        )
