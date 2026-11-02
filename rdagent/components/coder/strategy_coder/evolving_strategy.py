"""Asks the model for one strategy at a time and writes ``strategy.json``.

It is the phasor coder's evolving strategy -- the same spec, parser, retries and knowledge -- with
the strategy prompts: the output is a position, and the file is ``strategy.json``.
"""

from __future__ import annotations

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERKnowledge,
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.phasor_coder.evolving_strategy import (
    TICK,
    PhasorMultiProcessEvolvingStrategy,
)
from rdagent.components.coder.strategy_coder.workspace import StrategyWorkspace
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.crypto.strategy.run import SPEC_FILE
from rdagent.utils.agent.tpl import T

PHASOR_PROMPTS = "components.coder.phasor_coder.prompts"


class StrategyEvolvingStrategy(PhasorMultiProcessEvolvingStrategy):
    def _system_prompt(self, target_task: Task) -> str:
        listed = ", ".join(TICK + "$" + name + TICK for name in self._available_fields(target_task))
        return T(".prompts:strategy_implementation_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            data_contract=T(".prompts:strategy_data_contract").r(fields=listed),
            expression_reference=T(f"{PHASOR_PROMPTS}:phasor_expression_reference").r(fields=listed),
            python_reference=T(".prompts:strategy_python_reference").r(),
            output_format=T(".prompts:strategy_output_format").r(),
        )

    def _user_prompt(
        self,
        factor_information_str: str,
        successful_knowledge: list,
        error_knowledge: list,
        latest_execution,
        error_summary_critics: str | None,
    ) -> str:
        return T(".prompts:strategy_implementation_user").r(
            strategy_information_str=factor_information_str,
            queried_similar_successful_knowledge=successful_knowledge,
            queried_similar_error_knowledge=error_knowledge,
            error_summary_critics=error_summary_critics,
            latest_attempt_to_latest_successful_execution=latest_execution,
        )

    def error_summary(self, target_task: Task, failure: CoSTEERKnowledge, similar_errors: list) -> str:
        system_prompt = T(".prompts:strategy_error_summary_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            strategy_information_str=target_task.get_task_information(),
            latest_implementation=failure.implementation.file_dict.get(SPEC_FILE, ""),
            latest_feedback=str(failure.feedback),
        )
        rendered = list(similar_errors)
        user_prompt = T(".prompts:strategy_error_summary_user").r(similar_errors=rendered)
        while rendered and (
            APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
            >= APIBackend().chat_token_limit
        ):
            rendered = rendered[:-1]
            user_prompt = T(".prompts:strategy_error_summary_user").r(similar_errors=rendered)
        return APIBackend(use_chat_cache=self.settings.coder_use_cache).build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=False
        )

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        files = super().implement_one_task(target_task, queried_knowledge, workspace, prev_task_feedback)
        # The phasor coder names its artefact factor.json.
        return {SPEC_FILE: next(iter(files.values()))} if files else {}

    def assign_code_list_to_evo(self, code_list: list[dict | None], evo: EvolvingItem) -> EvolvingItem:
        for index, code in enumerate(code_list):
            if code is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = StrategyWorkspace(target_task=evo.sub_tasks[index])
            if code:
                evo.sub_workspace_list[index].inject_files(**code)
        return evo
