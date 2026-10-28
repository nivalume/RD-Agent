from __future__ import annotations

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERKnowledge,
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.phasor_coder.config import PhasorCoSTEERSettings
from rdagent.components.coder.phasor_coder.spec import FactorSpecError, parse_factor_spec
from rdagent.components.coder.phasor_coder.workspace import PhasorFactorWorkspace
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

TICK = chr(96)


class PhasorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """Asks the model for one factor at a time and writes ``factor.json``.

    The model never writes the workspace layout: it answers with the JSON object
    described by phasor_output_format, this class validates it and renders the
    file through :meth:`FactorSpec.to_files`. A malformed answer is sent back
    with the parse error, which is cheaper than a workspace round trip.

    A failed attempt is quoted in the next prompt. When a similar failure was
    later fixed, one extra call writes short critics of that pattern. The
    summary itself is not allowed to return a factor.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings: PhasorCoSTEERSettings

    def _available_fields(self, target_task: Task) -> list[str]:
        fields = getattr(target_task, "fields", None) or self.settings.fields
        return list(fields)

    def _system_prompt(self, target_task: Task) -> str:
        fields = self._available_fields(target_task)
        listed = ", ".join(TICK + "$" + name + TICK for name in fields)
        return T(".prompts:evolving_strategy_factor_implementation_v1_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            data_contract=T(".prompts:phasor_data_contract").r(),
            expression_reference=T(".prompts:phasor_expression_reference").r(fields=listed),
            python_reference=T(".prompts:phasor_python_reference").r(),
            output_format=T(".prompts:phasor_output_format").r(),
        )

    def error_summary(
        self,
        target_task: Task,
        failure: CoSTEERKnowledge,
        similar_errors: list,
    ) -> str:
        """Critics of one failed factor.json, informed by similar fixes.

        ``similar_errors`` is the list stored on
        ``CoSTEERQueriedKnowledgeV2.task_to_similar_error_successful_knowledge``:
        ``(error text, (failed knowledge, fixed knowledge))``. The list is
        shortened from the end until the prompt fits; this does not change the
        list the implementation prompt still shows.
        """
        system_prompt = T(".prompts:evolving_strategy_error_summary_v2_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            factor_information_str=target_task.get_task_information(),
            latest_implementation=failure.implementation.file_dict.get("factor.json", ""),
            latest_feedback=str(failure.feedback),
        )
        rendered = list(similar_errors)
        user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(similar_errors=rendered)
        for _ in range(10):
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt, system_prompt=system_prompt
                )
                < APIBackend().chat_token_limit
            ):
                break
            if not rendered:
                break
            rendered = rendered[:-1]
            user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(similar_errors=rendered)
        return APIBackend(use_chat_cache=self.settings.coder_use_cache).build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

    def _user_prompt(
        self,
        factor_information_str: str,
        successful_knowledge: list,
        error_knowledge: list,
        latest_execution,
        error_summary_critics: str | None,
    ) -> str:
        return T(".prompts:evolving_strategy_factor_implementation_v2_user").r(
            factor_information_str=factor_information_str,
            queried_similar_successful_knowledge=successful_knowledge,
            queried_similar_error_knowledge=error_knowledge,
            error_summary_critics=error_summary_critics,
            latest_attempt_to_latest_successful_execution=latest_execution,
        )

    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        task_information = target_task.get_task_information()
        fields = self._available_fields(target_task)
        system_prompt = self._system_prompt(target_task)

        successful_knowledge: list = []
        # A list of (error text, (failed knowledge, fixed knowledge)), not a dict.
        # The knowledge base stores it that way; slicing it as a dict raises.
        error_knowledge: list = []
        former_failed: list = []
        latest_execution = None
        if queried_knowledge is not None:
            successful_knowledge = list(
                queried_knowledge.task_to_similar_task_successful_knowledge.get(task_information) or []
            )
            if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
                error_knowledge = list(
                    queried_knowledge.task_to_similar_error_successful_knowledge.get(task_information) or []
                )
            traces = queried_knowledge.task_to_former_failed_traces.get(task_information)
            if traces:
                former_failed = list(traces[0] or [])
                latest_execution = traces[1]

        # The latest-attempt slot stays empty unless
        # v2_add_fail_attempt_to_latest_successful_execution is on, so the last
        # former attempt is what the next prompt quotes. Critics are a second
        # call, and only when another factor was fixed the same way: with no
        # such fix the summary would restate the parser.
        failure = former_failed[-1] if former_failed else latest_execution
        shown_attempt = latest_execution if latest_execution is not None else failure
        critics = (
            self.error_summary(target_task, failure, error_knowledge)
            if self.settings.v2_error_summary and failure is not None and error_knowledge
            else None
        )

        user_prompt = self._user_prompt(
            task_information, successful_knowledge, error_knowledge, shown_attempt, critics
        )
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=user_prompt, system_prompt=system_prompt
            )
            >= APIBackend().chat_token_limit
            and (successful_knowledge or error_knowledge)
        ):
            if error_knowledge:
                error_knowledge = error_knowledge[:-1]
            elif successful_knowledge:
                successful_knowledge = successful_knowledge[:-1]
            user_prompt = self._user_prompt(
                task_information, successful_knowledge, error_knowledge, shown_attempt, critics
            )

        response = ""
        for _ in range(max(1, self.settings.llm_retries)):
            response = APIBackend(
                use_chat_cache=self.settings.coder_use_cache
            ).build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            try:
                spec = parse_factor_spec(
                    response,
                    available_fields=fields,
                    max_params=self.settings.max_params,
                    max_cases=self.settings.max_cases,
                )
            except FactorSpecError as error:
                user_prompt = (
                    user_prompt
                    + "\n\nYour previous answer was rejected:\n"
                    + str(error)
                    + "\nReturn the complete JSON object again."
                )
                continue
            return spec.to_files()

        return {"factor.json": response}

    def assign_code_list_to_evo(self, code_list: list[dict | None], evo: EvolvingItem) -> EvolvingItem:
        for index, code in enumerate(code_list):
            if code is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = PhasorFactorWorkspace(
                    target_task=evo.sub_tasks[index]
                )
            if code:
                evo.sub_workspace_list[index].inject_files(**code)
        return evo
