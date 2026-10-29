"""``python -m rdagent.components.coder.phasor_coder`` renders the prompt and calls the model once."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Print the implementation prompt, call the model once, and print the factor."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m rdagent.components.coder.phasor_coder",
        description="Render the phasor factor prompt, call the model, and print the factor it writes.",
    )
    parser.add_argument("--print-only", action="store_true", help="print the prompts and do not call the model")
    parser.add_argument("--name", default="close_momentum")
    parser.add_argument(
        "--description",
        default="Cross-sectional rank of the return over the last 20 bars.",
    )
    parser.add_argument("--formulation", default="CSRank(Mean($close, 20) / $close - 1)")
    parser.add_argument("--frequency", default=None, help="bar size, such as 1h; omitted when the scenario fixes it")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=None,
        help="panel columns the factor may read; the coder default when omitted",
    )
    parser.add_argument(
        "--runtime-environment",
        action="store_true",
        help="list the factor environment's packages in the scenario prompt",
    )
    args = parser.parse_args(argv)

    from rdagent.components.coder.phasor_coder.config import PHASOR_COSTEER_SETTINGS
    from rdagent.components.coder.phasor_coder.evolving_strategy import PhasorMultiProcessEvolvingStrategy
    from rdagent.components.coder.phasor_coder.spec import FactorSpecError, parse_factor_spec
    from rdagent.components.coder.phasor_coder.task import PhasorFactorTask
    from rdagent.oai.llm_conf import LLM_SETTINGS
    from rdagent.oai.llm_utils import APIBackend
    from rdagent.scenarios.crypto.experiment.factor_experiment import CryptoFactorScenario
    from rdagent.scenarios.crypto.proposal.factor_proposal import _load_dotenv_and_refresh_settings

    _load_dotenv_and_refresh_settings()

    task = PhasorFactorTask(
        factor_name=args.name,
        factor_description=args.description,
        factor_formulation=args.formulation,
        frequency=args.frequency,
        fields=args.fields,
    )
    scenario = CryptoFactorScenario(with_runtime_environment=args.runtime_environment)
    strategy = PhasorMultiProcessEvolvingStrategy(scenario, PHASOR_COSTEER_SETTINGS)
    system_prompt = strategy._system_prompt(task)
    user_prompt = strategy._user_prompt(task.get_task_information(), [], [], None, None)

    print(f"model: {LLM_SETTINGS.chat_model}")
    print(f"base: {LLM_SETTINGS.openai_api_base or '<the provider default>'}")
    print(f"api key: {'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING (export OPENAI_API_KEY)'}")
    print("\n===== system prompt =====")
    print(system_prompt)
    print("\n===== user prompt =====")
    print(user_prompt)
    if args.print_only:
        return 0

    try:
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
    except Exception as error:  # the CLI reports the failure instead of a traceback
        print(f"\n===== model call failed =====\n{type(error).__name__}: {error}")
        return 1

    print("\n===== model response =====")
    print(response)
    try:
        spec = parse_factor_spec(
            response,
            available_fields=strategy._available_fields(task),
            max_params=PHASOR_COSTEER_SETTINGS.max_params,
            max_cases=PHASOR_COSTEER_SETTINGS.max_cases,
        )
    except FactorSpecError as error:
        print(f"\n===== spec rejected =====\n{error}")
        return 1

    files = spec.to_files()
    print("\n===== factor.json =====")
    print(files["factor.json"].rstrip("\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
