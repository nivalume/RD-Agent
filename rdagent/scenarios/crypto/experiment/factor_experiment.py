"""The crypto scenario: factor research on centralized-exchange perpetual futures.

This mirrors :mod:`rdagent.scenarios.qlib.experiment.factor_experiment`: the scenario renders the
prompts in `prompts.yaml` next to this module, and `CryptoFactorExperiment` is what a hypothesis is
converted into. The market and its calendar follow the jarvis-research crypto scenario
(`alpha/scenarios/crypto`); the data the prompt describes comes from the schema files under
`<data root>/schemas` (see :mod:`rdagent.scenarios.crypto.experiment.data_schema`).

Render the prompt without a configured environment::

    python -m rdagent.scenarios.crypto.experiment.factor_experiment
    python -m rdagent.scenarios.crypto.experiment.factor_experiment background calendar
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.crypto.conf import CryptoSettings
from rdagent.scenarios.crypto.experiment.data_schema import render_data_schemas
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

NO_DATA_SCHEMA = "No data schema is available; the data root may not be configured."

SECTIONS = (
    "background",
    "calendar",
    "data",
    "interface",
    "strategy",
    "output_format",
    "simulator",
    "experiment_setting",
    "rich_style_description",
)


class CryptoFactorExperiment(FactorExperiment[FactorTask, FBWorkspace, FactorFBWorkspace]):
    """A group of factors that share one hypothesis and one evaluation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The workspace of the whole experiment, where it is evaluated against the bars. The crypto
        # evaluation is not implemented yet, so it is the bare workspace; a factor's own code lives in
        # the per-task FactorFBWorkspace instead.
        self.experiment_workspace = FBWorkspace()
        self.stdout = ""
        self.base_features: dict[str, str] = {}
        self.base_feature_codes: dict[str, str] = {}


class CryptoFactorScenario(Scenario):
    """Crypto perpetual futures: 24/7 bars, no fundamentals, and funding that anchors the perpetual."""

    HOURS_PER_DAY = 24

    def __init__(self, with_runtime_environment: bool = True, data_root_dir: Path | None = None) -> None:
        super().__init__()
        self.with_runtime_environment = with_runtime_environment
        self.data_root_dir = Path(data_root_dir) if data_root_dir is not None else CryptoSettings().data_root_dir
        self._runtime_environment: str | None = None
        self._background = self._render(
            "crypto_quant_background",
            runtime_environment=self.get_runtime_environment() or None,
        )
        self._calendar = self._render("crypto_quant_calendar")
        self._source_data = self._render(
            "crypto_factor_data",
            data_schemas=render_data_schemas(self.data_root_dir) or NO_DATA_SCHEMA,
        )
        self._interface = self._render("crypto_factor_interface")
        self._strategy = self._render("crypto_factor_strategy")
        self._output_format = self._render("crypto_factor_output_format")
        self._simulator = self._render("crypto_factor_simulator")
        self._experiment_setting = self._render("crypto_factor_experiment_setting")
        self._rich_style_description = self._render("crypto_factor_rich_style_description")

    @staticmethod
    def _render(key: str, **context: object) -> str:
        """Render one template of this scenario's `prompts.yaml`."""
        return T(f".prompts:{key}").r(**context)

    @property
    def background(self) -> str:
        return self._background

    @property
    def calendar(self) -> str:
        """How the market's clock runs and which calendar features a factor may use."""
        return self._calendar

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def strategy(self) -> str:
        """How the factor code should process the bars."""
        return self._strategy

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
        action: str | None = None,
    ) -> str:
        """A static scenario describer: the market, its calendar and the data schemas.

        The interface, strategy, output format and simulator prompts stay reachable through their own
        properties but are left out of the composed prompt for now.
        """
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}

The calendar of the market:
{self.calendar}

The source data you can use:
{self.get_source_data_desc(task)}
"""

    def get_runtime_environment(self) -> str:
        """The packages of the factor environment; empty when the scenario is rendered without it."""
        if not self.with_runtime_environment:
            return ""
        if self._runtime_environment is None:
            self._runtime_environment = get_runtime_environment_by_env(env=get_factor_env())
        return self._runtime_environment


def main(argv: list[str] | None = None) -> int:
    """Print the scenario prompt, or the named sections of it."""
    parser = argparse.ArgumentParser(
        prog="python -m rdagent.scenarios.crypto.experiment.factor_experiment",
        description=(
            "Render the crypto factor prompts. Without a section the whole scenario prompt is printed."
        ),
    )
    parser.add_argument(
        "sections",
        nargs="*",
        metavar="SECTION",
        help=f"one of {', '.join(SECTIONS)}; all of them when none is named",
    )
    parser.add_argument(
        "--runtime-environment",
        action="store_true",
        help=(
            "execute the factor environment to list its packages; off by default so the prompt renders "
            "without a configured environment"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="the data root whose schemas are rendered; the CRYPTO_DATA_ROOT_DIR setting when omitted",
    )
    parser.add_argument("--list", action="store_true", help="list the sections and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(SECTIONS))
        return 0

    unknown = [name for name in args.sections if name not in SECTIONS]
    if unknown:
        parser.error(f"unknown section {unknown[0]!r}; choose from {', '.join(SECTIONS)}")

    scenario = CryptoFactorScenario(
        with_runtime_environment=args.runtime_environment,
        data_root_dir=args.data_root,
    )
    if not args.sections:
        print(scenario.get_scenario_all_desc())
        return 0

    for name in args.sections:
        text = scenario.get_source_data_desc() if name == "data" else getattr(scenario, name)
        print(f"===== {name} =====")
        print(text)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
