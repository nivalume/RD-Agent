"""The strategy scenario with its evaluation: the panel the strategies run on and how they are judged."""

from __future__ import annotations

import json
from pathlib import Path

from rdagent.core.experiment import Task
from rdagent.scenarios.crypto.experiment.strategy_experiment import CryptoStrategyScenario
from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.utils.agent.tpl import T


def panel_meta(panel: Path) -> dict:
    """The ``meta.json`` jarvis-data wrote next to the panel; empty when there is none yet."""
    try:
        return json.loads((panel / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


class CryptoStrategyBacktestScenario(CryptoStrategyScenario):
    """The strategy proposal's market, with the panel as the data and the backtest as the evaluation.

    The proposal's background is kept; the factor data schemas are replaced by the panel, and the
    evaluation setting (delay, volatility targeting, costs, periods, gates) is added, so that the
    proposal, the coder and the feedback all describe the same test.
    """

    def __init__(
        self,
        with_runtime_environment: bool = False,
        data_root_dir: Path | None = None,
        settings: CryptoStrategySettings | None = None,
    ) -> None:
        super().__init__(with_runtime_environment=with_runtime_environment, data_root_dir=data_root_dir)
        self.settings = settings or CryptoStrategySettings()
        meta = panel_meta(self.settings.panel_path)
        pool = str(meta.get("pool", "top100")).removeprefix("top")
        self._panel_data = T(".prompts:strategy_panel_data").r(
            interval=meta.get("interval", "4h"),
            first=str(meta.get("first", ""))[:10],
            last=str(meta.get("last", ""))[:10],
            instruments=meta.get("instruments"),
            pool=pool,
        )
        self._evaluation = T(".prompts:strategy_evaluation_setting").r(
            settings=self.settings, hide_lockbox=self.settings.hide_lockbox
        )

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._panel_data

    @property
    def experiment_setting(self) -> str:
        return self._evaluation

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
        action: str | None = None,
    ) -> str:
        if simple_background:
            return f"Background of the scenario:\n{self.background}"
        return (
            f"Background of the scenario:\n{self.background}\n\n"
            f"The data:\n{self.get_source_data_desc(task)}\n\n"
            f"{self.experiment_setting}\n"
        )
