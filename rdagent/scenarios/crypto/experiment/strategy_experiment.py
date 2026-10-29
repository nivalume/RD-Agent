"""The crypto strategy: a tradeable rule on one instrument's own time series.

A crypto strategy is not a factor. A factor is a characteristic whose value is ranked against the other
instruments at the same time; a strategy here reads one instrument's own bars and funding, decides a
position, and produces one PnL path for that instrument. The same rule with the same parameters runs on
every instrument and the paths are pooled, so a strategy is judged on the conditional distribution of its
own returns rather than on a cross-sectional IC.

That is the point of trading strategies here: the crypto cross-section is dominated by one common market
factor (the instruments mostly rise and fall together), and what is left of the dispersion is largely
microstructure -- thin books, stop runs, manipulation -- rather than information. A single factor's
cross-sectional IC is a weak and unstable statistic on such a cross-section. A mechanism that is a
property of the market (forced liquidations, crowded leverage, thin liquidity) is not, and it is
evaluable on every instrument instead of a handful of them.

:class:`StrategyTask` is the specification the proposal stage produces and a backtest will later consume.
It carries the economic mechanism next to the LaTeX rule -- who pays, why the transfer persists, what
risk is borne, and what the data should show -- so the thesis and the implementable rule cannot drift
apart between generations.

:class:`CryptoStrategyScenario` describes the same market to a strategy proposal: it subclasses the factor
scenario so that the calendar, the data schemas and the runtime-environment handling stay in one place,
and replaces only the background, because what the model is asked to build is different.
"""

from __future__ import annotations

import json
from pathlib import Path

from rdagent.core.experiment import Experiment, FBWorkspace, Task
from rdagent.scenarios.crypto.experiment.factor_experiment import CryptoFactorScenario

PHENOMENA = ("mean_reversion", "trend_following")
"""The two phenomena a strategy may trade. Everything else is out of scope for the crypto scenario."""

SPECIFICATION_FIELDS = (
    "description",
    "phenomenon",
    "mechanism",
    "return_source",
    "risk_borne",
    "signal",
    "position",
    "horizon",
    "gate",
    "testable_prediction",
    "fails_when",
    "parameters",
)
"""The keys a proposal may set on a strategy, in the order the prompt asks for them."""


def normalise_phenomenon(value: str) -> str:
    """``Mean-Reversion`` and ``mean reversion`` both name ``mean_reversion``."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


class StrategyTask(Task):
    """One tradeable rule: the mechanism, the LaTeX rule, and the prediction that falsifies it.

    The parameters are dimensionless by construction: a strategy is instrument-agnostic, so a threshold
    is a z-score or a multiple of realised volatility, never a price level, and a holding period is
    counted in bars, never in dates. Both are enforced by the prompt and checked here only where a
    violation would make the rule unrunnable.
    """

    def __init__(
        self,
        strategy_name: str,
        phenomenon: str = "",
        description: str = "",
        mechanism: str = "",
        return_source: str = "",
        risk_borne: str = "",
        signal: str = "",
        position: str = "",
        horizon: str = "",
        gate: str = "",
        testable_prediction: str = "",
        fails_when: str = "",
        parameters: dict[str, str] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        if not strategy_name:
            raise ValueError("a strategy needs a name")
        phenomenon = normalise_phenomenon(phenomenon)
        if phenomenon not in PHENOMENA:
            raise ValueError(
                f"{phenomenon!r} is not a phenomenon this scenario trades; choose one of {', '.join(PHENOMENA)}",
            )
        for name, value in (("signal", signal), ("position", position)):
            if not str(value).strip():
                raise ValueError(f"the strategy {strategy_name!r} has no {name}")
        super().__init__(strategy_name, *args, description=description, **kwargs)
        self.strategy_name = strategy_name
        self.phenomenon = phenomenon
        self.mechanism = mechanism
        self.return_source = return_source
        self.risk_borne = risk_borne
        self.signal = signal
        self.position = position
        self.horizon = horizon
        self.gate = gate
        self.testable_prediction = testable_prediction
        self.fails_when = fails_when
        self.parameters: dict[str, str] = parameters if isinstance(parameters, dict) else {}

    def _fields(self) -> list[tuple[str, str]]:
        """The specification as name/value pairs, in the order the prompt asks for them.

        Empty fields are dropped rather than printed as blanks: the trace shows every past trial, so a
        field the model left out must not cost a line in every later prompt.
        """
        values = {
            "strategy_name": self.strategy_name,
            "description": self.description,
            "phenomenon": self.phenomenon,
            "mechanism": self.mechanism,
            "return_source": self.return_source,
            "risk_borne": self.risk_borne,
            "signal": self.signal,
            "position": self.position,
            "horizon": self.horizon,
            "gate": self.gate,
            "testable_prediction": self.testable_prediction,
            "fails_when": self.fails_when,
            "parameters": json.dumps(self.parameters, ensure_ascii=False) if self.parameters else "",
        }
        return [(name, str(value)) for name, value in values.items() if str(value).strip()]

    def get_task_information(self) -> str:
        return "\n".join([f"{name}: {value}" for name, value in self._fields()])

    def get_task_brief_information(self) -> str:
        """What the next round's prompt shows of this strategy: the mechanism and the rule, no filler."""
        return self.get_task_information()


class CryptoStrategyExperiment(Experiment[StrategyTask, FBWorkspace, FBWorkspace]):
    """A group of strategy variants that share one hypothesis and one evaluation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # The workspace of the whole experiment, where the rules will be backtested. The crypto
        # evaluation is not implemented yet, so it is the bare workspace.
        self.experiment_workspace = FBWorkspace()
        self.stdout = ""


class CryptoStrategyScenario(CryptoFactorScenario):
    """The crypto market, described to a researcher who is asked for a strategy rather than a factor.

    The market, the calendar and the data schemas are the factor scenario's. Only the background differs --
    it names a strategy as the deliverable instead of a factor -- so the data-schema rendering and the
    runtime-environment handling are inherited rather than copied.
    """

    def __init__(self, with_runtime_environment: bool = True, data_root_dir: Path | None = None) -> None:
        super().__init__(with_runtime_environment=with_runtime_environment, data_root_dir=data_root_dir)
        self._strategy_background = self._render(
            "crypto_strategy_background",
            runtime_environment=self.get_runtime_environment() or None,
        )

    @property
    def background(self) -> str:
        """Overrides the factor background; ``get_scenario_all_desc`` picks it up through this property."""
        return self._strategy_background
