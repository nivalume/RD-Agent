"""The crypto qlib experiments and scenarios.

The experiments are the qlib ones with a :class:`CryptoQlibFBWorkspace` on the crypto templates, so
the qlib runners and feedback take them as they are: the runners pick the SOTA factors and models out
of ``based_experiments`` by ``isinstance(..., QlibFactorExperiment / QlibModelExperiment)``.

The scenarios are this package's own (:class:`rdagent.core.scenario.Scenario`), not subclasses of
the qlib ones. Each one is put together from two sets of prompts:

* the qlib prompts that hold for any market: the factor and model code interfaces, the model output
  format and the rich-style descriptions, and the description of the factor data folder
  (``get_data_folder_intro``);
* the crypto prompts: the market and its calendar (``scenarios.crypto.experiment.prompts``), and the
  evaluation setting, the simulators, the experiment setting and the example of a factor's output
  (``prompts.yaml`` here).

The factor and quant scenarios point the factor coder at the crypto tables before reading them.

Render a scenario without running anything::

    python -m rdagent.scenarios.crypto.qlib.experiment factor     # or model, quant
"""

from __future__ import annotations

import argparse

from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.model_coder.conf import get_model_env
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS
from rdagent.scenarios.crypto.qlib.data import use_crypto_factor_data
from rdagent.scenarios.crypto.qlib.workspace import (
    FACTOR_TEMPLATE,
    MODEL_TEMPLATE,
    CryptoQlibFBWorkspace,
)
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

CRYPTO_PROMPTS = "scenarios.crypto.experiment.prompts"
"""The crypto market and calendar prompts, shared with the proposal-only crypto scenario."""

QLIB_PROMPTS = "scenarios.qlib.experiment.prompts"
"""The qlib scenario prompts; the ones that do not depend on the market are used as they are."""


class CryptoQlibFactorExperiment(QlibFactorExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE)


class CryptoQlibModelExperiment(QlibModelExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = CryptoQlibFBWorkspace(template_folder_path=MODEL_TEMPLATE)


def _qlib(key: str, **context: object) -> str:
    return T(f"{QLIB_PROMPTS}:{key}").r(**context)


def _crypto(key: str, **context: object) -> str:
    return T(f"{CRYPTO_PROMPTS}:{key}").r(**context)


def _own(key: str, **context: object) -> str:
    return T(f".prompts:{key}").r(**context)


def _setting() -> str:
    settings = CRYPTO_QLIB_SETTINGS
    return _own(
        "crypto_qlib_setting",
        market=settings.market,
        benchmark=settings.benchmark,
        topk=settings.topk,
        n_drop=settings.n_drop,
        open_cost=settings.open_cost,
        close_cost=settings.close_cost,
        train_start=settings.train_start,
        train_end=settings.train_end,
        valid_start=settings.valid_start,
        valid_end=settings.valid_end,
        test_start=settings.test_start,
        test_end=settings.test_end,
    )


def _experiment_setting(model: str, factors: str) -> str:
    settings = CRYPTO_QLIB_SETTINGS
    return _own(
        "crypto_qlib_experiment_setting",
        market=settings.market,
        model=model,
        factors=factors,
        train_start=settings.train_start,
        train_end=settings.train_end,
        valid_start=settings.valid_start,
        valid_end=settings.valid_end,
        test_start=settings.test_start,
        test_end=settings.test_end,
    )


def _market() -> str:
    return _crypto("crypto_market")


def _factor_background(runtime_environment: str | None) -> str:
    background = _crypto("crypto_quant_background", market=_market(), runtime_environment=runtime_environment or None)
    return _own(
        "crypto_qlib_factor_background", factor_background=background, calendar=_crypto("crypto_quant_calendar")
    )


def _model_background(runtime_environment: str | None) -> str:
    return _own("crypto_qlib_model_background", market=_market(), runtime_environment=runtime_environment or None)


def _factor_simulator() -> str:
    return _own("crypto_qlib_factor_simulator", setting=_setting())


def _model_simulator() -> str:
    return _own("crypto_qlib_model_simulator", setting=_setting())


def _factor_output_format() -> str:
    return _own("crypto_qlib_factor_output_format")


class CryptoQlibFactorScenario(Scenario):
    """Factors on the crypto qlib data, evaluated with LightGBM and a daily top-k backtest."""

    def __init__(self) -> None:
        super().__init__()
        use_crypto_factor_data()
        self._runtime_environment: str | None = None
        self._background = _factor_background(self.get_runtime_environment())
        self._source_data = get_data_folder_intro()
        self._interface = _qlib("qlib_factor_interface")
        self._output_format = _factor_output_format()
        self._simulator = _factor_simulator()
        self._rich_style_description = _qlib("qlib_factor_rich_style_description")
        self._experiment_setting = _experiment_setting("LGBModel", "Alpha20 base features plus the factors")

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data

    @property
    def interface(self) -> str:
        return self._interface

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
    ) -> str:
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""

    def get_runtime_environment(self) -> str:
        """The packages of the factor environment, read from it once."""
        if self._runtime_environment is None:
            self._runtime_environment = get_runtime_environment_by_env(env=get_factor_env())
        return self._runtime_environment


class CryptoQlibModelScenario(Scenario):
    """Models trained on the base features of the crypto qlib data, with the same backtest."""

    def __init__(self) -> None:
        super().__init__()
        self._runtime_environment: str | None = None
        self._background = _model_background(self.get_runtime_environment())
        self._interface = _qlib("qlib_model_interface")
        self._output_format = _qlib("qlib_model_output_format")
        self._simulator = _model_simulator()
        self._rich_style_description = _qlib("qlib_model_rich_style_description")
        self._experiment_setting = _experiment_setting("The proposed model", "Alpha20 base features")

    @property
    def background(self) -> str:
        return self._background

    @property
    def interface(self) -> str:
        return self._interface

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
    ) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""

    def get_runtime_environment(self) -> str:
        """The packages of the model environment, read from it once."""
        if self._runtime_environment is None:
            self._runtime_environment = get_runtime_environment_by_env(env=get_model_env())
        return self._runtime_environment


MODEL_TAGS = ("model", "model tuning", "Tabular", "TimeSeries")
"""The ``filtered_tag`` values that ask for the model part of the quant scenario. The model coder's
evaluators pass the model type (``Tabular``, ``TimeSeries``)."""

FACTOR_TAGS = ("factor", "feature", "factors")


class CryptoQlibQuantScenario(Scenario):
    """Factors and models together: each round works on one of them, as the action of its hypothesis
    says."""

    def __init__(self) -> None:
        super().__init__()
        use_crypto_factor_data()
        self._runtime_environment: dict[str, str] = {}
        self._source_data = get_data_folder_intro()
        self._rich_style_description = _qlib("qlib_factor_rich_style_description")
        self._experiment_setting = _experiment_setting(
            "LGBModel, or the proposed model", "Alpha20 base features plus the factors"
        )

    def background(self, tag: str | None = None) -> str:
        assert tag in [None, "factor", "model"]
        quant = "The background of the scenario is as follows:\n" + _own(
            "crypto_qlib_quant_background",
            market=_market(),
            runtime_environment=self.get_runtime_environment(),
        )
        factor = (
            "This time, I need your help with the research and development of the factor. The background "
            "of the factor scenario is as follows:\n" + _factor_background(self.get_runtime_environment(tag="factor"))
        )
        model = (
            "This time, I need your help with the research and development of the model. The background "
            "of the model scenario is as follows:\n" + _model_background(self.get_runtime_environment(tag="model"))
        )
        if tag is None:
            return quant + "\n" + factor + "\n" + model
        return factor if tag == "factor" else model

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data

    def output_format(self, tag: str | None = None) -> str:
        assert tag in [None, "factor", "model"]
        factor = "The factor code should output the following format:\n" + _factor_output_format()
        model = "The model code should output the following format:\n" + _qlib("qlib_model_output_format")
        if tag is None:
            return factor + "\n" + model
        return factor if tag == "factor" else model

    def interface(self, tag: str | None = None) -> str:
        assert tag in [None, "factor", "model"]
        factor = "The factor code should be written in the following interface:\n" + _qlib("qlib_factor_interface")
        model = "The model code should be written in the following interface:\n" + _qlib("qlib_model_interface")
        if tag is None:
            return factor + "\n" + model
        return factor if tag == "factor" else model

    def simulator(self, tag: str | None = None) -> str:
        assert tag in [None, "factor", "model"]
        factor = "The factor code will be sent to the simulator:\n" + _factor_simulator()
        model = "The model code will be sent to the simulator:\n" + _model_simulator()
        if tag is None:
            return factor + "\n" + model
        return factor if tag == "factor" else model

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
        def common(tag: str | None = None) -> str:
            return f"""\n------Background of the scenario------
{self.background(tag)}
------The source dataset you can use------
{self.get_source_data_desc()}
"""

        def part(tag: str) -> str:
            return (
                common(tag)
                + f"""
------The interface you should follow to write the runnable code------
{self.interface(tag)}
"""
                + f"""
------The output of your code should be in the format------
{self.output_format(tag)}
"""
                + f"""
------The simulator user can use to test your solution------
{self.simulator(tag)}
"""
            )

        def whole() -> str:
            return common() + f"""
------The simulator user can use to test your solution------
{self.simulator(None)}
"""

        if simple_background:
            return common()
        if filtered_tag in ("hypothesis_and_experiment", "feedback"):
            return whole()
        if filtered_tag in FACTOR_TAGS:
            return part("factor")
        if filtered_tag in MODEL_TAGS:
            return part("model")
        if action in ("factor", "model"):
            return part(action)
        return whole()

    def get_runtime_environment(self, tag: str | None = None) -> str:
        """The packages of the factor environment, of the model environment, or both (``tag`` None);
        each environment is read once."""
        assert tag in [None, "factor", "model"]
        if tag is not None:
            if tag not in self._runtime_environment:
                env = get_factor_env() if tag == "factor" else get_model_env()
                self._runtime_environment[tag] = get_runtime_environment_by_env(env=env)
            return self._runtime_environment[tag]
        return (
            "=== [Environment to generate the factors] ===\n"
            + self.get_runtime_environment(tag="factor").strip()
            + "\n\n=== [Environment to train the models] ===\n"
            + self.get_runtime_environment(tag="model").strip()
        )


# The qlib feedback and the qlib model proposal, which the quant loop reuses, choose their quant
# prompts with `isinstance(scen, QlibQuantScenario)` (scenarios/qlib/developer/feedback.py,
# scenarios/qlib/proposal/model_proposal.py): the model feedback of a quant round would otherwise get
# the factor feedback prompt, whose answer has no "Decision", and no model would ever become SOTA.
# Registering declares that this class offers the quant interface -- background(tag),
# get_scenario_all_desc(action=...) -- without inheriting any of the qlib scenario's code.
QlibQuantScenario.register(CryptoQlibQuantScenario)


SCENARIOS = {
    "factor": CryptoQlibFactorScenario,
    "model": CryptoQlibModelScenario,
    "quant": CryptoQlibQuantScenario,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rdagent.scenarios.crypto.qlib.experiment",
        description="Print the whole prompt of a crypto qlib scenario.",
    )
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    args = parser.parse_args(argv)
    scenario = SCENARIOS[args.scenario]()
    print(scenario.get_scenario_all_desc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
