"""The crypto qlib experiments and scenarios.

The experiments are the qlib ones with a :class:`CryptoQlibFBWorkspace` on the crypto templates, so
the upstream runners and feedback take them as they are. The scenarios are the qlib ones with the
crypto market, calendar, evaluation setting and data in their prompts; their constructors point the
factor coder at the crypto tables before the qlib constructor reads them.

Render a scenario without running anything::

    python -m rdagent.scenarios.crypto.qlib.experiment factor     # or model, quant
"""

from __future__ import annotations

import argparse

from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS
from rdagent.scenarios.crypto.qlib.data import use_crypto_factor_data
from rdagent.scenarios.crypto.qlib.workspace import (
    FACTOR_TEMPLATE,
    MODEL_TEMPLATE,
    CryptoQlibFBWorkspace,
)
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment,
    QlibFactorScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils.agent.tpl import T

CRYPTO_PROMPTS = "scenarios.crypto.experiment.prompts"
"""The crypto market and calendar prompts, shared with the proposal-only crypto scenario."""


class CryptoQlibFactorExperiment(QlibFactorExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE)


class CryptoQlibModelExperiment(QlibModelExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = CryptoQlibFBWorkspace(template_folder_path=MODEL_TEMPLATE)


def _setting() -> str:
    settings = CRYPTO_QLIB_SETTINGS
    return T(".prompts:crypto_qlib_setting").r(
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
    return T(".prompts:crypto_qlib_experiment_setting").r(
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
    return T(f"{CRYPTO_PROMPTS}:crypto_market").r()


def _factor_background(runtime_environment: str | None) -> str:
    background = T(f"{CRYPTO_PROMPTS}:crypto_quant_background").r(
        market=_market(), runtime_environment=runtime_environment or None
    )
    return T(".prompts:crypto_qlib_factor_background").r(
        factor_background=background, calendar=T(f"{CRYPTO_PROMPTS}:crypto_quant_calendar").r()
    )


def _model_background(runtime_environment: str | None) -> str:
    return T(".prompts:crypto_qlib_model_background").r(
        market=_market(), runtime_environment=runtime_environment or None
    )


class CryptoQlibFactorScenario(QlibFactorScenario):
    def __init__(self) -> None:
        use_crypto_factor_data()
        super().__init__()
        self._background = _factor_background(self.get_runtime_environment())
        self._simulator = T(".prompts:crypto_qlib_factor_simulator").r(setting=_setting())
        self._experiment_setting = _experiment_setting("LGBModel", "Alpha20 base features plus the factors")


class CryptoQlibModelScenario(QlibModelScenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = _model_background(self.get_runtime_environment())
        self._simulator = T(".prompts:crypto_qlib_model_simulator").r(setting=_setting())
        self._experiment_setting = _experiment_setting("The proposed model", "Alpha20 base features")


class CryptoQlibQuantScenario(QlibQuantScenario):
    def __init__(self) -> None:
        use_crypto_factor_data()
        super().__init__()
        self._experiment_setting = _experiment_setting(
            "LGBModel, or the proposed model", "Alpha20 base features plus the factors"
        )

    def background(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        quant = "The background of the scenario is as follows:\n" + T(".prompts:crypto_qlib_quant_background").r(
            market=_market(), runtime_environment=self.get_runtime_environment()
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

    def simulator(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor = "The factor code will be sent to the simulator:\n" + T(".prompts:crypto_qlib_factor_simulator").r(
            setting=_setting()
        )
        model = "The model code will be sent to the simulator:\n" + T(".prompts:crypto_qlib_model_simulator").r(
            setting=_setting()
        )
        if tag is None:
            return factor + "\n" + model
        return factor if tag == "factor" else model


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
