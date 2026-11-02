"""The crypto qlib scenario: settings, workflow configs, workspace, converters and scenarios."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from jinja2 import Template

from rdagent.app.crypto_rd_loop.qlib_conf import (
    CryptoQlibFactorPropSetting,
    CryptoQlibModelPropSetting,
    CryptoQlibQuantPropSetting,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.core.scenario import Scenario
from rdagent.scenarios.crypto.qlib import conf as crypto_conf
from rdagent.scenarios.crypto.qlib import experiment as crypto_experiment
from rdagent.scenarios.crypto.qlib import workspace as crypto_workspace
from rdagent.scenarios.crypto.qlib.conf import CryptoQlibSettings
from rdagent.scenarios.crypto.qlib.experiment import (
    CryptoQlibFactorExperiment,
    CryptoQlibFactorScenario,
    CryptoQlibModelExperiment,
    CryptoQlibModelScenario,
    CryptoQlibQuantScenario,
)
from rdagent.scenarios.crypto.qlib.proposal import (
    CryptoQlibFactorHypothesis2Experiment,
    CryptoQlibModelHypothesis2Experiment,
)
from rdagent.scenarios.crypto.qlib.workspace import (
    FACTOR_TEMPLATE,
    MODEL_TEMPLATE,
    CryptoDataMissing,
    CryptoQlibFBWorkspace,
    CryptoQTDockerEnv,
    require_data,
)
from rdagent.scenarios.qlib.developer import feedback as qlib_feedback
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.qlib.proposal.model_proposal import (
    QlibModelHypothesis2Experiment,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import QlibCondaEnv

CONFIGS = sorted([*FACTOR_TEMPLATE.glob("*.yaml"), *MODEL_TEMPLATE.glob("*.yaml")])


def render(path: Path, env: dict[str, str]) -> dict:
    """A workflow config as qrun renders it: jinja with the run's environment."""
    return yaml.safe_load(Template(path.read_text()).render(**env))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_the_workflow_configs_are_the_crypto_ones(path):
    settings = CryptoQlibSettings(market="top50", topk=10, n_drop=2, open_cost=0.0004, test_end="2025-01-01")
    env = {
        **settings.run_env(),
        "feature_names": "['KLEN']",
        "feature_expressions": "['($high-$low)/$open']",
        "dataset_cls": "DatasetH",
        "num_features": "21",
    }
    config = render(path, env)
    assert config["qlib_init"] == {"provider_uri": "~/.qlib/qlib_data/crypto_data", "region": "us"}
    assert config["market"] == "top50" and config["benchmark"] == "BTCUSDT"
    backtest = config["port_analysis_config"]
    assert backtest["strategy"]["kwargs"]["topk"] == 10 and backtest["strategy"]["kwargs"]["n_drop"] == 2
    exchange = backtest["backtest"]["exchange_kwargs"]
    assert exchange["limit_threshold"] is None and exchange["min_cost"] == 0
    assert exchange["open_cost"] == 0.0004 and exchange["close_cost"] == 0.0005
    assert str(backtest["backtest"]["end_time"]) == "2025-01-01"  # YAML reads a date; so does qlib
    segments = config["task"]["dataset"]["kwargs"]["segments"]
    assert [str(day) for day in segments["train"]] == ["2021-01-01", "2023-06-30"]
    assert [str(day) for day in segments["test"]] == ["2024-07-01", "2025-01-01"]
    analysis = [record for record in config["task"]["record"] if record["class"] == "SigAnaRecord"]
    assert analysis[0]["kwargs"]["ann_scaler"] == 365
    text = path.read_text()
    assert "cn_data" not in text and "csi300" not in text and "SH000300" not in text


def test_the_configs_differ_from_qlib_only_in_the_crypto_settings():
    qlib = Path(crypto_workspace.__file__).parents[2] / "qlib" / "experiment"
    for path in CONFIGS:
        upstream = (qlib / path.parent.name / path.name).read_text().splitlines()
        ours = path.read_text().splitlines()
        assert len(upstream) == len(ours), path
        changed = [line for line, theirs in zip(ours, upstream) if line != theirs]
        assert all(
            any(
                word in line
                for word in ("crypto_", "region: us", "2021-", "2023-", "2024-", "null", "min_cost: 0", "365")
            )
            for line in changed
        ), changed


def test_the_run_env_holds_every_setting():
    env = CryptoQlibSettings().run_env()
    assert env["crypto_provider_uri"] == "~/.qlib/qlib_data/crypto_data" and env["crypto_market"] == "top100"
    assert "test_end" not in env  # up to the last day of the data
    assert CryptoQlibSettings(test_end="2025-01-01").run_env()["test_end"] == "2025-01-01"


def write_data(home: Path, pool: str = "top100") -> Path:
    directory = home / ".qlib" / "qlib_data" / "crypto_data"
    (directory / "calendars").mkdir(parents=True)
    (directory / "instruments").mkdir()
    (directory / "calendars" / "day.txt").write_text("2023-12-31\n2024-01-01\n")
    (directory / "instruments" / f"{pool}.txt").write_text("BTCUSDT\t2024-01-01\t2024-01-01\n")
    return directory


def test_the_data_must_be_written_first(tmp_path):
    settings = CryptoQlibSettings(provider_uri=str(tmp_path / ".qlib" / "qlib_data" / "crypto_data"))
    with pytest.raises(CryptoDataMissing, match="jarvis-history qlib --out .* --top 100"):
        require_data(settings)
    directory = write_data(tmp_path)
    assert require_data(settings) == directory
    with pytest.raises(CryptoDataMissing, match="top50"):
        require_data(CryptoQlibSettings(provider_uri=settings.provider_uri, market="top50"))
    elsewhere = CryptoQlibSettings(provider_uri=str(tmp_path / "data"))
    with pytest.raises(CryptoDataMissing, match="must be under ~/.qlib"):
        require_data(elsewhere, docker=True)  # conda reads any directory; docker mounts ~/.qlib


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_data(tmp_path)
    monkeypatch.setattr(crypto_conf, "CRYPTO_QLIB_SETTINGS", CryptoQlibSettings())
    monkeypatch.setattr(crypto_workspace, "CRYPTO_QLIB_SETTINGS", crypto_conf.CRYPTO_QLIB_SETTINGS)
    return tmp_path


@pytest.mark.parametrize("env_type,expected", [("docker", CryptoQTDockerEnv), ("conda", QlibCondaEnv)])
def test_both_environments_are_offered(env_type, expected, data_home, monkeypatch):
    monkeypatch.setattr(MODEL_COSTEER_SETTINGS, "env_type", env_type)
    assert isinstance(CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE).qlib_env(), expected)
    monkeypatch.setattr(MODEL_COSTEER_SETTINGS, "env_type", "nothing")
    assert CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE).qlib_env() is None


def test_every_run_gets_the_crypto_settings(data_home, monkeypatch):
    seen = {}

    def execute(self, qlib_config_name="conf.yaml", run_env={}, *args, **kwargs):
        seen.update(run_env)
        return None, ""

    monkeypatch.setattr(QlibFBWorkspace, "execute", execute)
    workspace = CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE)
    # The upstream runner passes the CN segments; the crypto ones replace them.
    runner_env = {"train_start": "2008-01-01", "test_end": "2020-08-01", "feature_names": "[]"}
    workspace.execute("conf_baseline.yaml", runner_env)
    assert seen["train_start"] == "2021-01-01" and seen["feature_names"] == "[]"
    assert seen["crypto_market"] == "top100"
    # No test end set: the backtest ends the day before the last of the calendar, never the
    # runner's CN date.
    assert seen["test_end"] == "2023-12-31"


def test_the_experiments_run_the_crypto_templates():
    factor = CryptoQlibFactorExperiment(sub_tasks=[])
    model = CryptoQlibModelExperiment(sub_tasks=[])
    assert isinstance(factor.experiment_workspace, CryptoQlibFBWorkspace)
    assert "conf_combined_factors.yaml" in factor.experiment_workspace.file_dict
    assert "conf_sota_factors_model.yaml" in model.experiment_workspace.file_dict
    assert "crypto_market" in factor.experiment_workspace.file_dict["conf_baseline.yaml"]


def test_hypotheses_become_crypto_qlib_experiments():
    hypothesis = Hypothesis("Momentum ranks", "", "", "", "", "")
    trace = Trace(scen=None)
    factors = CryptoQlibFactorHypothesis2Experiment().convert_response(
        json.dumps({"MOM20": {"description": "20-day return", "formulation": "c/c20-1", "variables": {}}}),
        hypothesis,
        trace,
    )
    assert isinstance(factors, CryptoQlibFactorExperiment)
    assert [task.factor_name for task in factors.sub_tasks] == ["MOM20"]
    assert isinstance(factors.based_experiments[0], CryptoQlibFactorExperiment)  # the baseline
    model_answer = {
        "TinyMLP": {
            "description": "an MLP",
            "formulation": "y = W2 relu(W1 x)",
            "architecture": "two layers",
            "variables": {},
            "hyperparameters": {"hidden": 32},
            "training_hyperparameters": {"n_epochs": 10},
            "model_type": "Tabular",
        }
    }
    model = CryptoQlibModelHypothesis2Experiment().convert_response(json.dumps(model_answer), hypothesis, trace)
    assert isinstance(model, CryptoQlibModelExperiment) and model.sub_tasks[0].name == "TinyMLP"


@pytest.fixture
def factor_tables(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["BTCUSDT", "ETHUSDT"]], names=["datetime", "instrument"]
    )
    frame = pd.DataFrame({"$close": range(6), "$funding_rate": 0.0001}, index=index, dtype="float64")
    folders = {}
    for name in ("all", "debug"):
        folder = tmp_path / name
        folder.mkdir()
        frame.to_hdf(folder / "daily_pv.h5", key="data")
        folders[name] = str(folder)
    settings = CryptoQlibSettings(factor_data_folder=folders["all"], factor_data_folder_debug=folders["debug"])
    monkeypatch.setattr(crypto_conf, "CRYPTO_QLIB_SETTINGS", settings)
    import rdagent.scenarios.crypto.qlib.data as crypto_data

    monkeypatch.setattr(crypto_data.use_crypto_factor_data, "__defaults__", (settings,))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", FACTOR_COSTEER_SETTINGS.data_folder)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder_debug", FACTOR_COSTEER_SETTINGS.data_folder_debug)
    for cls in (CryptoQlibFactorScenario, CryptoQlibModelScenario, CryptoQlibQuantScenario):
        monkeypatch.setattr(cls, "get_runtime_environment", lambda self, *args, **kwargs: "python 3.10")
    return settings


def test_the_factor_scenario_describes_the_crypto_setting(factor_tables):
    scenario = CryptoQlibFactorScenario()
    assert FACTOR_COSTEER_SETTINGS.data_folder == factor_tables.factor_data_folder
    assert "linear perpetual futures margined in a stablecoin" in scenario.background
    assert "Perpetual funding settles at 00:00, 08:00 and 16:00 UTC" in scenario.background
    assert "$funding_rate" in scenario.get_source_data_desc()
    assert "the `top100` pool" in scenario.simulator and "hold the 20 instruments" in scenario.simulator
    assert "funding payments are not in them" in scenario.simulator
    assert "Binance USD-M USDT perpetuals, top100" in scenario.experiment_setting
    assert "CSI300" not in scenario.experiment_setting
    whole = scenario.get_scenario_all_desc()
    assert "linear perpetual futures" in whole and "top100" in whole


def test_the_model_and_quant_scenarios(factor_tables):
    model = CryptoQlibModelScenario()
    assert "develop new models" in model.background and "linear perpetual futures" in model.background
    assert "Train the model defined in your class Net" in model.simulator
    quant = CryptoQlibQuantScenario()
    assert "develop new factors and models" in quant.background()
    assert "Perpetual funding settles" in quant.background("factor")
    assert "develop new models" in quant.background("model")
    assert "the `top100` pool" in quant.simulator("model")
    assert "Binance USD-M USDT perpetuals" in quant.experiment_setting


def test_the_scenarios_are_crypto_ones_on_the_qlib_templates(factor_tables):
    for cls, qlib_cls in (
        (CryptoQlibFactorScenario, QlibFactorScenario),
        (CryptoQlibModelScenario, QlibModelScenario),
        (CryptoQlibQuantScenario, QlibQuantScenario),
    ):
        assert issubclass(cls, Scenario) and qlib_cls not in cls.__mro__
    qlib = "scenarios.qlib.experiment.prompts"
    factor, model, quant = CryptoQlibFactorScenario(), CryptoQlibModelScenario(), CryptoQlibQuantScenario()
    # The market-free templates are the qlib ones ...
    assert factor.interface == T(f"{qlib}:qlib_factor_interface").r()
    assert model.interface == T(f"{qlib}:qlib_model_interface").r()
    assert model.output_format == T(f"{qlib}:qlib_model_output_format").r()
    assert quant.interface("model").endswith(T(f"{qlib}:qlib_model_interface").r())
    # ... the example of a factor's output is the crypto one.
    assert "BTCUSDT" in factor.output_format and "SH600000" not in factor.output_format
    assert "BTCUSDT" in quant.output_format("factor")
    assert "result.h5" in factor.get_scenario_all_desc()


def test_the_quant_scenario_answers_every_tag(factor_tables):
    quant = CryptoQlibQuantScenario()
    # The model coder's evaluators ask with the model type.
    for tag in ("TimeSeries", "Tabular", "model"):
        desc = quant.get_scenario_all_desc(filtered_tag=tag)
        assert "develop new models" in desc and "class Net" in desc and "result.h5" not in desc
    assert "result.h5" in quant.get_scenario_all_desc(filtered_tag="feature")
    assert "develop new models" in quant.get_scenario_all_desc(action="model")
    whole = quant.get_scenario_all_desc()
    assert "develop new factors and models" in whole and "class Net" in whole
    assert quant.get_scenario_all_desc(filtered_tag="feedback") == whole


def test_the_runtime_environment_is_read_once(monkeypatch):
    import rdagent.scenarios.crypto.qlib.experiment as module

    calls = []
    monkeypatch.setattr(module, "get_runtime_environment_by_env", lambda env: calls.append(env) or "python 3.10")
    monkeypatch.setattr(module, "get_factor_env", lambda: "factor env")
    monkeypatch.setattr(module, "get_model_env", lambda: "model env")
    quant = CryptoQlibQuantScenario.__new__(CryptoQlibQuantScenario)
    quant._runtime_environment = {}
    for _ in range(3):
        quant.get_runtime_environment()
        quant.get_runtime_environment(tag="model")
    assert calls == ["factor env", "model env"]


def test_the_qlib_feedback_and_proposal_treat_it_as_a_quant_scenario(factor_tables, monkeypatch):
    quant = CryptoQlibQuantScenario()
    assert isinstance(quant, QlibQuantScenario)  # registered, not inherited
    assert not isinstance(CryptoQlibFactorScenario(), QlibQuantScenario)
    hypothesis = Hypothesis("A smaller GRU", "", "", "", "", "")
    trace = Trace(scen=quant)
    context, _ = QlibModelHypothesis2Experiment().prepare_context(hypothesis, trace)
    assert context["scenario"] == quant.get_scenario_all_desc(action="model")

    prompts = []

    class Backend:
        def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, **kwargs):
            prompts.append(system_prompt)
            answer = {"Observations": "", "Feedback for Hypothesis": "", "New Hypothesis": "", "Reasoning": ""}
            return json.dumps({**answer, "Decision": True})

    monkeypatch.setattr(qlib_feedback, "APIBackend", Backend)
    task = ModelTask(
        name="GRU",
        description="a GRU",
        formulation="h_t = GRU(x_t, h_{t-1})",
        architecture="one layer",
        variables={},
        hyperparameters={},
        training_hyperparameters={},
        model_type="TimeSeries",
    )
    experiment = CryptoQlibModelExperiment(sub_tasks=[task], hypothesis=hypothesis)
    experiment.result = pd.Series(0.1, index=qlib_feedback.IMPORTANT_METRICS)
    experiment.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "model_cls = None"})]
    feedback = qlib_feedback.QlibModelExperiment2Feedback(quant).generate_feedback(experiment, trace)
    # The model feedback prompt, whose answer carries the "Decision" the feedback reads.
    assert feedback.decision is True
    assert all("quantitative analysis assistant" in prompt and "develop new models" in prompt for prompt in prompts)


def test_the_loops_wire_the_crypto_classes():
    factor, model, quant = CryptoQlibFactorPropSetting(), CryptoQlibModelPropSetting(), CryptoQlibQuantPropSetting()
    assert factor.scen.endswith("CryptoQlibFactorScenario")
    assert factor.hypothesis_gen.endswith("crypto.proposal.factor_proposal.CryptoFactorHypothesisGen")
    assert factor.runner == "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    assert model.hypothesis2experiment.endswith("CryptoQlibModelHypothesis2Experiment")
    assert quant.factor_hypothesis2experiment.endswith("CryptoQlibFactorHypothesis2Experiment")
    assert (factor.train_start, quant.test_start) == ("2021-01-01", "2024-07-01")
    os.environ["CRYPTO_QLIB_FACTOR_EVOLVING_N"] = "3"
    try:
        assert CryptoQlibFactorPropSetting().evolving_n == 3
    finally:
        del os.environ["CRYPTO_QLIB_FACTOR_EVOLVING_N"]
    assert crypto_experiment.SCENARIOS["quant"] is CryptoQlibQuantScenario
