"""The crypto qlib scenario: settings, workflow configs, workspace, converters and scenarios."""

import json
import os
from pathlib import Path

import numpy as np
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
from rdagent.core.proposal import Hypothesis, Trace
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
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.utils.env import QlibCondaEnv

CONFIGS = sorted([*FACTOR_TEMPLATE.glob("*.yaml"), *MODEL_TEMPLATE.glob("*.yaml")])


def render(path: Path, env: dict[str, str]) -> dict:
    """A workflow config as qrun renders it: jinja with the run's environment."""
    return yaml.safe_load(Template(path.read_text()).render(**env))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_the_workflow_configs_are_the_crypto_ones(path):
    settings = CryptoQlibSettings(market="top50", topk=10, n_drop=2, open_cost=0.0004, lockbox_end="2025-01-01")
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
    # The loop's test segment is the selection period, two days short for the labels; nothing after it
    # is loaded. YAML reads a date; so does qlib.
    assert str(backtest["backtest"]["end_time"]) == "2024-06-28"
    assert str(config["data_handler_config"]["end_time"]) == "2024-06-28"
    segments = config["task"]["dataset"]["kwargs"]["segments"]
    assert [str(day) for day in segments["train"]] == ["2021-01-01", "2022-12-29"]
    assert [str(day) for day in segments["valid"]] == ["2023-01-01", "2023-06-28"]
    assert [str(day) for day in segments["test"]] == ["2023-07-01", "2024-06-28"]
    locked = render(path, {**env, **settings.run_env(lockbox=True)})
    assert [str(day) for day in locked["task"]["dataset"]["kwargs"]["segments"]["test"]] == ["2024-07-01", "2025-01-01"]
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
                for word in ("crypto_", "region: us", "2021-", "2022-", "2023-", "2024-", "null", "min_cost: 0", "365")
            )
            for line in changed
        ), changed


def test_the_run_env_holds_every_setting():
    env = CryptoQlibSettings().run_env()
    assert env["crypto_provider_uri"] == "~/.qlib/qlib_data/crypto_data" and env["crypto_market"] == "top100"
    assert (env["test_start"], env["test_end"]) == ("2023-07-01", "2024-06-28")  # the selection period
    locked = CryptoQlibSettings().run_env(lockbox=True)
    assert locked["test_start"] == "2024-07-01" and "test_end" not in locked  # up to the last day of the data
    assert CryptoQlibSettings().run_env(lockbox=True, last_day="2025-02-01")["test_end"] == "2025-02-01"
    assert CryptoQlibSettings(lockbox_end="2025-01-01").run_env(lockbox=True)["test_end"] == "2025-01-01"


def test_no_segment_reads_the_prices_of_the_next():
    settings = CryptoQlibSettings()
    env = settings.run_env()
    days = [
        env["train_start"],
        env["train_end"],
        env["valid_start"],
        env["valid_end"],
        env["test_start"],
        env["test_end"],
    ]
    from datetime import date

    stops = [date.fromisoformat(day) for day in days]
    # A label reads the next two closes: each segment ends at least embargo_days before the next begins.
    for end, start in (
        (stops[1], stops[2]),
        (stops[3], stops[4]),
        (stops[5], date.fromisoformat(settings.lockbox_start)),
    ):
        assert (start - end).days > settings.embargo_days
    assert settings.debug_start == "2022-01-01"  # the debug data ends with the fitting period


def test_the_old_test_names_set_the_lockbox(monkeypatch):
    monkeypatch.setenv("CRYPTO_QLIB_TEST_START", "2024-09-01")
    monkeypatch.setenv("CRYPTO_QLIB_TEST_END", "2025-03-01")
    settings = CryptoQlibSettings()
    assert (settings.lockbox_start, settings.lockbox_end) == ("2024-09-01", "2025-03-01")


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
    # The loop runs on the selection period, never the runner's CN date.
    assert (seen["test_start"], seen["test_end"]) == ("2023-07-01", "2024-06-28")
    assert workspace.last_run == ("conf_baseline.yaml", runner_env)
    # The lockbox run: no end set, so it ends the day before the last of the calendar.
    workspace.lockbox = True
    workspace.execute("conf_baseline.yaml", runner_env)
    assert (seen["test_start"], seen["test_end"]) == ("2024-07-01", "2023-12-31")


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

    crypto_data._stamp_path(settings).write_text(crypto_data._debug_stamp(settings))

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


def test_the_loops_wire_the_crypto_classes():
    factor, model, quant = CryptoQlibFactorPropSetting(), CryptoQlibModelPropSetting(), CryptoQlibQuantPropSetting()
    assert factor.scen.endswith("CryptoQlibFactorScenario")
    assert factor.hypothesis_gen.endswith("crypto.proposal.factor_proposal.CryptoFactorHypothesisGen")
    assert factor.runner == "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    assert model.hypothesis2experiment.endswith("CryptoQlibModelHypothesis2Experiment")
    assert quant.factor_hypothesis2experiment.endswith("CryptoQlibFactorHypothesis2Experiment")
    assert (factor.train_start, quant.test_start) == ("2021-01-01", "2023-07-01")  # the selection period
    os.environ["CRYPTO_QLIB_FACTOR_EVOLVING_N"] = "3"
    try:
        assert CryptoQlibFactorPropSetting().evolving_n == 3
    finally:
        del os.environ["CRYPTO_QLIB_FACTOR_EVOLVING_N"]
    assert crypto_experiment.SCENARIOS["quant"] is CryptoQlibQuantScenario


def test_the_debug_data_is_written_again_for_another_period(tmp_path, monkeypatch):
    from rdagent.scenarios.crypto.qlib import data

    written = []
    settings = CryptoQlibSettings(
        factor_data_folder=str(tmp_path / "all"), factor_data_folder_debug=str(tmp_path / "debug")
    )

    def generate(settings):
        written.append(settings.train_end)
        for folder in (settings.factor_data_folder, settings.factor_data_folder_debug):
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / "daily_pv.h5").write_text("")
        data._stamp_path(settings).write_text(data._debug_stamp(settings))

    monkeypatch.setattr(data, "generate_crypto_factor_data", generate)
    data.use_crypto_factor_data(settings)
    data.use_crypto_factor_data(settings)
    assert written == ["2022-12-31"]
    data._stamp_path(settings).write_text("2024-01-01 onwards")  # cut before this change
    data.use_crypto_factor_data(settings)
    assert len(written) == 2


# -- the lockbox command -------------------------------------------------------------------------


class _Feedback:
    def __init__(self, decision):
        self.decision = decision


def _round(tmp_path, name, mean, seed, decision):
    rng = np.random.default_rng(seed)
    workspace = CryptoQlibFBWorkspace(template_folder_path=FACTOR_TEMPLATE)
    workspace.workspace_path = tmp_path / name
    workspace.workspace_path.mkdir(parents=True)
    days = pd.date_range("2023-07-01", periods=360, freq="D")
    pd.DataFrame({"return": rng.normal(mean, 0.01, 360), "bench": 0.0, "cost": 0.0001}, index=days).to_parquet(
        workspace.workspace_path / "ret.parquet"
    )
    workspace.last_run = ("conf_combined_factors.yaml", {"feature_names": "[]"})
    exp = CryptoQlibFactorExperiment(sub_tasks=[])
    exp.experiment_workspace = workspace
    exp.result = pd.Series({"IC": 0.03, "1day.excess_return_with_cost.annualized_return": 0.1})
    return exp, _Feedback(decision)


def test_trials_pbo_and_the_final_sota(tmp_path):
    from rdagent.scenarios.crypto.qlib import lockbox

    found = [(k, *_round(tmp_path, f"r{k}", 0.0, k, decision=k in (1, 3))) for k in range(6)]
    summary = lockbox.trials(found)
    assert summary["rounds"] == 6 and len(summary["sharpes"]) == 6 and 0 <= summary["pbo"] <= 1
    assert lockbox.sota(found)[0] == 3
    assert lockbox.sota([(0, found[0][1], _Feedback(False))]) is None


def test_the_lockbox_runs_on_a_copy_and_is_marked(tmp_path, monkeypatch, capsys):
    from rdagent.scenarios.crypto.qlib import lockbox

    found = [(k, *_round(tmp_path / "ws", f"r{k}", 0.001 * k, k, decision=True)) for k in range(3)]
    monkeypatch.setattr(lockbox, "rounds", lambda log: found)
    calls = []

    def execute(self, qlib_config_name="conf.yaml", run_env={}, *args, **kwargs):
        calls.append((qlib_config_name, run_env, self.lockbox, self.workspace_path))
        days = pd.date_range("2024-07-01", periods=200, freq="D")
        rng = np.random.default_rng(9)
        pd.DataFrame({"return": rng.normal(0.002, 0.01, 200), "bench": 0.0, "cost": 0.0}, index=days).to_parquet(
            self.workspace_path / "ret.parquet"
        )
        return pd.Series({"IC": 0.02, "ICIR": 0.2}), ""

    monkeypatch.setattr(CryptoQlibFBWorkspace, "execute", execute)
    log = tmp_path / "log"
    log.mkdir()
    assert lockbox.main([str(log)]) == 0
    printed = capsys.readouterr().out
    assert "The lockbox is closed" in printed and "PBO" in printed and not calls
    assert lockbox.main([str(log), "--open-lockbox"]) == 0
    printed = capsys.readouterr().out
    name, env, locked, path = calls[0]
    assert name == "conf_combined_factors.yaml" and env == {"feature_names": "[]"} and locked
    assert path == log / "lockbox" / "round_2" and path != found[2][1].experiment_workspace.workspace_path
    assert "Lockbox: IC 0.0200" in printed and "deflated" in printed
    # The SOTA's own selection-period backtest is untouched.
    assert len(pd.read_parquet(found[2][1].experiment_workspace.workspace_path / "ret.parquet")) == 360
    assert json.loads((log / lockbox.MARKER).read_text())["round"] == 2
    assert lockbox.main([str(log)]) == 0
    assert "The lockbox was opened on" in capsys.readouterr().out
