"""The crypto scenario renders its prompts without an LLM or a configured environment."""

import json
import re
from pathlib import Path

import pytest

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.scenarios.crypto.experiment.factor_experiment import (
    CryptoFactorExperiment,
    CryptoFactorScenario,
    main,
)

pytestmark = pytest.mark.offline

BAR_SCHEMA = {
    "slug": "market.crypto.bar",
    "version": 1,
    "name": "CryptoBar",
    "description": "Crypto bar history.",
    "columns": [
        {"name": "instrument_id", "type": "string"},
        {"name": "close", "type": "decimal", "precision": 38, "scale": 18},
        {"name": "ts_open", "type": "int64"},
    ],
    "annotations": {"fields": {"close": {"description": "Last traded price in the bar."}}},
}

FUNDING_SCHEMA = {
    "slug": "market.crypto_funding_rate",
    "version": 1,
    "name": "CryptoFundingRate",
    "description": "Crypto funding rate history.",
    "columns": [
        {"name": "funding_rate", "type": "decimal", "precision": 38, "scale": 18},
        {"name": "ts_event", "type": "int64"},
    ],
    "annotations": {"fields": {"funding_rate": {"description": "Settled funding rate."}}},
}


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A data root with the bar and funding schemas and their rows."""
    (tmp_path / "schemas").mkdir()
    for payload in (BAR_SCHEMA, FUNDING_SCHEMA):
        path = tmp_path / "schemas" / f"{payload['slug']}.v1.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    for dataset in ("crypto_bar", "crypto_funding_rate"):
        (tmp_path / "crypto" / "data" / dataset).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def scenario(data_root: Path) -> CryptoFactorScenario:
    return CryptoFactorScenario(with_runtime_environment=False, data_root_dir=data_root)


def test_background_comes_from_the_crypto_market(scenario):
    assert "Crypto assets traded on centralized exchanges" in scenario.background
    assert "funding payment between longs and shorts" in scenario.background
    assert "Runtime Environment" not in scenario.background


def test_calendar_describes_the_clock(scenario):
    assert "The clock:" in scenario.calendar
    for aspect in ("session", "funding", "expiries", "weekends"):
        assert f"- {aspect}:" in scenario.calendar
    # The calendar is the market's clock, not a description of the data.
    assert "bar" not in scenario.calendar.lower()


def test_prompts_do_not_name_any_instrument(scenario):
    # Naming a coin lets the model fall back on what it remembers about that coin instead of the data.
    texts = (
        scenario.background,
        scenario.calendar,
        scenario.get_source_data_desc(),
        scenario.output_format,
        scenario.simulator,
    )
    for text in texts:
        assert not re.search(r"\b(BTC|ETH|USDT|BNB|XRP|DOGE|ADA|SOL)\b", text)


def test_source_data_describes_the_injected_schemas(scenario):
    data = scenario.get_source_data_desc()
    assert "market.crypto.bar" in data
    assert "market.crypto_funding_rate" in data
    assert "| `close` | decimal(38,18) | Last traded price in the bar. |" in data
    assert "crypto/data/crypto_funding_rate" in data
    assert "crypto_bars_1h.h5" not in data


def test_source_data_falls_back_when_the_data_root_is_missing(tmp_path):
    scenario = CryptoFactorScenario(with_runtime_environment=False, data_root_dir=tmp_path / "missing")
    assert "No data schema is available" in scenario.get_source_data_desc()


def test_scenario_description_is_the_market_its_calendar_and_the_data(scenario):
    description = scenario.get_scenario_all_desc()
    assert scenario.background in description
    assert scenario.calendar in description
    assert "The source data you can use:" in description
    assert "market.crypto.bar" in description
    for excluded in ("The interface you should follow", "result.h5"):
        assert excluded not in description


def test_experiment_holds_the_factor_tasks():
    task = FactorTask("momentum_10", "10 bar momentum", "mean($close / $close[10] - 1)")
    experiment = CryptoFactorExperiment(sub_tasks=[task])
    assert experiment.sub_tasks == [task]
    assert experiment.experiment_workspace is not None


def test_main_prints_the_whole_prompt(capsys):
    assert main([]) == 0
    printed = capsys.readouterr().out
    assert "Background of the scenario:" in printed
    assert "Crypto assets traded on centralized exchanges" in printed


def test_main_prints_a_named_section(capsys):
    assert main(["calendar"]) == 0
    printed = capsys.readouterr().out
    assert "===== calendar =====" in printed
    assert "The clock:" in printed


def test_main_renders_the_schemas_of_the_data_root(capsys, data_root):
    assert main(["--data-root", str(data_root), "data"]) == 0
    printed = capsys.readouterr().out
    assert "===== data =====" in printed
    assert "market.crypto_funding_rate" in printed


def test_main_rejects_an_unknown_section():
    with pytest.raises(SystemExit):
        main(["unknown"])
