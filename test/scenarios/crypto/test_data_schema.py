"""The data root's schema files are scanned and rendered for the prompt."""

import json
from pathlib import Path

from rdagent.scenarios.crypto.experiment.data_schema import load_schemas, render_data_schemas

BAR = {
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

FUNDING = {
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


def write(root: Path, name: str, payload: dict) -> Path:
    path = root / "schemas" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_root(tmp_path: Path, *datasets: str) -> Path:
    """A data root whose dataset directories exist, so the schemas count as available."""
    for dataset in datasets:
        (tmp_path / "crypto" / "data" / dataset).mkdir(parents=True)
    return tmp_path


def test_load_schemas_keeps_the_latest_version(tmp_path):
    write(tmp_path, "market.crypto.bar.v1.json", {**BAR, "description": "old"})
    write(tmp_path, "market.crypto.bar.v2.json", {**BAR, "version": 2, "description": "new"})

    schemas = load_schemas(tmp_path / "schemas")

    assert schemas["market.crypto.bar"].version == 2
    assert schemas["market.crypto.bar"].description == "new"


def test_load_schemas_skips_a_broken_file(tmp_path):
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "broken.json").write_text("{not json", encoding="utf-8")
    write(tmp_path, "market.crypto.bar.v1.json", BAR)

    assert list(load_schemas(tmp_path / "schemas")) == ["market.crypto.bar"]


def test_render_injects_only_the_supported_schemas(tmp_path):
    root = make_root(tmp_path, "crypto_bar", "crypto_funding_rate", "crypto_open_interest")
    write(root, "market.crypto.bar.v1.json", BAR)
    write(root, "market.crypto_funding_rate.v1.json", FUNDING)
    write(root, "market.crypto_open_interest.v1.json", {**FUNDING, "slug": "market.crypto_open_interest"})

    text = render_data_schemas(root)

    assert "market.crypto.bar" in text
    assert "market.crypto_funding_rate" in text
    assert "crypto_open_interest" not in text


def test_render_skips_a_schema_without_its_rows(tmp_path):
    root = make_root(tmp_path)
    write(root, "market.crypto.bar.v1.json", BAR)

    assert render_data_schemas(root) == ""


def test_render_accepts_a_duckdb_file_instead_of_the_csv_partitions(tmp_path):
    # The data root may keep a dataset in one DuckDB file; the rows are still on disk.
    root = make_root(tmp_path)
    write(root, "market.crypto.bar.v1.json", BAR)
    (root / "crypto").mkdir(exist_ok=True)
    (root / "crypto" / "crypto_bar.duckdb").write_bytes(b"")

    text = render_data_schemas(root)

    assert "market.crypto.bar" in text
    assert "| `close` | decimal(38,18) | Last traded price in the bar. |" in text


def test_render_shows_the_columns_with_their_types_and_descriptions(tmp_path):
    root = make_root(tmp_path, "crypto_bar")
    write(root, "market.crypto.bar.v1.json", BAR)

    text = render_data_schemas(root)

    assert "| `close` | decimal(38,18) | Last traded price in the bar. |" in text
    assert "crypto/data/crypto_bar/<instrument>-<bar type>/<start>_<end>.csv.gz" in text
    assert "`ts_open` is the bar's open time" in text
