"""The crypto settings: the data root is a real path, whatever `.env` wrote."""

from pathlib import Path

from rdagent.scenarios.crypto.conf import CryptoSettings


def test_a_tilde_in_the_data_root_is_expanded(monkeypatch):
    # `.env` keeps `~` literal, so the settings have to do it; otherwise the root silently does not
    # exist and the prompt falls back to "no data schema is available".
    monkeypatch.setenv("CRYPTO_DATA_ROOT_DIR", "~/jarvis-data")

    assert CryptoSettings().data_root_dir == Path.home() / "jarvis-data"


def test_a_variable_in_the_data_root_is_expanded(monkeypatch):
    monkeypatch.setenv("DATA_HOME", "/srv/jarvis")
    monkeypatch.setenv("CRYPTO_DATA_ROOT_DIR", "$DATA_HOME/jarvis-data")

    assert CryptoSettings().data_root_dir == Path("/srv/jarvis/jarvis-data")


def test_a_plain_path_is_kept(monkeypatch):
    monkeypatch.setenv("CRYPTO_DATA_ROOT_DIR", "/data/crypto")

    assert CryptoSettings().data_root_dir == Path("/data/crypto")
