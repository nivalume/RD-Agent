"""Crypto cross-sectional factor and model mining on qlib.

The loops are this app's own (:mod:`rdagent.app.crypto_rd_loop.qlib_loops`): propose, code, run on
qlib, feedback, on the crypto scenario of :mod:`rdagent.scenarios.crypto.qlib`::

    python -m rdagent.app.crypto_rd_loop.qlib factor --loop_n 5    # factors, with LightGBM
    python -m rdagent.app.crypto_rd_loop.qlib model --loop_n 5     # models, on the base features
    python -m rdagent.app.crypto_rd_loop.qlib quant --loop_n 10    # each round picks factor or model

Before the first run, write the data with ``jarvis-history qlib --out ~/.qlib/qlib_data/crypto_data
--top 100`` (jarvis-trader-service). The first loop also writes the factor coder's tables
(``git_ignore_folder/crypto_factor_source_data``) from that directory.

Resume a dumped session the way the qlib loops do::

    python -m rdagent.app.crypto_rd_loop.qlib factor --path <log>/__session__/1/0_propose --step_n 1
"""

import asyncio
from pathlib import Path
from typing import Optional

import fire
from dotenv import load_dotenv

# The settings below are built when their modules are imported; load the repository's .env first.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from rdagent.app.crypto_rd_loop.qlib_conf import (  # noqa: E402
    CryptoQlibFactorPropSetting,
    CryptoQlibModelPropSetting,
    CryptoQlibQuantPropSetting,
)
from rdagent.app.crypto_rd_loop.qlib_loops import (  # noqa: E402
    CryptoQlibFactorRDLoop,
    CryptoQlibModelRDLoop,
    CryptoQlibQuantRDLoop,
)
from rdagent.scenarios.crypto.qlib.data import use_crypto_factor_data  # noqa: E402

LOOPS = {
    "factor": (CryptoQlibFactorRDLoop, CryptoQlibFactorPropSetting),
    "model": (CryptoQlibModelRDLoop, CryptoQlibModelPropSetting),
    "quant": (CryptoQlibQuantRDLoop, CryptoQlibQuantPropSetting),
}


def _run(
    kind: str,
    path: Optional[str],
    step_n: Optional[int],
    loop_n: Optional[int],
    all_duration: Optional[str],
    checkout: bool,
    base_features_path: Optional[str],
) -> None:
    loop_cls, setting_cls = LOOPS[kind]
    # A resumed loop does not build its scenario again, but its factor coder still reads the tables.
    use_crypto_factor_data()
    loop = loop_cls(setting_cls()) if path is None else loop_cls.load(path, checkout=checkout)
    loop._init_base_features(base_features_path)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


def factor(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    base_features_path: Optional[str] = None,
) -> None:
    """Mine factors: each round proposes factors, codes them, and runs them with LightGBM."""
    _run("factor", path, step_n, loop_n, all_duration, checkout, base_features_path)


def model(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    base_features_path: Optional[str] = None,
) -> None:
    """Mine models: each round proposes a model, codes it, and trains it on the base features."""
    _run("model", path, step_n, loop_n, all_duration, checkout, base_features_path)


def quant(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    base_features_path: Optional[str] = None,
) -> None:
    """Mine both: each round picks a factor or a model step (a bandit by default)."""
    _run("quant", path, step_n, loop_n, all_duration, checkout, base_features_path)


if __name__ == "__main__":
    fire.Fire({"factor": factor, "model": model, "quant": quant})
