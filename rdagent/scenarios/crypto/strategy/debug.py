"""The coder's debug panel, cut from the fitting period.

Code is checked on a small panel before it is backtested. That panel is the most traded
``debug_instruments`` over the last ``debug_days`` up to ``train_end``: it lies inside the fitting
period, so nothing the coder reports to the model (how often a strategy is in the market, how large
its positions are) comes from the selection period or the lockbox. ``jarvis-history panel`` also
writes a ``debug/`` slice, the most recent year; it is not used here for that reason.

The slice is written once to ``<panel>/debug_train/`` and written again when the panel or the
settings it was cut with change.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from rdagent.scenarios.crypto.strategy.conf import CryptoStrategySettings
from rdagent.scenarios.crypto.strategy.engine import FIELDS

DEBUG_DIR = "debug_train"
STAMP = "debug.json"


def _stamp(settings: CryptoStrategySettings) -> dict:
    panel = settings.panel_path
    try:
        written = json.loads((panel / "meta.json").read_text()).get("written")
    except (OSError, ValueError):
        written = (panel / "close.parquet").stat().st_mtime if (panel / "close.parquet").exists() else None
    return {
        "panel_written": written,
        "end": settings.train_end,
        "days": settings.debug_days,
        "instruments": settings.debug_instruments,
    }


def debug_panel(settings: CryptoStrategySettings | None = None) -> Path:
    """The debug panel's directory, written first if it is missing or out of date."""
    settings = settings or CryptoStrategySettings()
    panel = settings.panel_path
    target = panel / DEBUG_DIR
    stamp = _stamp(settings)
    try:
        if json.loads((target / STAMP).read_text()) == stamp:
            return target
    except (OSError, ValueError):
        pass
    close = pd.read_parquet(panel / "close.parquet")
    times = pd.DatetimeIndex(close.index)
    end = pd.Timestamp(settings.train_end, tz="UTC") + pd.Timedelta(days=1)
    rows = (times > end - pd.Timedelta(days=settings.debug_days)) & (times <= end)
    if not rows.any():
        raise ValueError(f"the panel at {panel} has no bars in the {settings.debug_days} days to {settings.train_end}")
    amount = pd.read_parquet(panel / "amount.parquet").reindex(index=close.index, columns=close.columns)
    columns = amount[rows].mean().sort_values(ascending=False).dropna().index[: settings.debug_instruments]
    staging = panel / f".{DEBUG_DIR}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in (*FIELDS, "pool"):
        frame = pd.read_parquet(panel / f"{name}.parquet").reindex(index=close.index, columns=close.columns)
        frame.loc[rows, list(columns)].to_parquet(staging / f"{name}.parquet")
    (staging / STAMP).write_text(json.dumps(stamp))
    shutil.rmtree(target, ignore_errors=True)
    try:
        staging.rename(target)
    except OSError:  # another process wrote it first
        shutil.rmtree(staging, ignore_errors=True)
    return target
