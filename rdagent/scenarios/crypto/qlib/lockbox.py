"""What a qlib session found, judged outside the period it was chosen on.

    python -m rdagent.scenarios.crypto.qlib.lockbox log/<session>                  # trials and PBO
    python -m rdagent.scenarios.crypto.qlib.lockbox log/<session> --rolling        # walk-forward of the SOTA
    python -m rdagent.scenarios.crypto.qlib.lockbox log/<session> --open-lockbox   # the SOTA on the lockbox

During the loop every workflow runs with the selection period as qlib's test segment, and the
feedback and the SOTA choice read its IC and backtest. So:

- every round is a trial: its daily excess return (with cost) over the selection period is kept
  in its workspace (``ret.parquet``); the probability of backtest overfitting (PBO, CSCV) says how
  often the round that is best on half of the selection period is below the median on the rest;
- ``--rolling`` retrains the final SOTA's workflow each quarter on everything before it (an
  expanding window, inside train, valid and select) and stitches the predictions: rank IC and the
  top-k excess return of a model that never saw the quarter it predicts;
- ``--open-lockbox`` runs the SOTA's workflow once more with the lockbox as qlib's test segment,
  in a copy of its workspace, and deflates its Sharpe by every trial. It writes
  ``<log>/lockbox_opened.json``: a model chosen after that has no out-of-sample period left until
  new data arrives.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdagent.scenarios.crypto.qlib.conf import CRYPTO_QLIB_SETTINGS, CryptoQlibSettings

ROLLING_SCRIPT = Path(__file__).resolve().parent / "rolling_template" / "rolling.py"
MARKER = "lockbox_opened.json"


def _loop(tag: str) -> int:
    for part in tag.split("."):
        if part.startswith("Loop_"):
            return int(part.removeprefix("Loop_"))
    return -1


def rounds(log: Path) -> list[tuple[int, Any, Any]]:
    """(loop index, experiment, feedback) of every round that ran, in order."""
    from rdagent.log.storage import FileStorage

    storage = FileStorage(log)
    feedback = {_loop(message.tag): message.content for message in storage.iter_msg(tag="feedback")}
    found = {}
    for message in storage.iter_msg(tag="runner result"):
        exp = message.content
        if getattr(exp, "result", None) is not None and getattr(exp, "experiment_workspace", None) is not None:
            found[_loop(message.tag)] = exp
    return [(index, found[index], feedback.get(index)) for index in sorted(found)]


def excess_returns(exp: Any) -> pd.Series | None:
    """A round's daily excess return with cost, from the backtest report in its workspace."""
    path = Path(exp.experiment_workspace.workspace_path) / "ret.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if not {"return", "bench", "cost"} <= set(frame.columns):
        return None
    return (frame["return"] - frame["bench"] - frame["cost"]).astype(np.float64)


def _per_bar_sharpe(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1))


def trials(found: list[tuple[int, Any, Any]]) -> dict[str, Any]:
    """Every round's selection-period excess returns, their per-day Sharpe ratios, and PBO."""
    import phasor

    series = {index: excess_returns(exp) for index, exp, _ in found}
    series = {index: values for index, values in series.items() if values is not None and len(values) > 1}
    sharpes = [_per_bar_sharpe(values.to_numpy()) for values in series.values()]
    out: dict[str, Any] = {"rounds": len(series), "sharpes": [value for value in sharpes if math.isfinite(value)]}
    if len(series) >= 2:
        matrix = pd.concat(series, axis=1).fillna(0.0).to_numpy(dtype=np.float64)
        splits = min(16, (len(matrix) // 2) * 2)
        out["pbo"] = float(phasor.robust.pbo_cscv(np.asfortranarray(matrix), n_splits=splits))
    else:
        out["pbo"] = float("nan")
    return out


def sota(found: list[tuple[int, Any, Any]]) -> tuple[int, Any] | None:
    """The last accepted round: the session's final SOTA."""
    accepted = [(index, exp) for index, exp, feedback in found if feedback is not None and feedback.decision]
    return accepted[-1] if accepted else None


def _copy_workspace(exp: Any, target: Path) -> Any:
    source = Path(exp.experiment_workspace.workspace_path)
    if not source.exists():
        raise FileNotFoundError(f"the SOTA's workspace {source} is gone; rerun the session to rebuild it")
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    workspace = copy.copy(exp.experiment_workspace)
    workspace.workspace_path = target
    return workspace


def rolling(exp: Any, target: Path, settings: CryptoQlibSettings = CRYPTO_QLIB_SETTINGS) -> dict[str, Any]:
    """The SOTA's workflow retrained each quarter on everything before it, within train to select."""
    from rdagent.scenarios.crypto.qlib.workspace import crypto_qlib_env

    workspace = _copy_workspace(exp, target)
    name, run_env = getattr(exp.experiment_workspace, "last_run", (None, {}))
    if name is None:
        raise ValueError("the SOTA's workspace has no recorded run; it was made before the lockbox support")
    shutil.copy(ROLLING_SCRIPT, target / "rolling.py")
    first = date.fromisoformat(settings.train_start) + timedelta(days=548)  # a year and a half to learn from
    env = {
        **run_env,
        **settings.run_env(),
        "rolling_first": first.replace(day=1).isoformat(),
        "rolling_end": settings.run_env()["test_end"],
        "rolling_valid_days": "180",
        "rolling_embargo_days": str(settings.embargo_days),
    }
    environment = crypto_qlib_env()
    environment.prepare()
    log = environment.check_output(local_path=str(workspace.workspace_path), entry=f"python rolling.py {name}", env=env)
    path = target / "rolling.json"
    if not path.exists():
        raise RuntimeError(f"the rolling run wrote no result:\n{log[-3000:]}")
    return json.loads(path.read_text())


def lockbox(exp: Any, target: Path, sharpes: list[float]) -> dict[str, Any]:
    """The SOTA's workflow once more, with the lockbox as qlib's test segment, in a copy of its workspace."""
    import phasor

    workspace = _copy_workspace(exp, target)
    name, run_env = getattr(exp.experiment_workspace, "last_run", (None, {}))
    if name is None:
        raise ValueError("the SOTA's workspace has no recorded run; it was made before the lockbox support")
    workspace.lockbox = True
    result, stdout = workspace.execute(qlib_config_name=name, run_env=run_env)
    if result is None:
        raise RuntimeError(f"the lockbox run failed:\n{str(stdout)[-3000:]}")
    found = {"metrics": {key: float(value) for key, value in result.items()}}
    probe = copy.copy(exp)
    probe.experiment_workspace = workspace
    excess = excess_returns(probe)
    if excess is not None:
        values = excess.to_numpy()
        found["days"] = int(values.size)
        found["sharpe"] = (
            float(values.mean() / values.std(ddof=1) * math.sqrt(365)) if values.std(ddof=1) > 0 else math.nan
        )
        found["deflated_sharpe"] = (
            float(phasor.robust.deflated_sharpe(values, np.asarray(sharpes))) if len(sharpes) >= 2 else math.nan
        )
    return found


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rdagent.scenarios.crypto.qlib.lockbox")
    parser.add_argument("log", type=Path)
    parser.add_argument("--rolling", action="store_true", help="walk the final SOTA forward, quarter by quarter")
    parser.add_argument("--open-lockbox", action="store_true", help="run the final SOTA on the lockbox (once)")
    args = parser.parse_args(argv)
    settings = CRYPTO_QLIB_SETTINGS
    found = rounds(args.log)
    if not found:
        print(f"No round ran under {args.log}.")
        return 1
    marker = args.log / MARKER
    if marker.exists():
        opened = json.loads(marker.read_text())
        print(f"The lockbox was opened on {opened.get('opened')} for round {opened.get('round')}.")
    summary = trials(found)
    print(f"{summary['rounds']} rounds with a selection-period backtest; PBO {_cell(summary['pbo'])}.")
    best = sota(found)
    if best is None:
        print("No round was accepted: there is no SOTA to walk forward or to open the lockbox with.")
        return 0
    index, exp = best
    print(f"Final SOTA: round {index}; selection period: " + ", ".join(
        f"{key} {_cell(float(exp.result[key]))}" for key in ("IC", "1day.excess_return_with_cost.annualized_return")
        if key in exp.result.index
    ))  # fmt: skip
    if args.rolling:
        walked = rolling(exp, args.log / "rolling" / f"round_{index}", settings)
        print(
            f"Walk-forward ({len(walked['quarters'])} quarters, {walked['days']} days): IC {_cell(walked['ic'])}, "
            f"ICIR {_cell(walked['icir'])}, rank IC {_cell(walked['rank_ic'])}, top-k excess "
            f"{_cell(walked['topk_excess_annual'])} a year, Sharpe {_cell(walked['topk_excess_sharpe'])}"
        )
    if args.open_lockbox:
        opened = lockbox(exp, args.log / "lockbox" / f"round_{index}", summary["sharpes"])
        if not marker.exists():
            marker.write_text(
                json.dumps(
                    {
                        "opened": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "round": index,
                        "lockbox_start": settings.lockbox_start,
                        "lockbox_end": settings.lockbox_end,
                    },
                    indent=1,
                )
            )
        metrics = opened["metrics"]
        print(
            "Lockbox: "
            + ", ".join(f"{key} {_cell(metrics[key])}" for key in ("IC", "ICIR", "Rank IC") if key in metrics)
            + f"; excess with cost: Sharpe {_cell(opened.get('sharpe', math.nan))}, deflated "
            f"{_cell(opened.get('deflated_sharpe', math.nan))} over {len(summary['sharpes'])} trials"
        )
    else:
        print("The lockbox is closed; --open-lockbox runs the SOTA on it, once the research is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
