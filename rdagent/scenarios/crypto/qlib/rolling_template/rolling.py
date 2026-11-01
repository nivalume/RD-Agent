"""An expanding walk-forward of a workflow config: each quarter predicted by a model trained on
everything before it, the predictions stitched into one out-of-sample series.

Run inside the qlib environment, in a copy of the workspace (the config and its data files):

    python rolling.py <config.yaml>

The config is rendered as ``qrun`` renders it (from the environment). Environment: ``rolling_first``
(first quarter traded), ``rolling_end`` (last day), ``rolling_valid_days`` (early-stopping window
before each quarter), ``rolling_embargo_days``, ``crypto_topk``. Writes ``rolling.json`` and
``rolling_daily.csv``.
"""

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import qlib
from qlib.cli.run import render_template
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from ruamel.yaml import YAML


def _set_fit(node, start, end):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "fit_start_time":
                node[key] = start
            elif key == "fit_end_time":
                node[key] = end
            else:
                _set_fit(value, start, end)
    elif isinstance(node, list):
        for value in node:
            _set_fit(value, start, end)


config = YAML(typ="safe", pure=True).load(render_template(sys.argv[1]))
qlib.init(**config["qlib_init"])
task = config["task"]
segments = task["dataset"]["kwargs"]["segments"]
start = pd.Timestamp(str(segments["train"][0]))
first = pd.Timestamp(os.environ["rolling_first"])
end = pd.Timestamp(os.environ["rolling_end"])
valid_days = int(os.environ.get("rolling_valid_days", "180"))
embargo = pd.Timedelta(days=int(os.environ.get("rolling_embargo_days", "2")))
topk = int(os.environ.get("crypto_topk", "20"))

frames, quarters = [], []
for q_start in pd.date_range(first, end, freq="QS"):
    q_end = min(q_start + pd.offsets.QuarterEnd(0), end)
    valid_end = q_start - embargo - pd.Timedelta(days=1)
    valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
    train_end = valid_start - embargo - pd.Timedelta(days=1)
    piece = copy.deepcopy(task)
    piece["dataset"]["kwargs"]["segments"] = {
        "train": [str(start.date()), str(train_end.date())],
        "valid": [str(valid_start.date()), str(valid_end.date())],
        "test": [str(q_start.date()), str(q_end.date())],
    }
    _set_fit(piece["dataset"], str(start.date()), str(train_end.date()))
    model = init_instance_by_config(piece["model"])
    dataset = init_instance_by_config(piece["dataset"])
    model.fit(dataset)
    score = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    frame = pd.concat({"score": score, "label": label}, axis=1).dropna()
    frames.append(frame)
    quarters.append([str(q_start.date()), str(q_end.date()), str(train_end.date())])
    print(f"{q_start.date()}..{q_end.date()}: trained to {train_end.date()}, {len(frame)} predictions", flush=True)

data = pd.concat(frames).sort_index()
daily = []
for day, group in data.groupby(level="datetime"):
    if len(group) < 2:
        continue
    ic = group["score"].corr(group["label"])
    rank_ic = group["score"].rank().corr(group["label"].rank())
    top = group.nlargest(min(topk, len(group)), "score")["label"].mean()
    daily.append({"datetime": day, "ic": ic, "rank_ic": rank_ic, "topk_excess": top - group["label"].mean()})
daily = pd.DataFrame(daily).set_index("datetime")
daily.to_csv("rolling_daily.csv")
excess = daily["topk_excess"]
summary = {
    "quarters": quarters,
    "days": int(len(daily)),
    "ic": float(daily["ic"].mean()),
    "icir": float(daily["ic"].mean() / daily["ic"].std()) if daily["ic"].std() > 0 else float("nan"),
    "rank_ic": float(daily["rank_ic"].mean()),
    "topk_excess_annual": float(excess.mean() * 365),
    "topk_excess_sharpe": float(excess.mean() / excess.std() * np.sqrt(365)) if excess.std() > 0 else float("nan"),
}
with open("rolling.json", "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps(summary))
