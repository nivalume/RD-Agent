# 加密货币横截面因子与模型挖掘（qlib）

RD-Agent 的 qlib 循环（提出假设 → 写代码 → 在 qlib 上训练回测 → 反馈），换成加密货币数据：

- 数据：`jarvis-history qlib` 写出的 qlib 数据目录。全部 USDT 本位永续（含已下架），每个 UTC 日一行；
  `top<N>` 标的池每月初按此前 30 天的平均成交额选出，没有前视。
- 场景描述：市场与日历用 `rdagent/scenarios/crypto/experiment/prompts.yaml` 里的加密货币 prompt
  （`crypto_market`、`crypto_quant_background`、`crypto_quant_calendar`），评估设定（标的池、标签、模型、
  回测、成本）在本目录的 `prompts.yaml`。
- 因子提案用 `CryptoFactorHypothesisGen` 和加密货币因子 prompt；模型提案、coder、runner、feedback 用 qlib 的。
- 工作流配置 `factor_template/`、`model_template/` 由 qlib 的配置生成，只改了：数据目录、标的池、基准
  （BTCUSDT）、日期、无涨跌停、Taker 手续费、`min_cost: 0`、topk/n_drop、年化按 365 天。

## 准备

1. 下载数据并导出（jarvis-trader-service 仓库）：

   ```bash
   # universe.toml 已包含全部 USDT 永续的 1d K 线和资金费率
   uv run --package jarvis-history jarvis-history download --kind bar --kind funding --interval 1d
   uv run --package jarvis-history jarvis-history qlib --out ~/.qlib/qlib_data/crypto_data --top 100
   ```

   数据目录必须在 `~/.qlib` 下：docker 模式只挂载这个目录。

2. 在 `.env` 设置（见 `.env.example` 的 crypto qlib 部分）：
   - `MODEL_CoSTEER_ENV_TYPE=docker` 或 `conda`：qlib 在哪里跑。docker 用 RD-Agent 的 qlib 镜像；conda 用
     `rdagent4qlib` 环境（注意 mlflow 需 < 3，mlflow 3 不再支持 qlib 用的文件存储）。
   - `CRYPTO_QLIB_MARKET`、`CRYPTO_QLIB_TOPK`、`CRYPTO_QLIB_N_DROP`、成本、各段日期。
   - LLM 的配置与其他场景相同。

## 运行

```bash
python -m rdagent.app.crypto_rd_loop.qlib factor --loop_n 5   # 因子挖掘，LightGBM
python -m rdagent.app.crypto_rd_loop.qlib model --loop_n 5    # 模型挖掘，基础特征
python -m rdagent.app.crypto_rd_loop.qlib quant --loop_n 10   # 每轮由 bandit 选因子或模型
python -m rdagent.scenarios.crypto.qlib.experiment factor     # 只打印场景 prompt，不调用 LLM
```

第一次运行时，会在 qlib 环境里执行 `factor_data_template/generate.py`，把因子代码读的 `daily_pv.h5`
写到 `git_ignore_folder/crypto_factor_source_data`（以及 30 个标的的 debug 版，取训练段最后一年）。数据更新后删掉这两个目录即可重新生成；
debug 版的区间设置变了会自动重写。

## 评估设定（默认）

| 项目 | 设定 |
| --- | --- |
| 标的池 | `top100`，月度调整 |
| 标签 | `Ref($close, -2) / Ref($close, -1) - 1`，按日横截面标准化 |
| 模型 | LightGBM，Alpha20 基础特征 + 挖出的因子 |
| 回测 | 每日持有预测最高的 20 个、每天最多换 4 个、只做多、收盘成交、双边 0.05%、基准 BTCUSDT |
| 分段 | 训练 2021-01-01–2022-12-31（拟合、标准化），验证 2023-01-01–2023-06-30（early stopping），选择 2023-07-01–2024-06-30（qlib 的 test 段：反馈、SOTA 都看它），锁箱 2024-07-01–数据最后一天的前一天；训练、验证、选择段各在结尾少 2 天（标签读后两天的收盘） |

回测收益只算价格收益，不含资金费率。

## 代码

| 文件 | 内容 |
| --- | --- |
| `conf.py` | `CryptoQlibSettings`（`CRYPTO_QLIB_` 环境变量） |
| `workspace.py` | `CryptoQlibFBWorkspace`：选 docker/conda、检查数据、每次运行注入加密货币设定 |
| `data.py` | 生成因子 coder 读的 `daily_pv.h5` |
| `experiment.py` | 实验类和 factor/model/quant 三个场景 |
| `proposal.py` | 假设到实验的转换 |
| `rdagent/app/crypto_rd_loop/qlib.py`、`qlib_conf.py` | 三个循环的入口和配置 |

对上游 qlib 代码只有一处改动：`QlibFBWorkspace.qlib_env()`，让场景可以换掉运行环境（docker 模式下不再下载 CN 数据）。

## 样本内外

循环里每个 workflow 的 qlib `test` 段都是选择段，数据也只加载到选择段结束；LLM 看到的 IC、回测和 SOTA 比较都来自选择段。
选择段被反复用来挑选，多轮之后它也是样本内。研究结束后：

```bash
python -m rdagent.scenarios.crypto.qlib.lockbox log/<session>                  # 每轮一个试验：PBO
python -m rdagent.scenarios.crypto.qlib.lockbox log/<session> --rolling        # 最终 SOTA 的扩展窗口 walk-forward
python -m rdagent.scenarios.crypto.qlib.lockbox log/<session> --open-lockbox   # 最终 SOTA 在锁箱上跑一次
```

- PBO：各轮选择段每日超额收益（含成本）做 CSCV。
- `--rolling`：在 SOTA 工作区的副本里，每季度用此前全部数据（最后 180 天做 early stopping）重新训练，拼出一条
  从未见过所预测季度的预测，给出 IC、ICIR、rank IC 和 top-k 超额收益；只在训练到选择段内。
- `--open-lockbox`：在副本里把 qlib 的 test 段换成锁箱再跑一次，Sharpe 按全部轮次做 deflated；写 `lockbox_opened.json`，
  之后再选出的模型在锁箱上已不是样本外。
