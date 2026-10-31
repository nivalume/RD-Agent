# 加密货币策略挖掘

循环：提出机制假设 → 写成目标仓位（phasor 表达式或 numpy 代码）→ 在 panel 上回测 → 反馈。

## 数据

`jarvis-history panel`（jarvis-trader-service，`feat/strategy-panel` 分支）写宽表 Parquet：每个字段一个文件
（行 = bar 收盘时间 UTC，列 = 标的），`funding_rate` 放在结算时刻所在的 bar，`pool.parquet` 是每个 bar 的
标的池（每月按此前 30 天平均成交额取前 N，无前视），`debug/` 是最近一年、30 个标的的小样本。

```bash
uv run --package jarvis-history jarvis-history download --kind bar --kind funding --interval 4h
uv run --package jarvis-history jarvis-history panel --interval 4h --top 100 --out ~/jarvis-data/strategy/4h
```

频率由 panel 决定（1h、4h、1d 都可以），`CRYPTO_STRATEGY_PANEL_DIR` 指向它。

## 运行

```bash
python -m rdagent.app.crypto_rd_loop.strategy --loop_n 20
python -m rdagent.scenarios.crypto.strategy.report log/<session>     # 三段表现 + deflated Sharpe
```

设置见 `.env.example` 的 `CRYPTO_STRATEGY_*`。

## 各部分

| 步骤 | 代码 | 做什么 |
|---|---|---|
| 提案 | `scenarios/crypto/proposal/strategy_proposal.py` | 机制、信号、仓位、可证伪预测 |
| 场景 | `scenario.py`、`prompts.yaml` | panel 描述和评估设定（延迟、波动率目标、成本、分段、门槛） |
| coder | `components/coder/strategy_coder/` | LLM 输出 `strategy.json`；在 debug panel 上检查形状、[-1, 1]、非全 0、前视（随机截断重算） |
| runner | `developer.py::StrategyRunner`、`run.py`、`engine.py` | 子进程里在全量 panel 回测每个参数组合 |
| 反馈 | `developer.py::StrategyExperiment2Feedback`、`results.py` | 硬门槛 + 与当前最优比较决定；LLM 只解释并给下一步 |
| 报告 | `report.py` | 唯一显示测试段的地方 |

## 回测

- 收盘 t 的仓位在收盘 t+`delay` 成交（默认 1 个 bar）。
- 每个标的仓位乘以 `target_vol / σ`（σ 为前 `vol_lookback` 个 bar 的实现波动率，倍数上限 `max_leverage`），
  再除以当前池内标的数；池外、已下架为 0。
- `phasor.from_orders`：目标仓位、双向、共享资金、手续费 + 滑点、永续资金费（正费率多头付钱）。
- 每个参数组合都回测，按训练段 Sharpe 选一个；验证段评判。
- 稳健性：多延迟 1 个 bar 的验证 Sharpe、各组合的验证 Sharpe、训练+验证段的分块置换检验 p 值、
  对历史全部试验的 deflated Sharpe。
- 诊断：按持有期的方向收益（检验 `testable_prediction` 的符号和衰减）、按仓位大小分组的下一根收益。

## 门槛与选择

验证 Sharpe > `min_valid_sharpe`；多延迟 1 个 bar 后损失不超过 `max_delay_loss`；置换 p 值 ≤ `max_p_value`。
通过的策略里，验证段 alpha Sharpe（去掉对等权市场的 beta）最高者，若高于此前最优，本轮接受。

## 测试段

`hide_test=true`（默认）时，测试段数字不进入任何 prompt：`exp.result`（trace 每轮都会打印）、反馈 prompt
都只含训练和验证段；完整结果在 `exp.strategy_results`，只由 `report.py` 读出。报告里的 deflated Sharpe
用整个会话的全部试验（每个策略的每个参数组合）计算。
