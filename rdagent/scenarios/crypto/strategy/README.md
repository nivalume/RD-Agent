# 加密货币策略挖掘

循环：提出机制假设 → 写成目标仓位（phasor 表达式或 numpy 代码）→ 在 panel 上回测 → 反馈。

## 数据

`jarvis-history panel`（jarvis-trader-service，`feat/strategy-panel` 分支）写宽表 Parquet：每个字段一个文件
（行 = bar 收盘时间 UTC，列 = 标的），`funding_rate` 放在结算时刻所在的 bar，`pool.parquet` 是每个 bar 的
标的池（每月按此前 30 天平均成交额取前 N，无前视）。导出的 `debug/` 是最近一年，落在锁箱里，不用；coder 用的检查样本由
RD-Agent 从训练段最后 `debug_days` 天、成交额前 `debug_instruments` 个标的切出，写在 `<panel>/debug_train/`。

```bash
uv run --package jarvis-history jarvis-history download --kind bar --kind funding --interval 4h
uv run --package jarvis-history jarvis-history panel --interval 4h --top 100 --out ~/jarvis-data/strategy/4h
```

频率由 panel 决定（1h、4h、1d 都可以），`CRYPTO_STRATEGY_PANEL_DIR` 指向它。

## 运行

```bash
python -m rdagent.app.crypto_rd_loop.strategy --loop_n 20
python -m rdagent.scenarios.crypto.strategy.report log/<session>                  # 训练、选择段，PBO，walk-forward
python -m rdagent.scenarios.crypto.strategy.report log/<session> --open-lockbox   # 研究结束后看一次锁箱
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
| 报告 | `report.py` | PBO、walk-forward；`--open-lockbox` 时唯一显示锁箱的地方 |

## 回测

- 收盘 t 的仓位在收盘 t+`delay` 成交（默认 1 个 bar）。
- 每个标的仓位乘以 `target_vol / σ`（σ 为前 `vol_lookback` 个 bar 的实现波动率，倍数上限 `max_leverage`），
  再除以当前池内标的数；池外、已下架为 0。
- `phasor.from_orders`：目标仓位、双向、共享资金、手续费 + 滑点、永续资金费（正费率多头付钱），
  `trades="positions"`（一笔交易 = 一个标的从开仓到平仓的一个来回，中间调仓不算新交易）。
- 每个参数组合都回测，按训练段 Sharpe 选一个；选择段（select）评判。

## 指标（需要 phasor ≥ 0.3.0）

对照《打开量化投资的黑箱》第 9 章，每段（训练、选择、锁箱）：

| 书中指标 | 字段 | 来源 |
|---|---|---|
| 平均收益、波动 | `annual_return`、`annual_volatility` | `phasor.returns.stats` |
| 回撤、恢复期 | `max_drawdown`、`max_drawdown_duration_bars`、`max_drawdown_recovery_bars` | 同上 |
| 风险收益比 | `sharpe`、`sortino`、`calmar`、`omega`、`alpha_sharpe`（去掉对等权市场的 beta） | 同上 |
| 盈利周期占比、收益集中度 | `hit_ratio`、`return_concentration`（最好 5% 的 bar 贡献的利润占比） | 同上 |
| 交易胜率 | `trades`、`trade_win_rate`、`profit_factor`、`mean_winning_trade_bars`、`mean_losing_trade_bars` | 组合的来回交易记录 |
| 预测效力 | `diagnostics.r_squared`、`next_return_correlation`、`next_return_by_position_quintile`、`quintile_monotonicity` | `phasor.predictions.power` |
| 时效衰减 | `diagnostics.signed_forward_return_by_horizon`、`robustness.select_sharpe_by_extra_delay_before_funding`（额外延迟 0–3） | `predictions.decay`、`validation.position_delay_test` |
| 参数敏感度 | `robustness.parameter_neighbourhood`（所选组合与网格邻居的 Sharpe 比） | `phasor.params.neighbors` |
| 重抽样回撤 | `robustness.bootstrap_train_and_select`（按天分块重抽样，回撤与 Sharpe 的分位） | `phasor.robust.bootstrap` |
| 运气、过拟合 | `permutation_p_value_train_and_select`、`select_deflated_sharpe_probability` | `validation.permutation_test`（二维）、`robust.deflated_sharpe` |
| 策略间关联 | 与已接受策略的相关系数、加入后组合 Sharpe 的变化 | `phasor.allocation.marginal_contribution` |

另有换手、多空暴露、持有 bar 数、资金费、成本占毛利比例、分年收益、盈利标的占比。

## 门槛与接受

硬门槛：选择段 Sharpe > `min_select_sharpe`；多延迟 1 个 bar 后损失不超过 `max_delay_loss`；置换 p 值 ≤ `max_p_value`；
设了 `min_neighbour_ratio` 时，邻居组合的训练 Sharpe 均值不低于所选组合的这个比例。

接受的策略组成策略库（各自按训练段波动率缩放到同风险后等权）。过了门槛的策略，只有当它与库内每条策略的选择段相关系数
≤ `max_correlation`，且加入后库的选择段 Sharpe 至少提高 `min_portfolio_gain` 时才入库；本轮有多条时取提高最多的。
库为空时第一条过门槛的入库。只提示不拒绝的情况（延迟后反而更好、参数是孤峰、收益集中在少数 bar）作为警告给到反馈。

## 样本内外

| 段 | 默认 | 用途 |
|---|---|---|
| train | 2021-01-01 ~ 2023-06-30 | 拟合：每个策略的参数组合在这里选；coder 的检查样本取自这里 |
| select | 2023-07-01 ~ 2024-06-30 | 选择：门槛、入库、模型看到的全部数字 |
| lockbox | 2024-07-01 ~ 末尾 | 锁箱：只在报告里、研究结束后看一次 |

- `hide_lockbox=true`（默认）时锁箱不进入任何 prompt：`exp.result`（trace 每轮都会打印）、反馈 prompt、
  `exp.strategy_returns`（建库用）都只有训练和选择段。完整结果在 `exp.strategy_results`。
- 选择段被模型反复看到，多轮之后它也是样本内；报告用两种办法估计这一点：对全部试验的选择段收益做 PBO（CSCV），
  对入库策略做扩展窗口 walk-forward（每季度用此前各季度的 Sharpe 重新选参数组合，只在训练+选择段内）。
- 报告默认不显示锁箱；`--open-lockbox` 显示一次，并在日志目录写 `lockbox_opened.json`。之后再选出的策略在锁箱上
  已不是样本外，只能等锁箱之后的新数据（前向测试）。锁箱的 deflated Sharpe 计入整个会话的全部试验，策略库的锁箱
  Sharpe 是整个挖掘过程最终的样本外数字。
- 旧设置名 `CRYPTO_STRATEGY_VALID_*`、`TEST_*`、`HIDE_TEST`、`MIN_VALID_SHARPE` 仍可用。
