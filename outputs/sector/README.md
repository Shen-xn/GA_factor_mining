# 当前结构化结果

本目录只保留正式策略运行所需产物，以及解释当前结论的少量诊断。历史基线、已被替代的实验矩阵和旧协议归档不在这里展示。

## 日常用户

按以下顺序查看：

1. `strategy/LATEST_STATUS.csv`：数据是否可用、市场状态和仓位；
2. `strategy/LATEST_ACTIONS.csv`：本次是否需要操作；
3. `strategy/LATEST_TARGET_PORTFOLIO.csv`：完整目标组合；
4. `strategy/LATEST_MARKET_RISK.csv`：板块广度风险、仓位来源与确认进度；
5. `strategy/SUMMARY.csv`、`strategy/ANNUAL_RESULTS.csv`：分段与逐年表现；
6. `strategy/ACCEPTANCE_GATE.json`：当前版本距收益、低频和成本目标还差什么；
7. `forward/STATUS.json`：当前前向协议是否正常。

`rotation/sector_feature_panel.parquet` 和 `adaptation/SELECTED_SCORES.parquet` 是本地运行缓存，不进入Git，但默认策略必须使用。对应JSON元数据会提交。

## 当前结论的解释材料

| 目录 | 回答的问题 |
|---|---|
| `model_frequency/` | 为什么正式模型采用季度扩展窗口训练 |
| `return_bridge/` | 理论Top5收益如何变成当前低频产品收益 |
| `prototype_recovery/` | 旧高收益原型能否复现，以及旧协议与当前协议差在哪里 |
| `bad_year_attribution/` | 2018和2022为什么亏损 |
| `defensive_exposure/` | 为什么没有采用防御期连续降仓方案 |
| `sector_strength_candidate/` | 为什么“领先板块强则恢复70%仓位”未晋级，且没有打开候选的2026结果 |
| `candidates/` | 三个低频候选为何被拒绝；所有选择数据严格截止2025年末 |
| `factors/` | 当前因子目录、依赖与相关性 |
| `feature_ablation/` | 为什么三项高相关派生因子仍被保留 |
| `low_risk/` | 为什么低风险腿选择 `511880.SH` |
| `etf_mapping/` | 当前板块到权益ETF映射为何尚不能实盘化 |
| `validation/` | 当前代码、文档和正式结果的验收摘要 |

除此之外的研究尝试不代表当前产品，不应重新放回本目录；如需恢复，应该作为新的研究协议单独运行。
