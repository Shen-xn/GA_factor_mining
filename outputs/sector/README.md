# 当前结构化结果

本目录只保留正式策略运行所需产物，以及解释当前结论的少量诊断。历史基线、已被替代的实验矩阵和旧协议归档不在这里展示。

## 日常用户

按以下顺序查看：

1. `strategy/LATEST_STATUS.csv`：数据是否可用、市场状态和仓位；
2. `strategy/LATEST_PLAN.json`：最新收盘计划、下一交易日和阻断原因；
3. `etf_mapping/ETF_EXECUTION_READINESS.json`：ETF执行层是否仅供参考或被阻止；
4. `strategy/LATEST_BROAD_MARKET_RISK.json`：真正宽基含义的大盘交叉诊断，不改变正式仓位；
5. `etf_mapping/LATEST_PROXY_CANDIDATES.csv`：最新目标的人工复核ETF候选，不是订单；
6. `strategy/LATEST_ACTIONS.csv`：全部安全门通过后才出现的操作；
7. `strategy/LATEST_TARGET_PORTFOLIO.csv`：完整板块目标组合；
8. `strategy/LATEST_MARKET_RISK.csv`：板块广度风险、仓位来源与确认进度；
9. `strategy/SUMMARY.csv`、`strategy/ANNUAL_RESULTS.csv`：分段与逐年表现；
10. `strategy/ACCEPTANCE_GATE.json`：当前版本是否通过收益、低频、风险和成本目标；
11. `strategy/TARGET_WEIGHT_TIMELINE.csv`：全部已执行和计划目标的完整权重时间线；
12. `etf_backtest/READINESS.json`：冻结目标翻译到真实ETF后的覆盖率、收益与晋级结论；
13. `forward/STATUS.json`：当前前向协议是否正常；`forward/archive/v8/`保留14日未见证据，v9-v13保留后续冻结基线。

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
| `candidates/` | 低频、健康分和Top20保留区候选为何被拒绝；所有选择数据严格截止2025年末 |
| `factors/` | 当前因子目录、依赖与相关性 |
| `feature_ablation/` | 为什么三项高相关派生因子仍被保留 |
| `low_risk/` | 为什么低风险腿选择 `511880.SH` |
| `etf_mapping/` | 当前板块到权益ETF映射为何尚不能实盘化 |
| `etf_backtest/` | 原始板块目标翻译到真实ETF后实际剩下多少覆盖与收益 |
| `etf_proxy_research/` | 事后登记的语义ETF代理为何没有进入默认映射 |
| `etf_native_research/` | ETF原生、行业子宇宙和行业专用模型为何全部未晋级 |
| `market_information_research/` | 宽基、全A宽度、申万一级、资金流和初始财报扩散为何尚未替代当前基准，以及Top20与固定日历低频候选为何失效 |
| `policy_promotion/` | `simple_v2`为何晋级，以及20/30bp完整路径是否保持相同投资决策 |
| `post_v2_research/` | `simple_v2`晋级后20条开发期候选为何均未打开2024—2025，以及为什么继续等权并停止叠加仓位规则和邻近参数 |
| `validation/` | 当前代码、文档和正式结果的验收摘要 |

除此之外的研究尝试不代表当前产品，不应重新放回本目录；如需恢复，应该作为新的研究协议单独运行。
