# 板块模块开发说明

用户入口只有一个：

```powershell
python -m ga_factor_mining.sector
```

运行前自检：

```powershell
python -m ga_factor_mining.sector --check
```

## 正式链路

1. `rotation/run_experiments.py`：构建无未来信息的板块特征；
2. `rotation/rolling_validation.py`：生成扩展窗口LightGBM评分；
3. `rotation/strategy.py`：将评分转换成低频持仓状态；
4. `rotation/risk.py`：市场状态和硬风险约束；
5. `rotation/low_risk.py`：构造货币ETF真实收益；
6. `rotation/product_backtest.py`：生成连续账本、指标和用户建议；
7. `rotation/etf_mapping.py`：解析ETF参考组合并执行映射/新鲜度安全门；
8. `rotation/etf_backtest.py`：冻结原目标并使用真实ETF复权开盘价回放实施层；
9. `rotation/forward_monitor.py`：维护冻结协议后的新样本记录。

`doctor.py`只做运行前检查，不参与投资决策。默认运行读取已冻结的季度评分、`simple_v2`和20bp成本，不执行研究搜索。`simple_v2`以Top20为持仓保留区、至少持有10日；组合回撤状态使用固定20bp政策参考净值，真实账户仍按实际成本完整记账。产品账本和ETF实施回放在两个进程中串行运行，避免同一进程累计大对象。产品时间轴明确拆成`planned`、`executed_unsettled`和`settled`：最新收盘计划不需要未来收益，只有下一开盘真实出现后才改变模拟持仓。

`risk.py`同时输出0—100的板块广度风险解释分。仓位仍由已冻结的离散状态与回撤保护决定；风险分暂不直接替代仓位规则，避免在没有封存验证时改变基准收益。产品日账本分别记录`market_base_exposure`、`drawdown_cap`、`risk_target_exposure`和实际组合仓位。

## 正式模块与研究模块

| 类型 | 模块 | 默认运行 |
|---|---|---|
| 正式 | `strategy.py`、`risk.py`、`low_risk.py`、`product_backtest.py` | 是 |
| 正式 | `forward_monitor.py` | 是 |
| 数据维护 | `refresh_data.py` | 仅`--update` |
| 研究 | `run_experiments.py`、`rolling_validation.py`、`adaptive_validation.py` | 否 |
| 研究 | `feature_ablation.py`、`ga_ablation.py`、`market_context_ablation.py` | 否 |
| 诊断 | `return_bridge.py`、`prototype_recovery.py`、`sector_strength_validation.py` | 否 |
| 正式安全门 | `etf_mapping.py`的最新解析与执行就绪检查 | 是 |
| 正式诊断 | `etf_backtest.py`的冻结目标真实ETF回放 | 是 |
| 研究 | `etf_proxy_research.py`的预登记语义代理检验 | 否 |
| 诊断 | `bad_year_attribution.py`、`etf_mapping.py`的历史映射研究 | 否 |

研究产物保留在 `outputs/sector/`，但未通过固定门槛的模块不能改变默认模型和策略。

## 数据与缓存

路径全部以仓库根目录为基准，不允许在代码中写个人绝对路径。运行数据契约见 `docs/DATA_CONTRACT.md`。

产品入口只投影读取必要列。缓存缺失或指纹不一致时会立即停止，不会在日常流程中自动触发全量特征构建。完整研究重建可能占用较多内存，应单独执行并设置底层数值库为单线程。

## 研究协议

- 5日标签为 `open[t+6] / open[t+1] - 1`；
- 训练样本标签必须在训练截止日前兑现；
- 横截面排名在实际投资宇宙内部重新计算；
- 2018—2023为开发期；
- 2024—2025参与模型频率和策略选择；
- 2026为已经观察过的诊断期；
- 产品状态从2018年连续承接，不在年度边界重置；
- 信号形成时不能查看未来开盘可用性；执行日缺报价记为未成交，持仓估值日缺报价按价格持平处理并显式计数；

## 可选诊断

```powershell
python -m ga_factor_mining.sector.rotation.return_bridge
python -m ga_factor_mining.sector.rotation.sector_strength_validation
python -m ga_factor_mining.sector.rotation.product_backtest --cost-sensitivity
python -m ga_factor_mining.sector.rotation.product_backtest --boundary-sensitivity
python -m ga_factor_mining.sector.rotation.etf_proxy_research
```

这些命令会写入结构化输出，但不会自动改变正式策略。

语义ETF代理必须先写入`configs/sector/etf_proxy_hypotheses.json`，随后才允许检验；结果只写诊断目录，不能自动修改默认映射。当前四个事后登记假设在截止2025年末的历史闸门中均未通过。

成本压力命令会用四个相互隔离的子进程依次重放10/20/30/50bp，再由独立进程生成正式账本和统一的`ACCEPTANCE_GATE.json`，避免重复回放造成内存碎片。普通默认运行只复用策略、评分、特征和低风险数据签名全部一致的完整四档结果，因此不会把已通过的成本门覆盖成“未评估”。`simple_v2`各档实际净值按对应成本记账，但组合回撤状态统一读取20bp政策参考净值，所以成本压力不会暗中变成另一套持仓策略。候选只允许使用2018—2025晋级；2026可作诊断，但`observation_used_for_selection`固定为`false`。本机默认Python原生运行时不稳定时，可用`GA_FACTOR_WORKER_PYTHON`指向另一个已验证的Python解释器，代码、数据和口径不变。

`prototype_recovery`只用于审计旧缓存，需要显式传入`--legacy-panel`和可选的`--expected-results`；个人旧目录不会写入项目代码。当前复现结论已经结构化保存在`outputs/sector/prototype_recovery/`。

## 提交前检查

```powershell
python -m unittest discover -s tests -t . -v
python -m ga_factor_mining.sector --check
python -m ga_factor_mining.sector
git status --short
```

数据、Parquet缓存、模型文件、Token、临时日志和报告排版预览不得进入Git。正式Markdown/PDF报告和不含授权行情的结构化研究摘要可以保留。
