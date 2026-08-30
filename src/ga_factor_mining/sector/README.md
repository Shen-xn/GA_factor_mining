# Sector Strategy

默认产品链路只有四层：

1. `rotation/run_experiments.py`：构建无未来信息的板块特征；
2. `rotation/rolling_validation.py`：生成扩展窗口LightGBM评分；
3. `rotation/strategy.py`：把评分转换成低频买卖和持仓状态；
4. `rotation/product_backtest.py`：按次日开盘成交，输出净值、仓位、建议和成本。

直接运行：

```powershell
python -m ga_factor_mining.sector
```

先更新新增行情再运行：

```powershell
python -m ga_factor_mining.sector --update
```

默认使用季度扩展窗口 LightGBM 评分、`simple_v1`、货币ETF `511880.SH` 和单边20bp成本，只执行一次2018年起连续承接的正式回放，不在选择期或观察期重置状态。季度重训只在2024-2025与年度基线做过一次对照并通过固定门槛，不会在日常运行中继续搜索频率。`--update`只更新新增日期和缓存尾部，不运行GA或策略搜索。其他模块均为可选研究诊断，不属于日常运行入口。

理论引擎和产品账本都不把缺失开盘价或收益填成0。候选缺少完整收益时不进入理论组合；已经持有的板块若缺少必要行情，回放会停止并报告具体日期和代码。

产品流程只读取必要特征列。2026观察期承接2024-2025选择期末持仓状态；两段数据都已经被研究者看到，不能再作为未来候选的独立样本外证明。当前板块到权益ETF的严格历史映射覆盖不足，因此普通板块动作仍是研究建议，不是可直接下单的ETF指令。

默认运行还会维护 `outputs/sector/forward/` 的冻结前向账本。冻结日为2026-08-10；之后的新数据才计入未见样本表现。决策代码或协议变化时停止追加，必须明确建立新版本，不能覆盖旧证据。

理论评分到产品的同口径对照使用：

```powershell
python -m ga_factor_mining.sector.rotation.return_bridge
```

该诊断只读取15列，不改变默认策略；五条路径使用完全相同的2018-01-03至2026-08-10收益日期。
