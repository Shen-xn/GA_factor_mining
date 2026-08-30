# GA Factor Mining - Sector Strategy

当前分支是 `sector_dev`，只包含板块轮动研究。个股项目位于独立的 `stock_dev` 分支。

## 现在有什么

项目已经有一个可直接运行的板块轮动原型：

- 扩展窗口 LightGBM，自2024年起每季度重新训练；
- 使用已经冻结的人工特征和模型参数；
- `t` 日收盘生成信号，`t+1` 日开盘交易；
- 缺失开盘价或收益不会被当作0收益，数据不完整会被拦截；
- `simple_v1` 只保留Top5入场、跌出Top10退出、至少持有5日和硬风险退出；
- 未配置资金进入货币ETF `511880.SH`，不固定持有红利板块；
- 默认计入单边20bp交易成本。

当前结构化结果采用2018年起唯一连续状态路径：2024-2025选择期累计收益约36.6%，年化约17.6%，区间最大回撤约10.0%；2026年截至8月10日累计约0.2%，最大回撤约7.5%。2018-2025八个完整年份中六年为正，完整路径最大回撤约18.9%。2018、2022仍为负，30bp成本下开发期累计也会转负，因此它是可运行的研究原型，还不是通过实盘验收的产品。

2024-2025曾用于选择训练窗口和产品规则，因此这里只称“选择期”，不能再冒充独立样本外验证。2026是已被观察过的诊断期，同样不能用于后续候选晋级；真正的新样本外证据只能来自未来尚未看到的数据。

当前产品协议已经冻结在2026-08-10。以后默认运行会自动追加前向快照；只有该日期之后的数据计入新样本外证据。模型、策略或决策代码指纹发生变化时，系统会停止续接旧成绩，不能静默“换策略后接着算”。

季度重训是唯一频率挑战者：它保持人工特征、LightGBM超参数和`simple_v1`不变，只从2024年开始把年度重训改为季度重训。选择依据只使用2024-2025；累计收益由约28.9%提高到36.8%，Sharpe与回撤同时改善，换手基本不变，因此晋级。2026结果在晋级完成后才查看，不参与选择。

已经单独拒绝两种看似直观的修补：板块负趋势入场门使2024年转亏、验证累计收益降至约8.1%；`DEFENSIVE`状态完全清仓虽然大幅改善2018和2022，却让2018-2023累计收益变负且2019转亏。默认策略不会因为局部年份改善就强行采用这些规则。

当前账本对2018和2022的精确归因显示，板块选择分别贡献约+6.99%和+1.89%，主要损失来自熊市残留仓位，成本又分别拖累约2.82%和2.40%。因此下一步不改LightGBM排序；如需继续测试，只考虑一个最小候选：在`DEFENSIVE`状态内根据市场趋势和宽度动态调整0%-30%的板块仓位。

因子重复筛查只使用2015-2023开发数据决定候选，2024-2025和2026只检查稳定性。当前有9组开发期高相关，其中7组在三个期间都持续高相关，3组是明确的派生表示。三项风险调整收益已经按当前正式历史训练安排和`simple_v1`逐项消融：删除后开发期累计分别为-1.85%、+5.96%和-15.27%，全部未过门，因此没有打开2024-2025或读取2026，正式18因子集保持不变。高相关只用于提出消融候选，不能直接作为删除依据。

GA库目前只有一个单因子shadow候选。旧GA增量结果是在旧持仓规则和2024重置口径下得到的，只能保留为“曾被拒绝”的历史记录，不能冒充当前产品证据。GA消融代码已经改为使用2018连续账本、当前`simple_v1`并止于2025选择期；在明确建立新研究协议前不重跑，也不会修改默认产品。

## 直接运行

```powershell
python -m pip install -e .
python -m ga_factor_mining.sector
```

第二条命令默认使用冻结滚动LightGBM、`simple_v1` 和20bp成本，只回放一次正式原型。不会自动运行参数搜索、GA挖掘、成本压力测试或边界实验。

需要先补齐Tushare行情时只增加一个开关：

```powershell
python -m ga_factor_mining.sector --update
```

更新过程只下载新增交易日、重算尾部特征并延长当前冻结模型评分；不会运行GA、参数搜索或全量特征重建。程序随后照常运行原型。行情距运行日超过7天时，`LATEST_STATUS.csv` 会显示“数据已过期，禁止执行”，并让 `LATEST_ACTIONS.csv` 保持为空。

主要结果在 `outputs/sector/strategy/`：

- `SUMMARY.csv`：开发期、选择期与观察期核心指标；
- `ANNUAL_RESULTS.csv`：2018年至今的逐年结果；
- `HISTORY_DAILY.parquet`、`HISTORY_ACTIONS.parquet`：2018年起唯一连续产品账本；
- `selection_daily.parquet`、`observation_daily.parquet`：逐日净值和仓位；
- `selection_actions.parquet`、`observation_actions.parquet`：买入、卖出和调仓建议；
- `LATEST_STATUS.csv`：数据截止日、新鲜度、最新策略动作和是否允许执行；
- `LATEST_ACTIONS.csv`：最新策略日需要执行的中文建议，无操作时只有表头；
- `LAST_REBALANCE_ACTIONS.csv`：上一批实际调仓记录，不能重复执行；
- `LATEST_TARGET_PORTFOLIO.csv`：当前目标板块、低风险仓位和权重；
- `POLICY.json`：当前策略参数；
- `RUN.json`：本次运行使用的模型、成本和可选开关。
- `REJECTED_EXPERIMENTS.json`：未通过开发期或选择期门槛的少量候选及拒绝原因。

真正的前向观察记录位于 `outputs/sector/forward/`：

- `PROTOCOL.json`：冻结日期、模型计划、策略参数和决策代码指纹；
- `SNAPSHOTS.csv`：按数据截止日追加的策略状态，不允许同日产生两个结果；
- `PERFORMANCE.json`：仅统计2026-08-10之后的未见数据表现；
- `STATUS.json`：当前协议是否仍匹配。

需要额外诊断时才显式开启：

```powershell
python -m ga_factor_mining.sector.rotation.product_backtest --cost-sensitivity
python -m ga_factor_mining.sector.rotation.product_backtest --boundary-sensitivity
python -m ga_factor_mining.sector.rotation.return_bridge
```

`return_bridge`只投影读取15列，用同一2086个收益日比较理论Top5、评分平滑和`simple_v1`。结果位于`outputs/sector/return_bridge/`，其中`RETURN_BRIDGE_SUMMARY.csv`是核心对照，`VALIDATION.json`核对日期和账本恒等式。

## 数据更新与研究重建

日常只使用 `python -m ga_factor_mining.sector --update`。下面的完整研究命令仅在修改因子或训练协议后使用：

```powershell
python -m ga_factor_mining.sector.rotation.run_experiments --with-lgbm
python -m ga_factor_mining.sector.rotation.rolling_validation
python -m ga_factor_mining.sector.rotation.adaptive_validation
python -m ga_factor_mining.sector
```

日常更新会同时更新板块行情、货币ETF行情与复权因子，并以流式方式替换缓存尾部；旧缓存与新计算的重叠区必须通过数值校验后才会替换正式文件。

## 研究口径

- 5日标签为 `open[t+6] / open[t+1] - 1`，不包含信号日无法成交的隔夜收益。
- 训练样本的标签必须在训练截止日前兑现。
- 2024-2025用于模型和策略选择，不能称为独立验证；2026只作观察说明，不用于选择参数。
- 横截面排名在实际行业和概念板块投资宇宙内计算。
- 产品账本包含持仓漂移、首次建仓、现金、单边换手和成本。
- 产品状态从2018年连续承接，不在2024选择期或2026观察期重新初始化。

## 项目边界

主流程只需要以下代码：

```text
src/ga_factor_mining/sector/rotation/run_experiments.py      特征与基础实验
src/ga_factor_mining/sector/rotation/rolling_validation.py   滚动LightGBM
src/ga_factor_mining/sector/rotation/strategy.py             买卖与持仓规则
src/ga_factor_mining/sector/rotation/product_backtest.py      产品账本和建议输出
```

GA因子、ETF映射、风险前沿和收益桥都是可选研究模块，不参与默认运行。当前真实行业ETF映射覆盖不足，正式回测仍使用板块指数收益；普通板块建议还不能直接视为ETF下单指令。

数据和Parquet缓存位于 `data/sector/` 与 `outputs/sector/`，均不提交Git。报告生成工具不放在研究代码中，最终报告只保留在 `reports/sector/`。

产品运行只投影读取15个必要字段，内存约174MB；不会先载入约1GB的完整特征面板，也不会在缓存过期时悄悄触发全量重建。

## 分支隔离

`sector_dev` 和 `stock_dev` 不直接合并。公共组件需要同步时，单独提交公共改动，再使用 `git cherry-pick` 带到另一分支。
