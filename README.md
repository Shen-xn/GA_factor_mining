# GA Factor Mining：板块轮动原型

`main` 是当前可交付基线，只包含板块研究线。它提供一个从本地行情、冻结LightGBM评分、低频持仓规则，到净值、仓位和中文操作建议的完整闭环。

当前版本定位是“可复现的研究原型”，不是券商交易系统：普通板块仍按板块指数收益回测，真实权益ETF映射覆盖不足，只有低风险腿使用货币ETF `511880.SH`。任何输出都应先人工复核，不能直接视为实盘委托。

## 五分钟开始

推荐环境：Windows 10/11，Python 3.11或3.12，至少8GB内存。

核心科学计算依赖已经锁定为项目验证过的版本，避免安装时自动升级到未经大缓存回放验证的新组合。

```powershell
git clone https://github.com/Shen-xn/GA_factor_mining.git
cd GA_factor_mining
git switch main

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

原始数据和Parquet缓存不进入Git。首次运行前，需要把配套运行数据包复制到仓库，具体文件和字段见 [数据契约](docs/DATA_CONTRACT.md)。复制完成后执行：

```powershell
python -m ga_factor_mining.sector --check
python -m ga_factor_mining.sector
```

`--check`只检查依赖、文件和数据指纹，不训练模型，也不改写策略结果。全部显示`[OK]`后，默认命令才会执行正式回放。

更完整的安装、更新和故障处理见 [使用手册](docs/GETTING_STARTED.md)。

## 默认运行做什么

默认入口固定使用：

- 扩展窗口LightGBM，自2024年起每季度重新训练；
- 冻结的18个人工特征和固定模型参数；
- `simple_v1`：Top5入场、跌出Top10退出、至少持有5个交易日；
- `t`日收盘形成计划，`t+1`日开盘成交，下一开盘出现后再结算收益；
- 未配置资金进入货币ETF `511880.SH`；
- 单边20bp交易成本；
- 2018年起唯一连续状态路径，不在2024或2026重置持仓。

默认运行不会自动执行GA挖掘、参数搜索、消融实验或报告生成，也不会在缓存失效时悄悄载入全量数据重建。产品账本和真实ETF回放在两个独立进程中串行完成，避免同一Python进程反复装载Pandas/NumPy大对象；产品流程只投影读取必要列，当前环境峰值内存约174MB。

## 用户看哪些结果

正式输出在 `outputs/sector/strategy/`：

- `LATEST_STATUS.csv`：数据日期、是否过期、市场状态和当前仓位；
- `LATEST_PLAN.json`、`LATEST_PLAN_ACTIONS.csv`：最新收盘形成的计划及其阻断原因；
- `LATEST_ACTIONS.csv`：只有全部执行安全门通过后才出现的中文动作；
- `LAST_REBALANCE_ACTIONS.csv`：上一批调仓记录，不能重复执行；
- `LATEST_TARGET_PORTFOLIO.csv`：当前目标板块与低风险权重；
- `LATEST_MARKET_RISK.csv/.json`：板块广度风险分、原始/确认状态、市场仓位与回撤上限；
- `SUMMARY.csv`：开发期、选择期和观察期汇总；
- `ANNUAL_RESULTS.csv`：2018年至今的逐年结果；
- `COST_SENSITIVITY.csv`：10/20/30/50bp完整重放压力结果；
- `TARGET_WEIGHT_TIMELINE.csv`：每个计划/执行日的完整板块与低风险目标权重；
- `ACCEPTANCE_GATE.csv/.json`：收益、回撤、换手和成本门是否通过；
- `HISTORY_DAILY.parquet`、`HISTORY_ACTIONS.parquet`：连续产品账本；
- `POLICY.json`、`RUN.json`：本次使用的规则、数据和运行环境。

ETF执行层在 `outputs/sector/etf_mapping/`：`ETF_EXECUTION_READINESS.json`汇总数据、日历、映射和回放安全门，`LATEST_MAPPING_RESOLUTION.csv`逐项解释板块如何映射或回退，`RESOLVED_ETF_TARGET_PORTFOLIO.csv`保证最终ETF权重守恒。被阻止的参考组合只会写入`BLOCKED_ORDERS.csv`，不会进入`LATEST_ACTIONS.csv`。

`outputs/sector/etf_backtest/`保存冻结板块目标到真实ETF复权开盘价的翻译回放。它不改变LightGBM排名、持有规则或原策略目标，只负责暴露ETF映射覆盖和实施损耗。当前20bp全期回放为`+16.79%`、最大回撤`-1.42%`，但历史平均映射覆盖仅`0.85%`，收益主要来自未映射权重回退`511880.SH`，因此明确标记为不可晋级，不能解释为可交易的板块策略。

本轮把本地权益ETF历史从143只扩展到273只，并修复了长区间复权因子可能被接口行数上限截断的问题；严格映射覆盖仍未改善。随后只用2018—2025检验22个ETF原生、行业子宇宙、人工评分和行业专用LightGBM候选，没有一个同时通过收益、回撤、年度稳定性和换手门，默认`simple_v1`保持不变。结构化结论见`outputs/sector/etf_native_research/`。

当前研究结论保存在：

- `reports/sector/rotation/SECTOR_ROTATION_REPORT.md`；
- `reports/sector/factor_mining/FACTOR_MINING_REPORT.md`；
- `outputs/sector/` 下的结构化研究记录。

研究产物可以保留，但日常用户只需要默认入口和 `outputs/sector/strategy/`。

这里的风险分来自同花顺行业与概念板块的趋势、上涨宽度和波动率，只能称为“板块广度健康分”，不是严格的宽基大盘指数。分数越高，越适合承担权益风险；它目前只增强信号解释，不改变已冻结的`simple_v1`交易逻辑。`LATEST_TARGET_PORTFOLIO.csv`中的连续风险调整强度等于模型横截面相对强度乘以板块广度健康分，它不是收益率预测；实际执行预算仍看风险目标仓位。数据过期时一律禁止执行。

## 更新到最新行情

Tushare Token推荐只放在当前终端环境变量中：

```powershell
$env:TUSHARE_TOKEN = "你的Token"
python -m ga_factor_mining.sector --check --update
python -m ga_factor_mining.sector --update
```

也可以设置 `TUSHARE_TOKEN_FILE`，或使用 `--token-file` 指向仓库外的文本文件。Token文件已被Git忽略，不要提交凭据。

增量更新只下载新增交易日、缓存请求日之后至少31个自然日的上交所交易日历、替换缓存尾部并延长冻结模型评分，不运行GA或参数搜索。行情距运行日超过7天时，策略仍可完成历史复现和最新收盘计划，但会清空执行动作并标记“禁止执行”。

## 验证代码

测试使用Python标准库，不要求额外安装pytest：

```powershell
python -m unittest discover -s tests -t . -v
```

当前基线通过104项测试，覆盖无未来信息标签、滚动训练边界、`planned → executed_unsettled → settled`时间轴、交易日历、缺失行情、持仓状态、ETF分年增量更新与头部缺口修复、映射安全门、真实ETF回放、成本、风险评分、前向协议和运行前自检。长回放中的标量风险计算和持仓遍历已改为更轻量的等价实现，降低Windows原生运行时的内存压力，不改变策略结果。

## 当前结果口径

截至本地数据的2026-08-28：

- 2018—2023开发期累计约 `+13.5%`，最大回撤约 `-18.9%`；
- 2024—2025选择期累计约 `+36.2%`，最大回撤约 `-10.0%`；
- 2026观察期累计约 `+0.7%`，最大回撤约 `-7.5%`。

2024—2025参与过模型频率和规则选择，2026也已经被研究者观察，二者都不是新的独立样本外证据。前向v8已封存2026-08-11至2026-08-28共14个真正未见交易日；v9至v12都只有同日冻结基线、没有未见数据，已经原样归档；当前v13仍以2026-08-28为冻结日，只有之后新到的数据才进入前向证据。历史协议保存在`outputs/sector/forward/archive/`。

旧项目中“多数年份大幅正收益”的年度LightGBM Top5原型已经重新运行并精确复现，但其收盘标签、次日收益和训练边界与可执行的次日开盘口径不一致，且没有交易成本。统一到当前协议后收益明显下降。随后唯一验证的“领先板块自身走强时恢复70%仓位”候选恶化了2018、2022和长期回撤，因此没有晋级，也没有建立`simple_v2`。完整证据见板块轮动报告和`outputs/sector/prototype_recovery/`、`outputs/sector/return_bridge/`。

本轮在不打开2026的前提下继续验证了健康分固定分档、Top20保留区和8/10日最低持有。按当前修正后的统一引擎，Top20保留区的2018—2025累计收益为`+66.8%`、最大回撤`-17.1%`，但年化换手仍为`21.4`倍、持有中位仍为5日。Top20加10日最低持有在20bp下达到累计`+73.2%`、年化`7.39%`、最大回撤`-16.75%`、年化换手`15.0`倍和持有中位10日，是目前最接近目标的候选；但30bp完整路径下开发期和全期收益转负，因此成本鲁棒性门失败，没有晋级。复核证明失效并非额外10bp费用本身，而是费用使净值更早触发回撤上限，从2018年末开始改变后续仓位路径，说明候选对净值路径不稳健。

独立市场信息复核也已完成。五个宽基指数、全A等权宽度和总市值加权宽度都没有改善封存的2018—2025基准；申万一级行业的20/60日中期价格评分虽把年化换手降到约7.3—7.6倍，却得到约-3%的年化收益和超过31%的最大回撤。资金流数据历史不足，未进入选择。正式版本因此仍为`simple_v1`。当前基准2018—2025累计约`+54.5%`、年化约`5.8%`，平均日换手约`8.9%`，尚未通过收益、低频和成本压力门；结构化证据见`outputs/sector/market_information_research/`，`ACCEPTANCE_GATE.json`会明确记录失败项。

当前数据截止2026-08-28，最新收盘计划对应2026-08-31开盘，目标为30%板块、70%低风险资产。交易日历、ETF目录、权益ETF行情和月度映射均已更新；但三个目标板块的严格ETF映射覆盖为0，执行层仍为`blocked`，`LATEST_ACTIONS.csv`保持为空。四个事后登记的语义代理假设在2025年末历史闸门中全部失败，也没有进入默认映射。该计划只能用于核对状态，不能补单。

## 分支与目录

- `main`：当前板块策略可交付基线；
- `sector_dev`：后续板块开发；
- `stock_dev`：独立个股研究，不进入当前主线。

```text
configs/                  冻结配置和因子注册表
data/sector/              本地原始数据，不提交Git
src/ga_factor_mining/     工程代码
outputs/sector/strategy/  正式产品输出
outputs/sector/           其他结构化研究产物
reports/sector/           面向人的最终报告
tests/                    可移植性和策略测试
docs/                     用户与数据文档
```

开发者和研究口径见 [板块模块说明](src/ga_factor_mining/sector/README.md)。
