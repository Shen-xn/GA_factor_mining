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
- `t`日收盘形成信号，`t+1`日开盘成交；
- 未配置资金进入货币ETF `511880.SH`；
- 单边20bp交易成本；
- 2018年起唯一连续状态路径，不在2024或2026重置持仓。

默认运行不会自动执行GA挖掘、参数搜索、消融实验或报告生成，也不会在缓存失效时悄悄载入全量数据重建。产品流程只投影读取必要列，当前环境峰值内存约174MB。

## 用户看哪些结果

正式输出在 `outputs/sector/strategy/`：

- `LATEST_STATUS.csv`：数据日期、是否过期、市场状态和当前仓位；
- `LATEST_ACTIONS.csv`：本次需要执行的中文动作，无动作时只有表头；
- `LAST_REBALANCE_ACTIONS.csv`：上一批调仓记录，不能重复执行；
- `LATEST_TARGET_PORTFOLIO.csv`：当前目标板块与低风险权重；
- `SUMMARY.csv`：开发期、选择期和观察期汇总；
- `ANNUAL_RESULTS.csv`：2018年至今的逐年结果；
- `HISTORY_DAILY.parquet`、`HISTORY_ACTIONS.parquet`：连续产品账本；
- `POLICY.json`、`RUN.json`：本次使用的规则、数据和运行环境。

当前研究结论保存在：

- `reports/sector/rotation/SECTOR_ROTATION_REPORT.md`；
- `reports/sector/factor_mining/FACTOR_MINING_REPORT.md`；
- `outputs/sector/` 下的结构化研究记录。

研究产物可以保留，但日常用户只需要默认入口和 `outputs/sector/strategy/`。

## 更新到最新行情

Tushare Token推荐只放在当前终端环境变量中：

```powershell
$env:TUSHARE_TOKEN = "你的Token"
python -m ga_factor_mining.sector --check --update
python -m ga_factor_mining.sector --update
```

也可以设置 `TUSHARE_TOKEN_FILE`，或使用 `--token-file` 指向仓库外的文本文件。Token文件已被Git忽略，不要提交凭据。

增量更新只下载新增交易日、替换缓存尾部并延长冻结模型评分，不运行GA或参数搜索。行情距运行日超过7天时，策略仍可完成历史复现，但会清空最新执行建议并标记“禁止执行”。

## 验证代码

测试使用Python标准库，不要求额外安装pytest：

```powershell
python -m unittest discover -s tests -t . -v
```

当前基线通过83项测试，覆盖无未来信息标签、滚动训练边界、缺失行情处理、持仓状态、成本、低风险资产、前向协议、运行前自检和领先板块强度判定。

## 当前结果口径

截至本地数据的2026-08-10：

- 2018—2023开发期累计约 `+13.5%`，最大回撤约 `-18.9%`；
- 2024—2025选择期累计约 `+36.6%`，最大回撤约 `-10.0%`；
- 2026观察期累计约 `+0.2%`，最大回撤约 `-7.5%`。

2024—2025参与过模型频率和规则选择，2026也已经被研究者观察，二者都不是新的独立样本外证据。只有冻结日2026-08-10之后从未见过的数据，才会进入 `outputs/sector/forward/` 的前向记录。

旧项目中“多数年份大幅正收益”的年度LightGBM Top5原型已经重新运行并精确复现，但其收盘标签、次日收益和训练边界与可执行的次日开盘口径不一致，且没有交易成本。统一到当前协议后收益明显下降。随后唯一验证的“领先板块自身走强时恢复70%仓位”候选恶化了2018、2022和长期回撤，因此没有晋级，也没有建立`simple_v2`。完整证据见板块轮动报告和`outputs/sector/prototype_recovery/`、`outputs/sector/return_bridge/`。

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
