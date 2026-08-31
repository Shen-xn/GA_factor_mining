# 使用手册

本文面向第一次拿到仓库的用户，目标是完成“安装、检查、回放、查看建议、更新数据”的闭环。

## 1. 安装

支持Python 3.11和3.12。建议始终使用独立虚拟环境。

Windows PowerShell：

```powershell
git clone https://github.com/Shen-xn/GA_factor_mining.git
cd GA_factor_mining
git switch main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

macOS或Linux：

```bash
git clone https://github.com/Shen-xn/GA_factor_mining.git
cd GA_factor_mining
git switch main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

安装完成后，下面的命令应能显示帮助：

```powershell
python -m ga_factor_mining.sector --help
```

## 2. 准备运行数据包

Git仓库不包含授权行情和大型Parquet缓存。需要从项目维护者处取得与当前 `main` 对应的运行数据包，并保持原有相对目录复制到仓库。

数据包包含两组内容：

- `data/sector/` 下6个原始Parquet；
- `outputs/sector/rotation/` 和 `outputs/sector/adaptation/` 下2个大型缓存Parquet。

完整文件名、用途和字段见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。JSON配置和缓存元数据已经随Git提交，不要用其他版本覆盖。

复制后运行：

```powershell
python -m ga_factor_mining.sector --check
```

检查分三层：

1. Python依赖是否齐全；
2. 运行数据包是否放在正确位置；
3. 原始数据、特征缓存和冻结评分的指纹是否一致。

返回码为0且最后显示 `[ready]` 才代表可以正式运行。检查失败不会重建缓存，也不会修改输出。

## 3. 正式回放

```powershell
python -m ga_factor_mining.sector
```

正常完成后，终端最后会显示 `outputs/sector/strategy/SUMMARY.csv`。优先查看：

1. `LATEST_STATUS.csv`：先确认数据、信号和指令状态；
2. `LATEST_PLAN.json`：核对最新收盘计划、下一交易日和阻断原因；
3. `outputs/sector/etf_mapping/ETF_EXECUTION_READINESS.json`：确认ETF执行层状态；
4. `LATEST_ACTIONS.csv`：只有全部安全门通过后才可能出现动作；
5. `LATEST_TARGET_PORTFOLIO.csv`：核对板块层完整目标权重；
6. `SUMMARY.csv` 和 `ANNUAL_RESULTS.csv`：查看历史表现。

默认命令会先在独立进程生成板块产品账本，再在另一个独立进程把冻结目标翻译到真实ETF开盘收益。这样可以降低Windows下原生数值库长时间运行造成的内存碎片风险。第二阶段完成不代表ETF可交易，仍必须以执行安全门和映射覆盖率为准。

`LATEST_ACTIONS.csv`为空不等于程序失败，可能是无需交易，也可能是数据、未来交易日历或ETF映射安全门阻止了执行。必须结合`LATEST_STATUS.csv`和`ETF_EXECUTION_READINESS.json`判断。上一批操作只保存在 `LAST_REBALANCE_ACTIONS.csv`，不能重复执行。

普通板块代码目前是研究建议。系统会把ETF解析结果单独写到`outputs/sector/etf_mapping/`，把真实ETF翻译回放写到`outputs/sector/etf_backtest/`；时点目录、映射覆盖、回放晋级或行情新鲜度任一不完整时，只生成`BLOCKED_ORDERS.csv`，不会伪装成可执行订单。

## 4. 增量更新

推荐使用环境变量，不把Token写入代码或仓库：

```powershell
$env:TUSHARE_TOKEN = "你的Token"
python -m ga_factor_mining.sector --check --update
python -m ga_factor_mining.sector --update
```

如果Token存放在仓库外文件：

```powershell
$env:TUSHARE_TOKEN_FILE = "D:\private\tushare_token.txt"
python -m ga_factor_mining.sector --check --update
python -m ga_factor_mining.sector --update
```

也可以临时传入：

```powershell
python -m ga_factor_mining.sector --update --token-file "D:\private\tushare_token.txt"
```

成本压力会把10/20/30/50bp拆到相互隔离的进程中，并复用经过周期、策略、成本、特征签名和低风险数据签名校验的小型结果缓存。最终正式账本和ETF回放也各自使用独立进程。如果本机默认Python存在原生运行时异常，可临时设置`GA_FACTOR_WORKER_PYTHON`指向另一个已验证、依赖齐全的Python解释器；数据、策略和验收口径不会因此改变。

更新会额外缓存请求日之后至少31个自然日的交易日历，用于生成严格晚于最新信号日的下一开市日。行情、特征、评分和日历通过尾部校验后才替换正式缓存；网络或数据校验失败时原文件保持不变。更新不会重新搜索模型参数或策略规则。

## 5. 运行测试

```powershell
python -m unittest discover -s tests -t . -v
```

测试不读取大型行情数据，适合在新机器上先确认代码安装正常。

## 6. 常见问题

### 显示“特征缓存不存在或已过期”

运行数据包缺失、混用了不同版本，或者原始行情已更新但特征缓存未同步。先重新执行 `--check`，按失败项替换成同一批次的数据包。默认产品入口不会自动触发高内存的全量重建。

### 显示“冻结评分与特征不一致”

`SELECTED_SCORES.parquet` 和 `sector_feature_panel.parquet` 不是同一协议生成。不要修改JSON绕过校验，应取得匹配的数据包或按研究流程重新建立模型协议。

### `LATEST_ACTIONS.csv`只有表头

先看 `LATEST_STATUS.csv`、`LATEST_PLAN.json`和`ETF_EXECUTION_READINESS.json`。可能是无需交易，也可能是数据超过7天、最新计划没有可信下一交易日、ETF映射过期或覆盖不足而被安全门禁止执行。

### 安装LightGBM失败

确认使用64位Python 3.11或3.12，并先升级pip：

```powershell
python -m pip install --upgrade pip
python -m pip install "lightgbm==4.6.0"
python -m pip install -e .
```

### 内存占用过高

日常只运行 `python -m ga_factor_mining.sector`。不要把 `run_experiments`、`rolling_validation` 或GA模块放入日常更新脚本；它们属于研究重建流程。

## 7. 版本边界

正式模型、策略规则或决策代码发生变化后，前向监控会拒绝把新结果续接到旧协议。维护者必须保留旧记录、建立新协议版本，再继续追加未来数据。
