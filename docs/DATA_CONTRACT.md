# 板块策略数据契约

原始行情受数据授权约束，大型缓存也不适合进入Git。因此代码仓库与运行数据包分开分发。

## 最小运行数据包

把下列文件放在仓库对应的相对路径中：

| 文件 | 必要字段 | 用途 |
|---|---|---|
| `data/sector/ths_index.parquet` | `ts_code,name,count,exchange,list_date,type` | 板块名称与投资宇宙 |
| `data/sector/ths_daily.parquet` | `ts_code,trade_date,open,high,low,close,pre_close,avg_price,vol,turnover_rate` | 板块行情与缓存指纹 |
| `data/sector/low_risk_fund_basic.parquet` | `ts_code,name,fund_type,list_date,delist_date` | 冻结选择货币ETF |
| `data/sector/low_risk_fund_daily.parquet` | `ts_code,trade_date,open,close,amount` | 低风险腿真实收益 |
| `data/sector/low_risk_fund_adj.parquet` | `ts_code,trade_date,adj_factor` | 低风险腿复权 |
| `data/sector/low_risk_fund_nav.parquet` | `ts_code,nav_date,adj_nav` | 货币ETF价格/净值核验 |
| `outputs/sector/rotation/sector_feature_panel.parquet` | 见下文 | 冻结特征缓存 |
| `outputs/sector/adaptation/SELECTED_SCORES.parquet` | `ts_code,trade_date,score_frozen_selected_5d` | 正式季度LightGBM评分 |

要生成下一交易日计划并解析ETF，还需要执行层数据：

| 文件 | 必要字段 | 用途 |
|---|---|---|
| `data/sector/trade_calendar.parquet` | `cal_date,is_open` | 从可信交易所日历取得下一开市日 |
| `data/sector/etf_basic.parquet` | `ts_code,index_code,index_name,list_date,list_status,etf_type` | 权益ETF候选目录 |
| `data/sector/equity_etf_daily.parquet` | `ts_code,trade_date,open,close,amount` | 映射验收与ETF行情新鲜度 |
| `data/sector/equity_etf_adj.parquet` | `ts_code,trade_date,adj_factor` | 权益ETF复权 |

缺少这四个文件不妨碍历史研究回放，但执行层必须标记`blocked`，不得猜下一自然日或沿用过期ETF映射。

下面两个小型元数据文件已经提交Git，必须与上述Parquet来自同一版本：

- `outputs/sector/rotation/sector_feature_panel.meta.json`；
- `outputs/sector/adaptation/SELECTED.json`。

不要把另一批数据的JSON或Parquet混入当前 `main`。

## 字段规则

- `trade_date`、`list_date`、`delist_date`和`nav_date`统一使用 `YYYYMMDD` 字符串；
- `ts_code`必须稳定且在同一文件中含义唯一；
- 价格、成交量、换手率和复权因子必须可转换为数值；
- 同一 `ts_code + trade_date` 在日行情和复权因子中必须唯一；
- 缺失开盘价不能填成前值或0；
- 回测形成信号时不能查看未来开盘价是否存在；实际成交/估值日遇到缺报价时，账本分别记录未成交或价格持平估值计数；
- 原始数据更新后必须同步更新特征和评分缓存，不能手工修改指纹。

正式特征缓存至少应包含：

```text
ts_code, trade_date, type, open, close,
forward_open_ret_1d, next_open_date, return_end_date,
ret_1d, ret_5d, ret_20d, ret_60d,
volatility_20d, ret_5d_rank, volatility_20d_rank,
18个人工模型特征及5个市场环境特征
```

完整列清单记录在 `sector_feature_panel.meta.json` 的 `feature_columns` 中。

## 一致性检查

```powershell
python -m ga_factor_mining.sector --check
```

检查会使用文件大小和SHA-256内容哈希确认原始行情与特征缓存一致，不依赖用户名、绝对目录或文件修改时间。因此，同一数据包可以复制到另一台机器或不同仓库目录。

检查通过不代表行情足够新。正式运行还会比较数据截止日与运行日；超过7天时只允许历史复现，不生成可执行建议。

## 更新凭据

Tushare Token只允许通过以下方式提供：

1. 环境变量 `TUSHARE_TOKEN`；
2. 环境变量 `TUSHARE_TOKEN_FILE` 指向仓库外文件；
3. 命令参数 `--token-file`；
4. 本地仓库根目录的 `tushare_token.txt`。

`tushare_token.txt` 和 `.env*` 已加入 `.gitignore`。任何Token都不得写入代码、配置、输出或文档。

## 维护者发布数据包

发布前应满足：

1. 在正式代码版本上完成增量更新；
2. 执行 `python -m ga_factor_mining.sector --check`；
3. 执行一次默认回放并核对 `LATEST_STATUS.csv`；
4. 历史回放包分发前8个Parquet；需要最新计划与ETF参考解析时，再包含执行层4个Parquet；
5. 同时告知接收者对应的Git提交号。

没有运行数据包时，用户仍可以安装项目、查看报告和运行全部单元测试，但不能复现正式策略净值。这是数据授权边界，不应通过把原始行情提交到Git来绕过。
