# Top-50 因子策略 V2

本目录是独立实验区，不修改V1代码、因子库或历史报告，只复用
`outputs/stock/v1/prepared_data.parquet`。

## 时间协议

| 用途 | 日期 | 是否允许选参数 |
|---|---|---|
| 因子发现 | 2015-01-01～2023-12-31 | 只搜索因子表达式 |
| 参数验证 | 2024-01-01～2025-12-31 | 是，所有实验选择只看这里 |
| 确认测试 | 2026-01-01～当前有效标签日 | 否，只允许冻结后运行一次 |

预测目标固定为未来5个交易日收益。每天先计算横截面收益分位，再按
`0/1/2/3/4`五级相关度训练。模型选出Top-50，本轮不处理最终实盘5只的人工筛选。

## 因子发现

遗传搜索每天按BLAKE2固定抽取700只股票，但时间算子始终先在连续完整历史上计算。
正式模式会把搜索产生的最多600个候选重新放回2015～2023完整股票横截面复核，
然后按完整数据适应度、峰值月份上限和0.80相关阈值贪心选出20个因子。

时间终端包括：`delta_5`、`slope_5`、`mean_5`、`std_20`、`zscore_20`、
`ts_rank_20`。时间终端只能直接作用于基础字段，不能递归套在复合表达式上。

正式遗传配置为80个体、20代、12精英、12随机移民、55%交叉、35%变异、
最大深度3。G01/G02启用5个体锦标赛、对称子树交叉和混合变异；G02还平衡
四类初始个体。最终因子不强制类别均衡。

## 滚动训练与防泄露

每个预测块开始前，训练数据末端额外剔除5个尚未兑现标签的交易日。候选模型为：

- M00：IC衰减投票
- M01：10箱或20箱条件期望
- M02：LightGBM回归
- M03：LightGBM Top-50二分类
- M04：LightGBM LambdaRank@50

训练窗口比较30/60/120日，更新频率比较每日/每5日，时间衰减比较无衰减、
10日半衰期和20日半衰期。最后额外运行价格技术、资金流、估值规模类别消融。

## 运行顺序

使用项目专用环境：

```powershell
cd C:\Users\s1171\Documents\ChatGPT\GA_factor_mining
$env:PYTHONPATH = "$PWD\src"
python -m ga_factor_mining.stock.top50.run_experiments --stage factor
python -m ga_factor_mining.stock.top50.run_experiments --stage operators
python -m ga_factor_mining.stock.top50.run_experiments --stage ga
python -m ga_factor_mining.stock.top50.run_experiments --stage models
python -m ga_factor_mining.stock.top50.run_experiments --stage windows
python -m ga_factor_mining.stock.top50.run_experiments --stage final-test
```

`--quick`只用于因子、算子、遗传和模型阶段的程序冒烟检查。程序明确禁止快速模式
冻结配置或运行2026测试。

## 选择规则

候选方案首先必须同时满足：2024和2025的Top-50平均超额收益均大于0、月度胜率
不低于55%、NDCG@50高于随机基线；模型阶段还必须高于M00旧IC基线。合格方案
依次按月度超额收益中位数、NDCG@50和低换手选择。无人达标时会明确记录
`eligible_pool=false`，并保留诊断用的相对最佳项，但它不代表通过验收门槛。

## 产物

机器产物统一写入 `outputs/stock/top50/`，人工报告写入 `reports/stock/top50/`。每个实验目录包含配置快照、配置SHA-256、随机种子、
运行时间、逐日指标、逐月指标、全股票预测、Top-50名单和最近一个滚动模型。

正式链路最终生成：

- `EXPERIMENT_MATRIX.csv`
- `VALIDATION_REPORT.md`
- `FINAL_STRATEGY_CONFIG.json`
- `TEST_2026_REPORT.md`
- `VALIDATION_TEST_GAP.md`
- `top50_predictions.parquet`

`final-test`只读取冻结文件，运行期间会校验文件未被修改。2026结果不会触发重新选参。
