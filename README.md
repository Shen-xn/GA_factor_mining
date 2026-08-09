# 简单遗传因子与滚动投票策略

本目录已迁移到 `C:\Users\s1171\Documents\ChatGPT\GA_factor_mining`，与原深度学习训练代码独立。基础数据位于当前目录的 `data/`，预处理缓存和策略输出位于 `outputs/`。

当前20个遗传因子的逐项公式、基础字段和峰值月份见 `FACTOR_LIBRARY_DETAILS.md`。

## 目标定义

每个股票交易日的监督标签固定为：

```text
future_ret_5d(t) = close_qfq(t+5) / close_qfq(t) - 1
```

因子 IC 是同一交易日全部股票的因子值与未来 5 日收益之间的 Spearman 相关系数。它衡量横向排序能力，不评价当日涨跌。

## 两阶段流程

1. 因子库挖掘仅使用 2026 年以前数据。遗传表达式由基础特征、四种一元变换和六种二元运算组成。
2. 适应度取表现最好的若干历史月份的绝对月均 IC 均值，因此允许因子只在部分市场阶段有效。
3. 搜索阶段固定抽取月份及每日股票子样本；候选因子再做横截面相关去重，最终保留 20 个。
4. 2026 年每 5 个交易日重新计算权重。权重只使用此前 30 个已经兑现标签的交易日 IC。
5. 因为标签需要未来 5 日才能知道，再平衡时会额外跳过最近 5 个交易日，防止隐性未来信息泄露。
6. 因子平均 IC 的正负决定投票方向，绝对值决定可靠度：`sign(IC) * abs(IC)^ic_power`，之后归一并限制单因子权重。
7. 固定权重用于接下来 5 个交易日，每天重新做股票横截面排序并加权投票。

## 遗传搜索细节

终端节点来自已有的连续特征，缺失状态标记不参与公式搜索。表达式支持：

```text
一元：neg、abs、signed_log、signed_sqrt
二元：add、sub、mul、safe_div、min、max
```

表达式最大深度为 3。适应度不是全历史平均 IC，而是抽样历史月份中绝对月均 IC 最高的 3 个月均值，并加入很轻的逐日波动与节点数惩罚：

```text
fitness = mean(top3(abs(monthly_mean_ic)))
          - 0.05 * std(daily_ic)
          - 0.0005 * expression_nodes
```

正式搜索使用 48 个历史月份、每个交易日固定抽取 700 只股票、80 个个体和 20 代。固定 BLAKE2 哈希保证跨进程抽样一致。每代 15% 随机移民用于减轻种群主题收敛。最终候选按横截面相关系数 `0.80` 去重，并限制同一搜索峰值月份最多保留 3 个。

搜索完成后，`validate_factor_library.py` 会在全部 2026 年前股票和交易日上重新计算月 IC。搜索抽样成绩与全量历史验收成绩会分别记录。

## 滚动可靠度

每个再平衡日使用此前 30 个可观测标签日。因为标签是未来 5 日收益，最近 5 个交易日尚未兑现，不能进入 IC 计算。

最终可靠度采用 5 日半衰期，但没有丢弃 30 日窗口中的旧数据：

```text
decay(age) = 0.5 ** (age / 5)
reliability_i = weighted_mean(IC_i, decay)
weight_i = reliability_i / sum(abs(reliability))
```

单因子绝对权重上限是 `0.25`。当前 20 个因子全部参与；每个因子将股票分为高于或低于当日中位数两票，IC 为负时自动反向投票。该二元投票比直接使用极端分位值更不容易被单个因子的尾部形状支配。

## 小决策树实验

`backtest_tree_2026.py` 是不使用 IC 构建决策模型的对照实验。每 5 个交易日：

1. 跳过最近 5 个尚未兑现标签的日期。
2. 取此前 30 个交易日作为一个训练月。
3. 20 个因子和标签都按日转换为横截面分位数。
4. 前 25 日拟合候选树，后 5 日按标签分位数 MAE 选择复杂度。
5. 用完整 30 日重新拟合，并预测接下来 5 日。

防过拟合约束为：最大深度候选 `1/2/3`、单叶至少包含训练样本的 `3%/5%/10%`、最多 8 个叶子、`ccp_alpha=1e-5` 剪枝。树训练和选择完全不使用 IC；最终仍报告 IC，以便与原投票方法比较。

运行命令：

```powershell
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe run_pipeline.py tree --config config.json
```

实验结果与逐棵树结构见 `outputs/TREE_EXPERIMENT_REPORT.md` 和 `tree_model_records_2026.json`。

## LightGBM 实验

LightGBM 使用与小树相同的无泄露滚动月数据，提供两种目标：

- `regression`：用 L2 损失预测未来 5 日收益的横截面分位数。
- `ranker`：按交易日分组，使用 LambdaRank 直接优化顶部约 10% 对应的 `NDCG@500`。

两者都只允许深度 2/3、4/7 个叶子，单叶至少 3000/5000 个样本，并使用 L1/L2 正则、行列采样和过去 5 日早停。Ranker 将未来收益分位数离散为 0–9 十级相关度。

```powershell
# 回归版
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe run_pipeline.py lightgbm --config config.json

# 横截面排序版
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe run_pipeline.py lightgbm_ranker --config config.json
```

完整对照见 `outputs/LIGHTGBM_EXPERIMENT_REPORT.md`。Ranker 对顶部组合有明显改善，但全横截面并不单调，不能把其负 Rank IC 简单解释为模型完全无效。

## 运行

先快速验收：

```powershell
cd C:\Users\s1171\Documents\ChatGPT\GA_factor_mining
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe build_factor_library.py --config config.json --quick
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe backtest_2026.py --config config.json
```

统一入口：

```powershell
D:\Users\s1171\qt_MLE\.conda\qt_mle\python.exe run_pipeline.py all --config config.json
```

`all` 会依次执行正式搜索、全量历史复核和 2026 回测。添加 `--quick` 时只做程序链路检查，并跳过昂贵的全量历史复核。

完整搜索时去掉 `--quick`。当前配置直接复用本项目的 `outputs/prepared_data.parquet`；如需从本地 `data/` 重建，可使用 `--force-data`。

## 输出

- `factor_library.json/csv`：20 个因子表达式及历史阶段性 IC
- `rolling_factor_weights_2026.csv`：每次再平衡的 IC、方向与权重
- `test_predictions_2026.parquet`：逐日逐股票投票分数和 5 日标签
- `daily_backtest_2026.csv`：每日 IC、头部/尾部及多空收益
- `backtest_summary_2026.json`：2026 汇总指标
- `factor_monthly_ic_full_history.csv`：20 个因子的全量历史逐月 IC
- `rolling_parameter_sweep_2026.csv`：可靠度与投票参数扫描结果
- `ACCEPTANCE_REPORT.md`：本次正式运行的验收报告
- `tree_models_2026.joblib`：19 个滚动小树模型
- `tree_model_records_2026.json`：树深度、叶数、验证误差和可读规则
- `TREE_EXPERIMENT_REPORT.md`：决策树与 IC 投票对照结论
- `lightgbm_models_2026.joblib`：LightGBM 回归版滚动模型
- `lightgbm_ranker_models_2026.joblib`：LambdaRank 滚动模型
- `LIGHTGBM_EXPERIMENT_REPORT.md`：两种 LightGBM 与旧策略对照

## 验收注意

2026 数据若被反复用来调参数，它在统计意义上会从测试集变成开发集。本实现允许按你的要求观察并调整 2026 结果，但最终上线前应另留后续时间作为真正未见样本。回测收益为 5 日重叠标签的研究统计，不等同于每日可累加收益。

当前分数使用当日收盘后可计算的特征，评价的是从该收盘到未来第 5 个收盘的横截面排序能力。若转成可交易策略，需要另行定义下一交易日开盘成交、涨跌停过滤、停牌处理和交易成本，不能直接把这里的重叠 5 日收益逐日累加。
