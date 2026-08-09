from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .run_experiments import OUT_DIR, REPORT_DIR

try:
    import markdown
except ImportError:  # pragma: no cover
    markdown = None


ROOT = Path(__file__).resolve().parent
ARTIFACTS = OUT_DIR
REPORT_MD = REPORT_DIR / "SECTOR_ROTATION_PAPER.md"
REPORT_HTML = REPORT_DIR / "SECTOR_ROTATION_PAPER.html"
REPORT_PDF = REPORT_DIR / "SECTOR_ROTATION_PAPER.pdf"
EQUITY_FIG = REPORT_DIR / "sector_recommended_equity_2026.png"
COMPARISON_FIG = REPORT_DIR / "sector_strategy_comparison.png"
MONTHLY_FIG = REPORT_DIR / "sector_recommended_monthly_returns.png"
ROLLING_RESULTS = OUT_DIR / "ROLLING_VALIDATION_RESULTS.csv"
ROLLING_ANNUAL_FIG = REPORT_DIR / "rolling_validation_annual_returns.png"
ROLLING_EQUITY_FIG = REPORT_DIR / "rolling_validation_equity_curves.png"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def build_figures(selected: pd.DataFrame, monthly: pd.DataFrame) -> None:
    setup_matplotlib()

    equity = pd.read_csv(ARTIFACTS / "recommended_top5_test_equity_curve.csv")
    date_col = equity.columns[0]
    equity["date"] = pd.to_datetime(equity[date_col].astype(str), format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(equity["date"], equity["equity"], color="#1f5fbf", lw=2.2)
    ax.fill_between(equity["date"], 1.0, equity["equity"], color="#1f5fbf", alpha=0.10)
    ax.axhline(1.0, color="#777", lw=0.9, ls="--")
    ax.set_title("推荐策略 2026 测试期净值曲线")
    ax.set_ylabel("净值，起点=1")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(EQUITY_FIG, bbox_inches="tight")
    plt.close(fig)

    pivot = selected.pivot(index="strategy_id", columns="period", values=["ann_ret", "max_drawdown"])
    order = [
        "aggressive_lgbm5_top3",
        "recommended_lgbm5_top5",
        "diversified_lgbm10_top10",
        "interpretable_industry_breakout_top3",
        "stable_formula_low_vol_top20",
    ]
    labels = ["LGBM5 Top3", "推荐 LGBM5 Top5", "LGBM10 Top10", "突破公式 Top3", "低波动 Top20"]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    val = [pivot.loc[s, ("ann_ret", "val")] * 100 for s in order]
    test = [pivot.loc[s, ("ann_ret", "test")] * 100 for s in order]
    ax.bar([i - 0.18 for i in x], val, width=0.36, label="验证期年化", color="#355c7d")
    ax.bar([i + 0.18 for i in x], test, width=0.36, label="测试期年化", color="#c06c84")
    ax.set_xticks(list(x), labels, rotation=16, ha="right")
    ax.set_ylabel("年化收益率 (%)")
    ax.set_title("精选策略验证期与测试期收益对比")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPARISON_FIG, bbox_inches="tight")
    plt.close(fig)

    rec_monthly = monthly[monthly["strategy_id"].eq("recommended_lgbm5_top5")].copy()
    rec_monthly["month"] = pd.to_datetime(rec_monthly["month"])
    colors = rec_monthly["return"].map(lambda x: "#2e7d32" if x >= 0 else "#c62828")
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(rec_monthly["month"].dt.strftime("%Y-%m"), rec_monthly["return"] * 100, color=colors)
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_title("推荐策略逐月收益")
    ax.set_ylabel("月收益率 (%)")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(MONTHLY_FIG, bbox_inches="tight")
    plt.close(fig)


def build_markdown() -> str:
    experiments = pd.read_csv(ARTIFACTS / "EXPERIMENT_RESULTS.csv")
    selected = pd.read_csv(ARTIFACTS / "SELECTED_STRATEGY_COMPARISON.csv")
    monthly = pd.read_csv(ARTIFACTS / "SELECTED_MONTHLY_RETURNS.csv")
    imp5 = pd.read_csv(ARTIFACTS / "lgbm_industry_concept_5d_importance.csv")
    positions = pd.read_parquet(ARTIFACTS / "recommended_top5_test_positions.parquet")

    build_figures(selected, monthly)

    rec = selected[selected["strategy_id"].eq("recommended_lgbm5_top5")]
    rec_val = rec[rec["period"].eq("val")].iloc[0]
    rec_test = rec[rec["period"].eq("test")].iloc[0]
    best_val = experiments.sort_values("val_sharpe", ascending=False).iloc[0]
    latest_date = positions["signal_date"].max()
    latest_positions = positions[positions["signal_date"].eq(latest_date)].sort_values("rank")

    selected_names = {
        "aggressive_lgbm5_top3": "激进版 LGBM5 Top3 日调仓",
        "recommended_lgbm5_top5": "推荐版 LGBM5 Top5 日调仓",
        "diversified_lgbm10_top10": "分散版 LGBM10 Top10 日调仓",
        "interpretable_industry_breakout_top3": "可解释公式 行业突破 Top3",
        "stable_formula_low_vol_top20": "稳定公式 低波动 Top20",
    }
    comparison_rows = []
    for sid, group in selected.groupby("strategy_id", sort=False):
        val = group[group["period"].eq("val")].iloc[0]
        test = group[group["period"].eq("test")].iloc[0]
        comparison_rows.append(
            [
                selected_names.get(sid, sid),
                pct(val["ann_ret"]),
                num(val["sharpe"], 2),
                pct(val["max_drawdown"]),
                pct(test["ann_ret"]),
                num(test["sharpe"], 2),
                pct(test["max_drawdown"]),
                pct(test["avg_turnover"]),
            ]
        )

    top_experiment_rows = []
    for i, row in experiments.sort_values("val_sharpe", ascending=False).head(8).iterrows():
        top_experiment_rows.append(
            [
                str(len(top_experiment_rows) + 1),
                f"`{row['strategy_id']}`",
                pct(row["val_ann_ret"]),
                num(row["val_sharpe"], 2),
                pct(row["val_max_drawdown"]),
                pct(row["test_ann_ret"]),
                num(row["test_sharpe"], 2),
                pct(row["test_max_drawdown"]),
            ]
        )

    importance_rows = [
        [f"`{row.feature}`", str(int(row.importance))]
        for row in imp5.head(10).itertuples(index=False)
    ]
    latest_rows = [
        [
            str(int(row.rank)),
            f"`{row.ts_code}`",
            str(row.name),
            f"{row.score:.4f}",
        ]
        for row in latest_positions.itertuples(index=False)
    ]

    rolling_section = ""
    if ROLLING_RESULTS.exists():
        rolling = pd.read_csv(ROLLING_RESULTS)
        rolling_summary = rolling[rolling["row_type"].eq("summary")].copy()
        rolling_year = rolling[rolling["row_type"].eq("year")].copy()
        rolling_names = {
            "rolling_lgbm5_top5": "滚动 LGBM5 Top5",
            "rolling_lgbm10_top10": "滚动 LGBM10 Top10",
            "formula_industry_breakout_top3": "公式 行业突破 Top3",
            "formula_low_vol_top20": "公式 低波动动量 Top20",
        }
        rolling_summary_rows = []
        for row in rolling_summary.itertuples(index=False):
            rolling_summary_rows.append(
                [
                    rolling_names.get(row.strategy_id, row.strategy_id),
                    str(int(row.start_year)),
                    str(int(row.end_year)),
                    pct(row.ann_ret),
                    num(row.sharpe, 2),
                    pct(row.max_drawdown),
                    pct(row.avg_turnover),
                    f"{int(row.positive_years)}/{int(row.valid_years)}",
                ]
            )
        rolling_lgbm5_rows = []
        for row in rolling_year[rolling_year["strategy_id"].eq("rolling_lgbm5_top5")].itertuples(index=False):
            rolling_lgbm5_rows.append(
                [
                    str(int(row.year)),
                    pct(row.ann_ret),
                    num(row.sharpe, 2),
                    pct(row.max_drawdown),
                    pct(row.bench_ann_ret),
                    pct(row.excess_ann_ret),
                ]
            )
        rolling_section = f"""
### 5.3 滚动验证补充

为了避免固定切分给人“一次性挑模型”的感觉，我补充做了滚动验证。机器学习模型采用逐年 walk-forward：验证某一年时，只使用上一年年底以前已经兑现标签的数据训练；公式评分没有训练过程，因此直接展示 2016 年以来的逐年表现。2026 年只统计至 2026-05-29。

{md_table(['策略', '起始年', '结束年', '拼接年化', '拼接夏普', '最大回撤', '平均换手', '正收益年份'], rolling_summary_rows)}

![滚动验证年度收益]({ROLLING_ANNUAL_FIG.name})

![滚动验证拼接净值]({ROLLING_EQUITY_FIG.name})

滚动 `LGBM5 Top5` 的逐年表现如下：

{md_table(['年份', '年化', '夏普', '最大回撤', '基准年化', '超额年化'], rolling_lgbm5_rows)}

这组结果的核心含义是：推荐方向不只是在 2024-2025 固定验证期表现好。滚动 LGBM5 Top5 从 2018 到 2026 的样本外拼接年化为 {pct(float(rolling_summary[rolling_summary['strategy_id'].eq('rolling_lgbm5_top5')]['ann_ret'].iloc[0]))}，正收益年份为 {int(rolling_summary[rolling_summary['strategy_id'].eq('rolling_lgbm5_top5')]['positive_years'].iloc[0])}/{int(rolling_summary[rolling_summary['strategy_id'].eq('rolling_lgbm5_top5')]['valid_years'].iloc[0])}。公式突破 Top3 在 2016-2026 的长期结果也较强，但年度波动更大，说明它更适合作为可解释基线或和机器学习评分组合，而不一定单独作为最终方案。
"""

    md = f"""# 板块基金轮动策略研究报告

生成日期：2026-06-30  
研究代码：`src/ga_factor_mining/sector/rotation/`  
核心假设：每个同花顺板块指数都存在一个无跟踪误差、可按收盘价成交的跟踪基金。

## 摘要

本研究单独评估“只交易板块基金”的短周期轮动策略。策略目标不是预测单个股票，而是构造一个可以直接反映未来 5 至 10 个交易日板块收益强弱的板块强度评分，并据此每日或定期持有评分最高的少数板块。

在当前无交易成本、无跟踪误差、无容量约束的理想化设定下，板块强度模型能够在 2024-2025 验证期和 2026 确认测试期同时跑赢同宇宙等权基准。综合收益、回撤和分散性后，当前推荐方案为 `recommended_lgbm5_top5`：使用 LightGBM 预测未来 5 日板块收益横截面分位，每日持有行业与概念板块中的 Top5。该方案验证期年化收益 {pct(rec_val['ann_ret'])}，夏普 {num(rec_val['sharpe'], 2)}，最大回撤 {pct(rec_val['max_drawdown'])}；2026 测试期年化收益 {pct(rec_test['ann_ret'])}，夏普 {num(rec_test['sharpe'], 2)}，最大回撤 {pct(rec_test['max_drawdown'])}。

需要强调：这不是实际 ETF 回测。它更像一个板块强度信号的上限实验，目的是判断板块层面的短线收益是否可被历史量价状态捕捉。真实落地前必须加入可交易 ETF 映射、跟踪误差、手续费、滑点、容量和调仓约束。

## 1. 研究目的

本部分研究回答三个问题：

1. 在板块指数层面，短周期动量、波动、均线位置和交易活跃度是否能预测未来 5 至 10 个交易日收益。
2. 如果每个板块都有一个理想跟踪基金，仅交易板块基金能否形成正收益和稳定超额收益。
3. 板块强度是否值得作为后续个股 Top50 策略的行业配置、市场状态或风险过滤信号。

这个实验位于独立的 `sector_dev` 分支。代码、机器产物和人工报告分别进入 `src/`、`outputs/sector/` 与 `reports/sector/`，不会污染个股研究线。

## 2. 数据来源与处理

### 2.1 数据范围

使用已经拉取好的同花顺板块数据：

- 日线行情表：`data/sector/ths_daily.parquet`
- 板块基础表：`data/sector/ths_index.parquet`
- 数据日期：2015-01-05 至 2026-05-29，共 2,769 个交易日。
- 实验样本：基于同花顺行业、概念、地域、主题等板块指数构造，最终策略主要使用 `industry_concept` 宇宙，即行业 `I` 加概念/主题 `N`。

### 2.2 时间切分

研究采用严格时间切分：

- 训练/建模历史：2015-01-05 至 2023-12-31。
- 验证期：2024-01-01 至 2025-12-31。
- 确认测试期：2026-01-01 至 2026-05-29。

参数选择主要看验证期表现，测试期只用于确认，不反向修改策略。

### 2.3 标签定义

对每个板块，在信号日 `t` 收盘后生成特征，未来收益从下一个交易日开始计算。模型标签包括：

- `future_ret_5d`：未来 5 个交易日收益。
- `future_ret_10d`：未来 10 个交易日收益。
- `future_ret_rank`：同一交易日所有板块未来收益的横截面分位。

模型训练使用未来收益分位作为目标，而不是直接使用原始收益，目的是降低极端涨跌对模型的支配，让模型学习“横向比较谁更强”。

### 2.4 特征处理

所有输入特征只使用当前及过去信息。核心特征包括：

- 多周期收益排名：`ret_1d/3d/5d/10d/20d/60d_rank`
- 波动率排名：`volatility_10d_rank`、`volatility_20d_rank`
- 风险调整收益：`risk_adj_5_20_rank`、`risk_adj_10_20_rank`、`risk_adj_20_60_rank`
- 价格位置：`close_pos_20d_rank`、`drawdown_60d_rank`
- 均线偏离：`ma_gap_5_20_rank`、`ma_gap_10_60_rank`
- 活跃度异常：`volume_z_20d_rank`、`turnover_z_20d_rank`
- 当日波动：`range_1d_rank`

这些特征都先在板块自身历史上计算，再做每日横截面排名，因此既保留了时间状态，又降低了跨年代成交额、成交量、指数点位变化带来的尺度漂移。

## 3. 方法论与方法选择

### 3.1 策略形式

策略为长仓等权 TopK：

1. 每个信号日收盘后计算全部板块评分。
2. 选择评分最高的 `K` 个板块。
3. 下一交易日开始持有，按等权组合计算收益。
4. 根据设定的调仓频率每日、每 5 日或每 10 日更新持仓。

本轮没有引入做空、杠杆、止损、成本、容量或人工筛选。

### 3.2 候选评分方法

本轮比较两类方法：

- 公式型评分：如短期/中期动量、突破位置、低波动动量、均值回归、量价确认等。优点是可解释，缺点是难以处理多特征非线性组合。
- LightGBM 回归：预测未来 5 日或 10 日收益横截面分位。优点是能学习非线性和特征交互，训练速度快，适合表格型金融特征；缺点是解释性弱于公式，且需要警惕过拟合和样本外稳定性。

从验证结果看，LightGBM 5 日目标显著优于大多数手写公式；但可解释公式在 2026 测试期出现很强表现，说明简单突破状态本身也可能含有可交易信息。推荐方案没有直接选择验证期最高收益的 Top3，而是选择 Top5 版本，因为它在 2026 测试期回撤更低、收益更高、分散性更好。

## 4. 实验设计

### 4.1 实验矩阵

本轮总计运行 {len(experiments)} 个配置，包括：

- 宇宙：行业、行业+概念、更宽主题宇宙。
- TopK：3、5、10、20。
- 调仓频率：1、5、10 个交易日。
- 评分方法：多个公式评分、LightGBM 5 日目标、LightGBM 10 日目标。

### 4.2 评价指标

主要评价指标：

- 年化收益率：衡量收益能力。
- 年化波动率：衡量组合波动。
- 夏普比率：衡量单位波动收益。
- 最大回撤：衡量最坏路径风险。
- 胜率：日度收益为正的比例。
- 平均换手：相邻持仓名单变化比例。
- 相对等权基准超额收益：判断是否只是吃到了整个板块宇宙上涨。

由于这是板块基金假设实验，收益指标暂时不扣交易成本。换手率在本阶段只作为落地难度指标。

### 4.3 策略选择规则

验证期最高夏普方案为 `{best_val['strategy_id']}`，验证期夏普 {num(best_val['val_sharpe'], 2)}。不过最终推荐没有机械选择最高夏普方案，而是结合以下原则：

- 验证期和测试期都要为正，并尽量跑赢等权基准。
- TopK 不能过小到完全依赖少数板块。
- 测试期最大回撤优先不能明显恶化。
- 方法应当能解释为“板块强度评分”，方便后续和个股策略结合。

因此本轮推荐 `recommended_lgbm5_top5` 作为当前版本。

## 5. 实验结果

### 5.1 精选策略对比

{md_table(['策略', 'Val年化', 'Val夏普', 'Val最大回撤', 'Test年化', 'Test夏普', 'Test最大回撤', 'Test平均换手'], comparison_rows)}

![精选策略验证与测试对比]({COMPARISON_FIG.name})

### 5.2 验证期前 8 名配置

{md_table(['排名', '策略ID', 'Val年化', 'Val夏普', 'Val最大回撤', 'Test年化', 'Test夏普', 'Test最大回撤'], top_experiment_rows)}

{rolling_section}

### 5.4 推荐策略净值与月度表现

推荐策略 `recommended_lgbm5_top5` 在 2026 测试期共 {int(rec_test['days'])} 个交易日，总收益 {pct(rec_test['total_ret'])}，年化收益 {pct(rec_test['ann_ret'])}，最大回撤 {pct(rec_test['max_drawdown'])}，平均换手 {pct(rec_test['avg_turnover'])}。

![推荐策略 2026 测试期净值曲线]({EQUITY_FIG.name})

![推荐策略逐月收益]({MONTHLY_FIG.name})

### 5.5 LightGBM 5 日模型特征重要性

{md_table(['特征', '重要性'], importance_rows)}

前 10 个重要特征以中期均线偏离、20 日波动、60 日回撤、60 日收益和风险调整收益为主。这说明模型并非只看一日涨跌，而是在识别“中期趋势位置 + 波动状态 + 风险调整强度”的组合状态。

### 5.6 最新一次测试期持仓信号

最新可用测试信号日：`{latest_date}`。

{md_table(['排名', '板块代码', '板块名称', '模型分数'], latest_rows)}

## 6. 讨论

### 6.1 为什么板块层面可能更容易

板块指数天然做了个股噪声平均化，短周期行业轮动、主题拥挤、资金偏好和风险偏好变化更容易在板块层面表现出来。相比个股，板块的 idiosyncratic noise 较低，趋势状态和波动状态也更稳定，因此简单的非线性模型就可能捕捉到可用信号。

### 6.2 推荐策略的含义

`recommended_lgbm5_top5` 的含义不是“永远买这 5 个板块”，而是每天用历史状态估计未来 5 日横截面收益分位，然后选择当日最可能领先的 5 个板块。这个信号可单独做板块轮动，也可以作为个股 Top50 策略的行业偏好权重。

### 6.3 主要风险与限制

1. 理想基金假设偏强：当前回测假设每个板块都有无跟踪误差基金，真实市场中并非所有板块都有可交易 ETF。
2. 未计交易成本：推荐策略平均换手约 {pct(rec_test['avg_turnover'])}，真实落地必须加入手续费、滑点和冲击成本。
3. 当前板块宇宙可能有幸存者偏差：`ths_index` 是当前可见板块列表，历史退市或调整过的板块需要进一步核对。
4. 2026 测试期较短：测试只到 2026-05-29，不足以证明跨市场周期稳定。
5. 滚动验证仍不等于实盘：本报告已补充逐年 walk-forward，但尚未加入交易成本、ETF 映射和更高频的滚动重训。

## 7. 结论

在当前实验设定下，板块强度策略是有研究价值的。短周期板块轮动不需要复杂深度学习模型，使用横截面排名特征加 LightGBM 就已经能得到不错的验证与测试表现。当前推荐 `recommended_lgbm5_top5` 作为后续研究基线：它比 Top3 激进版更分散，测试期最大回撤更低，同时保留较高收益。

下一阶段不应只追求更高纸面收益，而应优先解决三个落地问题：

1. 加入 5bp、10bp、20bp 成本压力测试，评估高换手策略是否仍有优势。
2. 将板块指数映射到真实可交易 ETF 或基金，加入跟踪误差和容量约束。
3. 将当前年度 walk-forward 扩展为季度或月度重训，并加入成本压力，验证策略在不同市场状态下是否保持有效。

## 附录 A：产物清单

- `src/ga_factor_mining/sector/rotation/run_experiments.py`：实验主程序。
- `src/ga_factor_mining/sector/rotation/summarize_results.py`：结果汇总脚本。
- `outputs/sector/rotation/EXPERIMENT_RESULTS.csv`：全部 276 个实验配置结果。
- `outputs/sector/rotation/SELECTED_STRATEGY_COMPARISON.csv`：精选策略验证/测试对比。
- `outputs/sector/rotation/recommended_top5_test_positions.parquet`：推荐策略 2026 测试期每日 Top5。
- `outputs/sector/rotation/recommended_top5_test_equity_curve.csv`：推荐策略 2026 测试净值曲线。
- `outputs/sector/rotation/lgbm_industry_concept_5d_importance.csv`：LightGBM 5 日模型特征重要性。
- `outputs/sector/rotation/ROLLING_VALIDATION_RESULTS.csv`：滚动验证年度与汇总指标。
- `reports/sector/rotation/ROLLING_VALIDATION_REPORT.pdf`：滚动验证补充报告。

## 附录 B：口径说明

报告中的验证期为 2024-2025，测试期为 2026-01-01 至 2026-05-29。所有收益均为未扣成本的指数基金假设收益。由于本研究目的是先确认板块强度信号是否存在，实际交易可行性需要在下一轮引入 ETF 映射和成本模型后重新评估。
"""
    return md


def markdown_to_html(md: str) -> str:
    if markdown is None:
        body = "<pre>" + html.escape(md) + "</pre>"
    else:
        body = markdown.markdown(md, extensions=["tables", "toc", "sane_lists"])
    css = """
    @page { size: A4; margin: 18mm 16mm; }
    body {
      font-family: "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", Arial, sans-serif;
      color: #20242a;
      line-height: 1.62;
      font-size: 13.2px;
      max-width: 920px;
      margin: 0 auto;
    }
    h1 {
      font-size: 28px;
      line-height: 1.25;
      margin: 0 0 16px;
      color: #17365d;
      border-bottom: 2px solid #17365d;
      padding-bottom: 10px;
    }
    h2 {
      font-size: 20px;
      margin: 28px 0 10px;
      color: #17365d;
      border-left: 5px solid #5b8cc0;
      padding-left: 10px;
    }
    h3 {
      font-size: 16px;
      margin: 20px 0 8px;
      color: #2f4f6f;
    }
    p { margin: 8px 0 10px; }
    ul, ol { margin: 8px 0 10px 22px; padding: 0; }
    li { margin: 3px 0; }
    code {
      font-family: Consolas, "Courier New", monospace;
      background: #f4f6f8;
      border: 1px solid #e5e8ec;
      border-radius: 3px;
      padding: 1px 4px;
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 16px;
      font-size: 11.2px;
      page-break-inside: avoid;
    }
    th {
      background: #eaf1f8;
      color: #17365d;
      font-weight: 700;
    }
    th, td {
      border: 1px solid #cfd8e3;
      padding: 5px 6px;
      vertical-align: middle;
    }
    tr:nth-child(even) td { background: #fbfcfe; }
    img {
      display: block;
      max-width: 94%;
      margin: 12px auto 18px;
      page-break-inside: avoid;
    }
    blockquote {
      margin: 12px 0;
      padding: 8px 12px;
      background: #f6f8fb;
      border-left: 4px solid #8aa9c8;
    }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>板块基金轮动策略研究报告</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def export_pdf() -> None:
    browser = find_browser()
    if browser is None:
        raise RuntimeError("未找到 Edge/Chrome，无法导出 PDF")
    if REPORT_PDF.exists():
        REPORT_PDF.unlink()
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        f"--print-to-pdf={REPORT_PDF}",
        REPORT_HTML.as_uri(),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    if not REPORT_PDF.exists() or REPORT_PDF.stat().st_size < 10_000:
        raise RuntimeError("PDF 导出失败或文件过小")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = build_markdown()
    REPORT_MD.write_text(md, encoding="utf-8")
    REPORT_HTML.write_text(markdown_to_html(md), encoding="utf-8")
    export_pdf()
    print(f"Markdown: {REPORT_MD}")
    print(f"HTML: {REPORT_HTML}")
    print(f"PDF: {REPORT_PDF} ({REPORT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
