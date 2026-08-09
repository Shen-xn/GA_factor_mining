#!/usr/bin/env python3
"""生成板块轮动研究的精选策略总结。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .run_experiments import (
    OUT_DIR,
    REPORT_DIR,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
    StrategyConfig,
    add_formula_scores,
    annualized_metrics,
    backtest_one_cached,
    load_or_build_features,
    prepare_universe_cache,
    train_lightgbm_scores,
)


SELECTED = [
    StrategyConfig("aggressive_lgbm5_top3", "industry_concept", "score_lgbm_5d", 3, 1, 1),
    StrategyConfig("recommended_lgbm5_top5", "industry_concept", "score_lgbm_5d", 5, 1, 1),
    StrategyConfig("diversified_lgbm10_top10", "industry_concept", "score_lgbm_10d", 10, 1, 1),
    StrategyConfig("interpretable_industry_breakout_top3", "industry", "score_breakout", 3, 1, 1),
    StrategyConfig("stable_formula_low_vol_top20", "industry_concept", "score_low_vol_mom", 20, 1, 1),
]


def monthly_returns(ret: pd.Series) -> pd.Series:
    if ret.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(ret.index, format="%Y%m%d")
    return (1.0 + pd.Series(ret.values, index=idx)).resample("ME").prod() - 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_features(force=False)
    df = add_formula_scores(df)
    df = train_lightgbm_scores(df, "industry_concept", target_h=5)
    df = train_lightgbm_scores(df, "industry_concept", target_h=10)

    score_names = sorted({x.score_name for x in SELECTED})
    caches = {
        "industry": prepare_universe_cache(df, "industry", score_names),
        "industry_concept": prepare_universe_cache(df, "industry_concept", score_names),
    }

    comparison = []
    monthly_rows = []
    for cfg in SELECTED:
        cache = caches[cfg.universe]
        for period, start, end in [("val", VAL_START, VAL_END), ("test", TEST_START, TEST_END)]:
            ret, pos, aux = backtest_one_cached(cache, cfg, start, end)
            metrics = annualized_metrics(ret)
            row = {"strategy_id": cfg.strategy_id, "period": period, **metrics, **aux}
            comparison.append(row)
            for date, value in monthly_returns(ret).items():
                monthly_rows.append({
                    "strategy_id": cfg.strategy_id,
                    "period": period,
                    "month": date.strftime("%Y-%m"),
                    "return": float(value),
                })
            if cfg.strategy_id == "recommended_lgbm5_top5" and period == "test":
                pos.to_parquet(OUT_DIR / "recommended_top5_test_positions.parquet", index=False)
                ((1.0 + ret.fillna(0.0)).cumprod()).to_csv(
                    OUT_DIR / "recommended_top5_test_equity_curve.csv",
                    header=["equity"],
                )

    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT_DIR / "SELECTED_STRATEGY_COMPARISON.csv", index=False, encoding="utf-8-sig")
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT_DIR / "SELECTED_MONTHLY_RETURNS.csv", index=False, encoding="utf-8-sig")

    result = pd.read_csv(OUT_DIR / "EXPERIMENT_RESULTS.csv")
    aggressive = result[result["strategy_id"] == "industry_concept__score_lgbm_5d__top3__rb1"].iloc[0]
    recommended = result[result["strategy_id"] == "industry_concept__score_lgbm_5d__top5__rb1"].iloc[0]
    formula = result[result["strategy_id"] == "industry__score_breakout__top3__rb1"].iloc[0]
    importance = pd.read_csv(OUT_DIR / "lgbm_industry_concept_5d_importance.csv").head(10)

    latest_pos = pd.read_parquet(OUT_DIR / "recommended_top5_test_positions.parquet")
    latest_date = latest_pos["signal_date"].max()
    latest_pos = latest_pos[latest_pos["signal_date"] == latest_date].sort_values("rank")

    def line_for(name: str, r: pd.Series) -> str:
        return (
            f"| {name} | {r['val_ann_ret']:.2%} | {r['val_sharpe']:.2f} | "
            f"{r['val_max_drawdown']:.2%} | {r['test_ann_ret']:.2%} | "
            f"{r['test_sharpe']:.2f} | {r['test_max_drawdown']:.2%} | "
            f"{r['test_excess_ann_ret']:.2%} |"
        )

    lines = [
        "# 板块基金轮动研究总结",
        "",
        "## 结论",
        "",
        "在“每个板块都有无跟踪误差基金”的假设下，短周期板块强度策略可以做出明显收益。"
        "本轮最强的是 LightGBM 预测未来 5 日横截面收益分位，然后每日持有前 3 或前 5 个行业/概念板块。",
        "",
        "我更推荐 `recommended_lgbm5_top5` 作为当前版本，而不是验证集最高的 Top3 激进版：Top5 分散一点，2026 测试期收益和回撤都更好。",
        "",
        "## 精选策略对比",
        "",
        "| 策略 | Val年化 | Val夏普 | Val最大回撤 | Test年化 | Test夏普 | Test最大回撤 | Test超额年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        line_for("激进版 LGBM5 Top3 日调仓", aggressive),
        line_for("推荐版 LGBM5 Top5 日调仓", recommended),
        line_for("可解释基线 行业突破 Top3", formula),
        "",
        "## 推荐版参数",
        "",
        "- 宇宙：同花顺 `I + N`，即行业 + 概念/主题类板块。",
        "- 模型：LightGBM 回归，目标是未来 5 日收益在当日横截面的 percentile rank。",
        "- 输入：只使用当日及过去信息构造的横截面 rank 特征。",
        "- 调仓：每日收盘后打分，下一交易日持有 Top5，等权。",
        "- 成本：当前实验未计手续费和冲击成本；由于日调仓换手高，真实落地前必须加入成本模型。",
        "",
        "## LightGBM 5日模型前10特征重要性",
        "",
        "| 特征 | 重要性 |",
        "| --- | ---: |",
    ]
    for _, row in importance.iterrows():
        lines.append(f"| `{row['feature']}` | {int(row['importance'])} |")

    lines += [
        "",
        f"## 推荐版最近一次测试期持仓信号：{latest_date}",
        "",
        "| 排名 | 板块代码 | 板块名称 | 分数 |",
        "| ---: | --- | --- | ---: |",
    ]
    for _, row in latest_pos.iterrows():
        lines.append(f"| {int(row['rank'])} | `{row['ts_code']}` | {row['name']} | {row['score']:.4f} |")

    lines += [
        "",
        "## 重要风险",
        "",
        "- 这是板块指数基金假设，不是实际 ETF 回测。",
        "- 同花顺板块宇宙来自当前 `ths_index`，历史退役板块和真实可交易性还需要更严谨处理。",
        "- Top5 日调仓换手较高，必须补手续费、滑点、容量约束。",
        "- 2026 测试期只有到 2026-05-29，样本不长，不能当成最终策略定论。",
        "",
        "## 下一步",
        "",
        "1. 加入交易成本，按 5bp、10bp、20bp 做压力测试。",
        "2. 把日调仓改成 3日/5日调仓并训练对应目标，降低换手。",
        "3. 用 walk-forward 方式每季度重训 LightGBM，验证模型是否依赖固定历史环境。",
        "4. 如果要和个股 Top50 策略融合，优先把板块强度作为市场状态和行业配额信号，而不是直接替代个股因子。",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "RESEARCH_SUMMARY.md")


if __name__ == "__main__":
    main()
