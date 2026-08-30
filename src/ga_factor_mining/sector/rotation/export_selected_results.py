#!/usr/bin/env python3
"""导出精选策略的结构化结果，不生成任何展示报告。"""

from __future__ import annotations

import pandas as pd

from .run_experiments import (
    OUT_DIR,
    OBSERVATION_END,
    OBSERVATION_START,
    VAL_END,
    VAL_START,
    StrategyConfig,
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
    index = pd.to_datetime(ret.index, format="%Y%m%d")
    return (1.0 + pd.Series(ret.values, index=index)).resample("ME").prod() - 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_features(force=False)
    df = train_lightgbm_scores(df, "industry_concept", target_h=5)
    df = train_lightgbm_scores(df, "industry_concept", target_h=10)

    score_names = sorted({strategy.score_name for strategy in SELECTED})
    caches = {
        "industry": prepare_universe_cache(df, "industry", score_names),
        "industry_concept": prepare_universe_cache(df, "industry_concept", score_names),
    }

    comparison = []
    monthly_rows = []
    for strategy in SELECTED:
        cache = caches[strategy.universe]
        periods = (("val", VAL_START, VAL_END), ("observation", OBSERVATION_START, OBSERVATION_END))
        for period, start, end in periods:
            ret, positions, aux = backtest_one_cached(cache, strategy, start, end)
            comparison.append({
                "strategy_id": strategy.strategy_id,
                "period": period,
                **annualized_metrics(ret),
                **aux,
            })
            for date, value in monthly_returns(ret).items():
                monthly_rows.append({
                    "strategy_id": strategy.strategy_id,
                    "period": period,
                    "month": date.strftime("%Y-%m"),
                    "return": float(value),
                })
            if strategy.strategy_id == "recommended_lgbm5_top5" and period == "observation":
                positions.to_parquet(OUT_DIR / "recommended_top5_observation_positions.parquet", index=False)
                (1.0 + ret.fillna(0.0)).cumprod().rename_axis("date").to_csv(
                    OUT_DIR / "recommended_top5_observation_equity_curve.csv",
                    header=["equity"],
                )

    pd.DataFrame(comparison).to_csv(
        OUT_DIR / "SELECTED_STRATEGY_COMPARISON.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(monthly_rows).to_csv(
        OUT_DIR / "SELECTED_MONTHLY_RETURNS.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[done] {OUT_DIR / 'SELECTED_STRATEGY_COMPARISON.csv'}")


if __name__ == "__main__":
    main()
