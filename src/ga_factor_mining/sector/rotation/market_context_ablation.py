#!/usr/bin/env python3
"""验证市场宽度与波动状态是否为板块模型提供稳定增量。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ...common.paths import ensure_output_dir
from .ga_ablation import passes_incremental_gate
from .low_risk import DEFAULT_LOW_RISK_CODE, build_low_risk_return_frame, low_risk_data_signature
from .product_backtest import run_product_backtest
from .rolling_validation import FEATURE_COLS, fit_predict_lgbm, fit_predict_lgbm_periodic
from .run_experiments import (
    MARKET_CONTEXT_COLS,
    FEATURE_PROTOCOL_VERSION,
    OBSERVATION_END,
    OBSERVATION_START,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_or_build_features,
)
from .strategy import StrategyPolicy


def main() -> None:
    panel = load_or_build_features()
    low_risk_frame = build_low_risk_return_frame(panel)
    adaptation_meta = json.loads(
        Path("outputs/sector/adaptation/SELECTED.json").read_text(encoding="utf-8")
    )
    if adaptation_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise RuntimeError("自适应评分缓存与当前特征协议不一致")
    if adaptation_meta.get("feature_cache_signature") != current_feature_cache_signature():
        raise RuntimeError("自适应评分缓存与当前特征数据不一致")
    missing = [column for column in MARKET_CONTEXT_COLS if column not in panel.columns]
    if missing:
        raise RuntimeError(f"特征面板缺少市场上下文: {missing}")
    policy = StrategyPolicy()

    baseline_scores = pd.read_parquet("outputs/sector/adaptation/SELECTED_SCORES.parquet")
    baseline_score = next(
        column for column in baseline_scores.columns if column not in {"ts_code", "trade_date"}
    )
    baseline_panel = panel.merge(baseline_scores, on=["ts_code", "trade_date"], how="left")
    _, _, baseline_validation = run_product_backtest(
        baseline_panel,
        baseline_score,
        VAL_START,
        VAL_END,
        cost_bps=20.0,
        strategy_policy=policy,
        low_risk_frame=low_risk_frame,
    )

    context_score = "score_manual_plus_market_context"
    if int(adaptation_meta["retrain_months"]) == 12:
        context_predictions = fit_predict_lgbm(
            panel,
            5,
            [2024, 2025, 2026],
            training_years=adaptation_meta["training_years"],
            recency_half_life_years=adaptation_meta["half_life_years"],
            score_col=context_score,
            feature_cols=FEATURE_COLS + MARKET_CONTEXT_COLS,
        )
    else:
        context_predictions = fit_predict_lgbm_periodic(
            panel,
            5,
            VAL_START,
            OBSERVATION_END,
            retrain_months=int(adaptation_meta["retrain_months"]),
            training_years=adaptation_meta["training_years"],
            recency_half_life_years=adaptation_meta["half_life_years"],
            score_col=context_score,
            feature_cols=FEATURE_COLS + MARKET_CONTEXT_COLS,
        )
    context_panel = panel.merge(context_predictions, on=["ts_code", "trade_date"], how="left")
    _, _, context_validation = run_product_backtest(
        context_panel,
        context_score,
        VAL_START,
        VAL_END,
        cost_bps=20.0,
        strategy_policy=policy,
        low_risk_frame=low_risk_frame,
    )
    passed = passes_incremental_gate(baseline_validation, context_validation)

    _, _, baseline_observation = run_product_backtest(
        baseline_panel,
        baseline_score,
        OBSERVATION_START,
        OBSERVATION_END,
        cost_bps=20.0,
        strategy_policy=policy,
        low_risk_frame=low_risk_frame,
    )
    rows = [
        {"period": "validation", "model": "manual_baseline", **baseline_validation},
        {"period": "validation", "model": "manual_plus_market_context", **context_validation},
        {"period": "observation", "model": "manual_baseline", **baseline_observation},
    ]
    if passed:
        _, _, context_observation = run_product_backtest(
            context_panel,
            context_score,
            OBSERVATION_START,
            OBSERVATION_END,
            cost_bps=20.0,
            strategy_policy=policy,
            low_risk_frame=low_risk_frame,
        )
        rows.append(
            {"period": "observation", "model": "manual_plus_market_context", **context_observation}
        )

    output_dir = ensure_output_dir("sector", "market_context_ablation")
    pd.DataFrame(rows).to_csv(output_dir / "RESULTS.csv", index=False, encoding="utf-8-sig")
    if passed:
        context_predictions.to_parquet(output_dir / "SELECTED_SCORES.parquet", index=False)
    (output_dir / "DECISION.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "selected": bool(passed),
                "feature_group": MARKET_CONTEXT_COLS,
                "selection_period": "2024-2025 validation",
                "baseline_model_spec": {
                    "variant": adaptation_meta["variant"],
                    "training_years": adaptation_meta["training_years"],
                    "half_life_years": adaptation_meta["half_life_years"],
                    "retrain_months": adaptation_meta["retrain_months"],
                },
                "observation_used_for_selection": False,
                "low_risk_code": DEFAULT_LOW_RISK_CODE,
                "low_risk_data_signature": low_risk_data_signature(),
                "gate": {
                    "annual_return": "greater_than_baseline",
                    "sharpe_improvement_min": 0.05,
                    "max_drawdown_worsening_max": 0.02,
                    "daily_turnover_worsening_max": 0.01,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] selected={passed}")


if __name__ == "__main__":
    main()
