#!/usr/bin/env python3
"""在冻结选择期比较GA表示加入人工特征模型后的产品增量。"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd

from ...common.paths import ensure_output_dir
from ..factor_mining.run_factor_mining import load_config, load_discovery_frame, make_evaluator
from .low_risk import DEFAULT_LOW_RISK_CODE, build_low_risk_return_frame, low_risk_data_signature
from .product_backtest import (
    PRODUCT_HISTORY_START,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
)
from .rolling_validation import FEATURE_COLS, fit_predict_lgbm, fit_predict_lgbm_periodic
from .run_experiments import (
    FEATURE_PROTOCOL_VERSION,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_feature_subset,
)
from .strategy import get_strategy_policy


def passes_incremental_gate(baseline: dict, candidate: dict) -> bool:
    """增量必须改善收益和夏普，且不能明显放大回撤或换手。"""
    return (
        candidate["ann_ret"] > baseline["ann_ret"]
        and candidate["sharpe"] >= baseline["sharpe"] + 0.05
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02
        and candidate["avg_turnover"] <= baseline["avg_turnover"] + 0.01
    )


def _metric_row(period: str, model: str, factor_name: str, metrics: dict) -> dict:
    return {"period": period, "model": model, "factor_name": factor_name, **metrics}


def main() -> None:
    factor_dir = Path("outputs/sector/factor_mining")
    validation = pd.read_csv(factor_dir / "factor_full_validation.csv", encoding="utf-8-sig")
    shadows = validation[validation["status"].eq("shadow")].copy()
    if shadows.empty:
        raise RuntimeError("没有通过结构门和单因子验证门的 GA shadow 候选")

    selected_adaptation = json.loads(
        Path("outputs/sector/adaptation/SELECTED.json").read_text(encoding="utf-8")
    )
    if selected_adaptation.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise RuntimeError("自适应评分缓存与当前特征协议不一致")

    panel_columns = set(product_feature_columns("score_breakout", external_score=True))
    panel_columns.update(FEATURE_COLS)
    panel_columns.update({"future_ret_5d_rank", "future_ret_5d_end_date"})
    panel = load_feature_subset(panel_columns)
    low_risk_frame = build_low_risk_return_frame(panel)
    if selected_adaptation.get("feature_cache_signature") != current_feature_cache_signature():
        raise RuntimeError("自适应评分缓存与当前特征数据不一致")
    policy = get_strategy_policy("simple_v1")
    baseline_scores = pd.read_parquet("outputs/sector/adaptation/SELECTED_SCORES.parquet")
    baseline_score_col = next(
        column for column in baseline_scores.columns if column not in {"ts_code", "trade_date"}
    )
    baseline_panel = panel.merge(baseline_scores, on=["ts_code", "trade_date"], how="left")
    baseline_daily, baseline_actions, _ = run_product_backtest(
        baseline_panel,
        baseline_score_col,
        PRODUCT_HISTORY_START,
        VAL_END,
        cost_bps=20.0,
        strategy_policy=policy,
        low_risk_frame=low_risk_frame,
    )
    _, _, baseline_selection = summarize_backtest_period(
        baseline_daily, baseline_actions, VAL_START, VAL_END
    )
    rows = [_metric_row("selection", "manual_baseline", "", baseline_selection)]
    del baseline_panel
    gc.collect()

    cfg = load_config("configs/sector/factor_mining.json")
    ga_frame, _ = load_discovery_frame(cfg)
    evaluator = make_evaluator(ga_frame, cfg)
    candidate_predictions: dict[str, pd.DataFrame] = {}
    candidate_metrics: dict[str, dict] = {}

    for shadow in shadows.itertuples(index=False):
        expression = json.loads(shadow.expression_json)
        value = evaluator.eval(expression) * int(shadow.direction)
        feature_name = f"ga_{shadow.factor_name}_rank"
        ga_feature = ga_frame[["ts_code", "trade_date"]].copy()
        ga_feature[feature_name] = value.groupby(ga_frame["trade_date"]).rank(pct=True).astype("float32")
        candidate_panel = panel.merge(ga_feature, on=["ts_code", "trade_date"], how="left")
        score_col = f"score_plus_{shadow.factor_name}"
        if int(selected_adaptation["retrain_months"]) == 12:
            predictions = fit_predict_lgbm(
                candidate_panel,
                5,
                list(range(int(PRODUCT_HISTORY_START[:4]), int(VAL_END[:4]) + 1)),
                training_years=selected_adaptation["training_years"],
                recency_half_life_years=selected_adaptation["half_life_years"],
                score_col=score_col,
                feature_cols=FEATURE_COLS + [feature_name],
            )
        else:
            predictions = fit_predict_lgbm_periodic(
                candidate_panel,
                5,
                PRODUCT_HISTORY_START,
                VAL_END,
                retrain_months=int(selected_adaptation["retrain_months"]),
                training_years=selected_adaptation["training_years"],
                recency_half_life_years=selected_adaptation["half_life_years"],
                score_col=score_col,
                feature_cols=FEATURE_COLS + [feature_name],
            )
        candidate_predictions[shadow.factor_name] = predictions
        scored = candidate_panel.merge(predictions, on=["ts_code", "trade_date"], how="left")
        candidate_daily, candidate_actions, _ = run_product_backtest(
            scored,
            score_col,
            PRODUCT_HISTORY_START,
            VAL_END,
            cost_bps=20.0,
            strategy_policy=policy,
            low_risk_frame=low_risk_frame,
        )
        _, _, metrics = summarize_backtest_period(
            candidate_daily, candidate_actions, VAL_START, VAL_END
        )
        candidate_metrics[shadow.factor_name] = metrics
        rows.append(_metric_row("selection", "manual_plus_ga", shadow.factor_name, metrics))
        del candidate_panel, scored, ga_feature
        gc.collect()

    passing = [
        name for name, metrics in candidate_metrics.items()
        if passes_incremental_gate(baseline_selection, metrics)
    ]
    selected_factor = max(
        passing,
        key=lambda name: candidate_metrics[name]["sharpe"],
        default=None,
    )

    output_dir = ensure_output_dir("sector", "ga_ablation")
    if selected_factor is not None:
        predictions = candidate_predictions[selected_factor]
        predictions.to_parquet(output_dir / "SELECTED_SCORES.parquet", index=False)

    pd.DataFrame(rows).to_csv(output_dir / "RESULTS.csv", index=False, encoding="utf-8-sig")
    (output_dir / "DECISION.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "selected_factor_for_future_protocol": selected_factor,
                "passing_factors": passing,
                "selection_period": "2024-2025 model/factor selection",
                "continuous_product_history_start": PRODUCT_HISTORY_START,
                "product_policy_name": "simple_v1",
                "baseline_model_spec": {
                    "variant": selected_adaptation["variant"],
                    "training_years": selected_adaptation["training_years"],
                    "half_life_years": selected_adaptation["half_life_years"],
                    "retrain_months": selected_adaptation["retrain_months"],
                },
                "observation_used_for_selection": False,
                "observation_evaluated": False,
                "independent_out_of_sample_available": False,
                "current_frozen_product_changed": False,
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
    print(f"[done] selected_factor={selected_factor}")


if __name__ == "__main__":
    main()
