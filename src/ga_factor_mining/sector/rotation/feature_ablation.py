#!/usr/bin/env python3
"""在当前季度模型与simple_v1口径下逐项消融明确派生因子。"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile

import pandas as pd

from ...common.paths import ensure_output_dir
from .low_risk import build_low_risk_return_frame
from .product_backtest import (
    PRODUCT_HISTORY_START,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
)
from .rolling_validation import FEATURE_COLS, fit_predict_lgbm_periodic
from .run_experiments import (
    FEATURE_PROTOCOL_VERSION,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_feature_subset,
)
from .strategy import get_strategy_policy


SELECTED_META = Path("outputs/sector/adaptation/SELECTED.json")
SELECTED_SCORES = Path("outputs/sector/adaptation/SELECTED_SCORES.parquet")
FORMAL_SUMMARY = Path("outputs/sector/strategy/SUMMARY.csv")
DEVELOPMENT_START = "20180101"
DEVELOPMENT_END = "20231231"
SCORE_COLUMN = "score_feature_ablation_5d"

CANDIDATES = {
    "drop_risk_adj_5_20": "risk_adj_5_20_rank",
    "drop_risk_adj_10_20": "risk_adj_10_20_rank",
    "drop_risk_adj_20_60": "risk_adj_20_60_rank",
}

DEVELOPMENT_GATE = {
    "total_return_loss_max": 0.02,
    "sharpe_loss_max": 0.10,
    "max_drawdown_worsening_max": 0.02,
    "turnover_worsening_max": 0.01,
    "positive_years_not_lower": True,
    "bad_year_worsening_max": 0.02,
}

SELECTION_GATE = {
    "total_return_loss_max": 0.02,
    "sharpe_loss_max": 0.05,
    "max_drawdown_worsening_max": 0.02,
    "turnover_worsening_max": 0.01,
    "each_year_positive": True,
}


def _annual_returns(daily: pd.DataFrame, start: str, end: str) -> dict[int, float]:
    selected = daily[daily["date"].between(start, end)].copy()
    selected["year"] = selected["date"].str[:4].astype(int)
    return {
        int(year): float((1.0 + frame["net_return"]).prod() - 1.0)
        for year, frame in selected.groupby("year")
    }


def passes_development_gate(
    baseline: dict,
    candidate: dict,
    baseline_years: dict[int, float],
    candidate_years: dict[int, float],
) -> bool:
    """简化可以小幅牺牲收益，但不能减少正收益年份或放大坏年份。"""
    return (
        candidate["total_ret"] >= baseline["total_ret"] - 0.02
        and candidate["sharpe"] >= baseline["sharpe"] - 0.10
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02
        and candidate["avg_turnover"] <= baseline["avg_turnover"] + 0.01
        and sum(value > 0 for value in candidate_years.values())
        >= sum(value > 0 for value in baseline_years.values())
        and candidate_years.get(2018, -1.0) >= baseline_years.get(2018, -1.0) - 0.02
        and candidate_years.get(2022, -1.0) >= baseline_years.get(2022, -1.0) - 0.02
    )


def passes_selection_gate(
    baseline: dict,
    candidate: dict,
    candidate_years: dict[int, float],
) -> bool:
    return (
        candidate["total_ret"] >= baseline["total_ret"] - 0.02
        and candidate["sharpe"] >= baseline["sharpe"] - 0.05
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02
        and candidate["avg_turnover"] <= baseline["avg_turnover"] + 0.01
        and candidate_years.get(2024, -1.0) > 0.0
        and candidate_years.get(2025, -1.0) > 0.0
    )


def _evaluate(
    product_panel: pd.DataFrame,
    scores: pd.DataFrame,
    score_column: str,
    low_risk_frame: pd.DataFrame,
    end: str,
    period_start: str,
    period_end: str,
) -> tuple[dict, dict[int, float]]:
    scored = product_panel.merge(
        scores[["ts_code", "trade_date", score_column]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    daily, actions, _ = run_product_backtest(
        scored,
        score_column,
        PRODUCT_HISTORY_START,
        end,
        cost_bps=20.0,
        strategy_policy=get_strategy_policy("simple_v1"),
        low_risk_frame=low_risk_frame,
    )
    _, _, metrics = summarize_backtest_period(daily, actions, period_start, period_end)
    annual = _annual_returns(daily, period_start, period_end)
    del scored, daily, actions
    gc.collect()
    return metrics, annual


def _result_row(
    variant: str,
    dropped_feature: str | None,
    metrics: dict,
    annual: dict[int, float],
    passed: bool,
) -> dict:
    return {
        "variant": variant,
        "dropped_feature": dropped_feature,
        "feature_count": len(FEATURE_COLS) - int(dropped_feature is not None),
        "passed_gate": passed,
        **metrics,
        **{f"return_{year}": value for year, value in sorted(annual.items())},
    }


def _assert_frozen_baseline(period: str, metrics: dict) -> None:
    """候选与正式账本的基线必须逐口径一致，否则禁止解释消融。"""
    formal = pd.read_csv(FORMAL_SUMMARY).set_index("period").loc[period]
    for key in ("total_ret", "max_drawdown", "avg_turnover"):
        if abs(float(metrics[key]) - float(formal[key])) > 1e-12:
            raise RuntimeError(
                f"{period} 基线未对齐正式账本: {key} "
                f"{metrics[key]} != {formal[key]}"
            )


def main() -> None:
    output_dir = ensure_output_dir("sector", "feature_ablation")
    selected_meta = json.loads(SELECTED_META.read_text(encoding="utf-8"))
    feature_signature = current_feature_cache_signature()
    if selected_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise RuntimeError("冻结模型与当前特征协议不一致")
    if selected_meta.get("feature_cache_signature") != feature_signature:
        raise RuntimeError("冻结模型与当前特征缓存不一致")
    if selected_meta.get("variant") != "expanding_quarterly":
        raise RuntimeError("当前正式模型不是季度扩展窗口")

    frozen_scores = pd.read_parquet(SELECTED_SCORES)
    frozen_score_column = next(
        column for column in frozen_scores.columns if column not in {"ts_code", "trade_date"}
    )
    frozen_scores = frozen_scores[frozen_scores["trade_date"].le(VAL_END)].copy()

    model_columns = {
        "ts_code",
        "trade_date",
        "type",
        "future_ret_5d_rank",
        "future_ret_5d_end_date",
        *FEATURE_COLS,
    }
    model_panel = load_feature_subset(model_columns)
    model_panel = model_panel[model_panel["trade_date"].le(DEVELOPMENT_END)].copy()

    with tempfile.TemporaryDirectory(prefix="feature-ablation-") as folder:
        temp_dir = Path(folder)
        development_score_paths: dict[str, Path] = {}
        for variant, dropped_feature in CANDIDATES.items():
            features = [column for column in FEATURE_COLS if column != dropped_feature]
            scores = fit_predict_lgbm_periodic(
                model_panel,
                5,
                DEVELOPMENT_START,
                DEVELOPMENT_END,
                retrain_months=12,
                score_col=SCORE_COLUMN,
                feature_cols=features,
            ).dropna(subset=[SCORE_COLUMN])
            path = temp_dir / f"{variant}_development.parquet"
            scores.to_parquet(path, index=False)
            development_score_paths[variant] = path
            del scores
            gc.collect()
        del model_panel
        gc.collect()

        product_panel = load_feature_subset(
            product_feature_columns("score_breakout", external_score=True)
        )
        # 保留2018年前数据作为市场状态滚动预热，但产品收益仍从2018开始。
        product_panel = product_panel[
            product_panel["trade_date"].le(DEVELOPMENT_END)
        ].copy()
        low_risk_frame = build_low_risk_return_frame(product_panel)
        baseline_scores = frozen_scores[
            frozen_scores["trade_date"].le(DEVELOPMENT_END)
        ]
        baseline_metrics, baseline_years = _evaluate(
            product_panel,
            baseline_scores,
            frozen_score_column,
            low_risk_frame,
            DEVELOPMENT_END,
            DEVELOPMENT_START,
            DEVELOPMENT_END,
        )
        _assert_frozen_baseline("development", baseline_metrics)
        development_rows = [
            _result_row("baseline", None, baseline_metrics, baseline_years, True)
        ]
        development_passers: list[str] = []
        for variant, dropped_feature in CANDIDATES.items():
            scores = pd.read_parquet(development_score_paths[variant])
            metrics, annual = _evaluate(
                product_panel,
                scores,
                SCORE_COLUMN,
                low_risk_frame,
                DEVELOPMENT_END,
                DEVELOPMENT_START,
                DEVELOPMENT_END,
            )
            passed = passes_development_gate(
                baseline_metrics, metrics, baseline_years, annual
            )
            development_rows.append(
                _result_row(variant, dropped_feature, metrics, annual, passed)
            )
            if passed:
                development_passers.append(variant)
            del scores
            gc.collect()
        pd.DataFrame(development_rows).to_csv(
            output_dir / "DEVELOPMENT_RESULTS.csv", index=False, encoding="utf-8-sig"
        )
        del product_panel, low_risk_frame
        gc.collect()

        selection_rows: list[dict] = []
        selection_passers: list[str] = []
        if development_passers:
            model_panel = load_feature_subset(model_columns)
            model_panel = model_panel[model_panel["trade_date"].le(VAL_END)].copy()
            selection_score_paths: dict[str, Path] = {}
            for variant in development_passers:
                dropped_feature = CANDIDATES[variant]
                features = [column for column in FEATURE_COLS if column != dropped_feature]
                scores = fit_predict_lgbm_periodic(
                    model_panel,
                    5,
                    VAL_START,
                    VAL_END,
                    retrain_months=3,
                    score_col=SCORE_COLUMN,
                    feature_cols=features,
                ).dropna(subset=[SCORE_COLUMN])
                development_scores = pd.read_parquet(development_score_paths[variant])
                combined = pd.concat(
                    [development_scores, scores], ignore_index=True
                ).drop_duplicates(["ts_code", "trade_date"], keep="last")
                path = temp_dir / f"{variant}_through_selection.parquet"
                combined.to_parquet(path, index=False)
                selection_score_paths[variant] = path
                del scores, development_scores, combined
                gc.collect()
            del model_panel
            gc.collect()

            product_panel = load_feature_subset(
                product_feature_columns("score_breakout", external_score=True)
            )
            product_panel = product_panel[
                product_panel["trade_date"].le(VAL_END)
            ].copy()
            low_risk_frame = build_low_risk_return_frame(product_panel)
            baseline_metrics, baseline_years = _evaluate(
                product_panel,
                frozen_scores,
                frozen_score_column,
                low_risk_frame,
                VAL_END,
                VAL_START,
                VAL_END,
            )
            _assert_frozen_baseline("selection", baseline_metrics)
            selection_rows.append(
                _result_row("baseline", None, baseline_metrics, baseline_years, True)
            )
            for variant in development_passers:
                scores = pd.read_parquet(selection_score_paths[variant])
                metrics, annual = _evaluate(
                    product_panel,
                    scores,
                    SCORE_COLUMN,
                    low_risk_frame,
                    VAL_END,
                    VAL_START,
                    VAL_END,
                )
                passed = passes_selection_gate(baseline_metrics, metrics, annual)
                selection_rows.append(
                    _result_row(
                        variant, CANDIDATES[variant], metrics, annual, passed
                    )
                )
                if passed:
                    selection_passers.append(variant)
                del scores
                gc.collect()
            del product_panel, low_risk_frame
            gc.collect()

        selection_frame = pd.DataFrame(selection_rows)
        if selection_frame.empty:
            selection_frame = pd.DataFrame(
                columns=["variant", "dropped_feature", "feature_count", "passed_gate"]
            )
        selection_frame.to_csv(
            output_dir / "SELECTION_RESULTS.csv", index=False, encoding="utf-8-sig"
        )

    selected_candidate = None
    if selection_passers:
        selection_frame = pd.DataFrame(selection_rows).set_index("variant")
        selected_candidate = max(
            selection_passers,
            key=lambda variant: float(selection_frame.loc[variant, "total_ret"]),
        )
    decision = {
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "feature_cache_signature": feature_signature,
        "baseline_variant": selected_meta["variant"],
        "baseline_feature_count": len(FEATURE_COLS),
        "candidates": CANDIDATES,
        "development_period": [DEVELOPMENT_START, DEVELOPMENT_END],
        "selection_period": [VAL_START, VAL_END],
        "observation_evaluated": False,
        "observation_used_for_decision": False,
        "lightgbm_hyperparameters_changed": False,
        "strategy_policy_changed": False,
        "frozen_baseline_replay_matched": True,
        "development_gate": DEVELOPMENT_GATE,
        "selection_gate": SELECTION_GATE,
        "development_passers": development_passers,
        "selection_passers": selection_passers,
        "selected_candidate_for_future_protocol": selected_candidate,
        "formal_feature_set_changed": False,
        "promotion_requires_explicit_command": True,
    }
    (output_dir / "DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[done] development_passers={development_passers} "
        f"selection_passers={selection_passers}"
    )


if __name__ == "__main__":
    main()
