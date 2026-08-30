#!/usr/bin/env python3
"""只比较年度与季度扩展窗口重训，不读取2026选择参数。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from ...common.paths import ensure_output_dir
from .low_risk import DEFAULT_LOW_RISK_CODE, build_low_risk_return_frame, low_risk_data_signature
from .product_backtest import (
    PRODUCT_HISTORY_START,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
)
from .rolling_validation import FEATURE_COLS, fit_predict_lgbm_periodic, periodic_prediction_bounds
from .run_experiments import (
    FEATURE_PROTOCOL_VERSION,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_feature_subset,
)
from .strategy import get_strategy_policy


BASELINE_META = Path("outputs/sector/adaptation/SELECTED.json")
BASELINE_SCORES = Path("outputs/sector/adaptation/SELECTED_SCORES.parquet")
FREQUENCY_DIR = Path("outputs/sector/model_frequency")
CHALLENGE_START = VAL_START
CHALLENGE_END = VAL_END
QUARTERLY_SCORE = "score_expanding_quarterly_5d"


def passes_frequency_gate(baseline: dict, candidate: dict, annual_returns: dict[int, float]) -> bool:
    """更频繁重训必须带来明确增量，不能只产生不同预测。"""
    return (
        candidate["total_ret"] >= baseline["total_ret"] + 0.02
        and candidate["sharpe"] >= baseline["sharpe"]
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02
        and candidate["avg_turnover"] <= baseline["avg_turnover"] + 0.01
        and annual_returns.get(2024, -1.0) > 0.0
        and annual_returns.get(2025, -1.0) > 0.0
    )


def _annual_returns(daily: pd.DataFrame) -> dict[int, float]:
    selected = daily[daily["date"].between(VAL_START, VAL_END)].copy()
    selected["year"] = selected["date"].str[:4].astype(int)
    return {
        int(year): float((1.0 + frame["net_return"]).prod() - 1.0)
        for year, frame in selected.groupby("year")
    }


def _score_stability(scores: pd.DataFrame, annual_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """量化季度模型相对年度模型的横截面变化幅度。"""
    rows = []
    for trade_date, day in scores.groupby("trade_date", sort=True):
        valid = day[[annual_col, QUARTERLY_SCORE]].dropna()
        if len(valid) < 5:
            continue
        correlation = float(valid[annual_col].corr(valid[QUARTERLY_SCORE], method="spearman"))
        annual_top = set(valid.nlargest(5, annual_col).index)
        quarterly_top = set(valid.nlargest(5, QUARTERLY_SCORE).index)
        rows.append(
            {
                "trade_date": str(trade_date),
                "rank_correlation": correlation,
                "top5_overlap": len(annual_top & quarterly_top) / 5.0,
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "mean_rank_correlation": float(daily["rank_correlation"].mean()),
        "p10_rank_correlation": float(daily["rank_correlation"].quantile(0.10)),
        "mean_top5_overlap": float(daily["top5_overlap"].mean()),
        "p10_top5_overlap": float(daily["top5_overlap"].quantile(0.10)),
    }
    return daily, summary


def main() -> None:
    baseline_meta = json.loads(BASELINE_META.read_text(encoding="utf-8"))
    feature_signature = current_feature_cache_signature()
    if baseline_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise RuntimeError("冻结年度模型与当前特征协议不一致")
    if baseline_meta.get("feature_cache_signature") != feature_signature:
        raise RuntimeError("冻结年度模型与当前特征缓存不一致")
    if int(baseline_meta.get("retrain_months", 0)) != 12:
        raise RuntimeError("当前基线不是年度重训，不能进行年度/季度对照")

    # 训练阶段只加载模型需要的列，并明确截断在2025年底。
    model_columns = {
        "ts_code",
        "trade_date",
        "type",
        "future_ret_5d_rank",
        "future_ret_5d_end_date",
        *FEATURE_COLS,
    }
    model_panel = load_feature_subset(model_columns)
    model_panel = model_panel[model_panel["trade_date"].le(CHALLENGE_END)].copy()
    quarterly_scores = fit_predict_lgbm_periodic(
        model_panel,
        5,
        CHALLENGE_START,
        CHALLENGE_END,
        retrain_months=3,
        score_col=QUARTERLY_SCORE,
    )
    quarterly_scores = quarterly_scores[
        quarterly_scores["trade_date"].between(CHALLENGE_START, CHALLENGE_END)
    ].dropna(subset=[QUARTERLY_SCORE])
    del model_panel
    gc.collect()

    annual_scores = pd.read_parquet(BASELINE_SCORES)
    annual_col = next(
        column for column in annual_scores.columns if column not in {"ts_code", "trade_date"}
    )
    annual_scores = annual_scores[annual_scores["trade_date"].le(CHALLENGE_END)].copy()
    comparison_scores = annual_scores[
        annual_scores["trade_date"].between(CHALLENGE_START, CHALLENGE_END)
    ].merge(quarterly_scores, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")
    stability_daily, stability_summary = _score_stability(comparison_scores, annual_col)

    # 2024年前沿用年度评分，确保两条产品路径拥有完全相同的历史状态。
    candidate_scores = annual_scores[["ts_code", "trade_date", annual_col]].rename(
        columns={annual_col: QUARTERLY_SCORE}
    )
    candidate_scores = candidate_scores[candidate_scores["trade_date"].lt(CHALLENGE_START)]
    candidate_scores = pd.concat(
        [candidate_scores, quarterly_scores], ignore_index=True
    ).drop_duplicates(["ts_code", "trade_date"], keep="last")

    product_panel = load_feature_subset(
        product_feature_columns("score_breakout", external_score=True)
    )
    product_panel = product_panel[
        product_panel["trade_date"].between(PRODUCT_HISTORY_START, CHALLENGE_END)
    ].copy()
    low_risk_frame = build_low_risk_return_frame(product_panel)
    policy = get_strategy_policy("simple_v1")

    results = []
    metrics_by_variant: dict[str, dict] = {}
    annual_by_variant: dict[str, dict[int, float]] = {}
    for variant, scores, score_col in (
        ("annual_expanding", annual_scores, annual_col),
        ("quarterly_expanding", candidate_scores, QUARTERLY_SCORE),
    ):
        scored = product_panel.merge(
            scores[["ts_code", "trade_date", score_col]],
            on=["ts_code", "trade_date"],
            how="left",
            validate="many_to_one",
        )
        daily, actions, _ = run_product_backtest(
            scored,
            score_col,
            PRODUCT_HISTORY_START,
            CHALLENGE_END,
            cost_bps=20.0,
            strategy_policy=policy,
            low_risk_frame=low_risk_frame,
        )
        _, _, metrics = summarize_backtest_period(daily, actions, VAL_START, VAL_END)
        annual_returns = _annual_returns(daily)
        metrics_by_variant[variant] = metrics
        annual_by_variant[variant] = annual_returns
        results.append(
            {
                "variant": variant,
                "retrain_months": 12 if variant == "annual_expanding" else 3,
                "return_2024": annual_returns.get(2024),
                "return_2025": annual_returns.get(2025),
                **metrics,
            }
        )
        del scored, daily, actions
        gc.collect()

    baseline = metrics_by_variant["annual_expanding"]
    candidate = metrics_by_variant["quarterly_expanding"]
    passed = passes_frequency_gate(
        baseline,
        candidate,
        annual_by_variant["quarterly_expanding"],
    )
    output_dir = ensure_output_dir("sector", "model_frequency")
    pd.DataFrame(results).to_csv(
        output_dir / "ANNUAL_VS_QUARTERLY.csv", index=False, encoding="utf-8-sig"
    )
    stability_daily.to_csv(
        output_dir / "DAILY_STABILITY.csv", index=False, encoding="utf-8-sig"
    )
    quarterly_scores.to_parquet(
        output_dir / "QUARTERLY_SCORES_2024_2025.parquet", index=False
    )
    (output_dir / "DECISION.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": feature_signature,
                "baseline": "annual_expanding",
                "challenger": "quarterly_expanding",
                "challenge_period": [CHALLENGE_START, CHALLENGE_END],
                "shared_product_history_start": PRODUCT_HISTORY_START,
                "shared_state_before_challenge": True,
                "observation_used_for_selection": False,
                "observation_evaluated": False,
                "lightgbm_hyperparameters_changed": False,
                "passed": passed,
                "selected_variant_for_future_protocol": (
                    "quarterly_expanding" if passed else None
                ),
                "current_frozen_product_changed": False,
                "selection_metrics": {
                    "annual_expanding": baseline,
                    "quarterly_expanding": candidate,
                },
                "selection_year_returns": annual_by_variant,
                "gate": {
                    "selection_total_return_improvement_min": 0.02,
                    "sharpe_not_lower": True,
                    "max_drawdown_worsening_max": 0.02,
                    "daily_turnover_worsening_max": 0.01,
                    "each_selection_year_positive": True,
                },
                "stability": stability_summary,
                "retrain_windows": [
                    {
                        "train_end": train_end,
                        "predict_start": predict_start,
                        "predict_end": predict_end,
                    }
                    for train_end, predict_start, predict_end in periodic_prediction_bounds(
                        CHALLENGE_START, CHALLENGE_END, 3
                    )
                ],
                "low_risk_code": DEFAULT_LOW_RISK_CODE,
                "low_risk_data_signature": low_risk_data_signature(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] quarterly_passed={passed}")


def promote_quarterly() -> None:
    """将已通过选择门的季度评分晋级；不在此读取2026产品表现。"""
    decision = json.loads((FREQUENCY_DIR / "DECISION.json").read_text(encoding="utf-8"))
    if not decision.get("passed"):
        raise RuntimeError("季度重训没有通过选择门，禁止晋级")
    if decision.get("selected_variant_for_future_protocol") != "quarterly_expanding":
        raise RuntimeError("晋级目标不是冻结的季度扩展窗口")
    if decision.get("observation_evaluated") or decision.get("observation_used_for_selection"):
        raise RuntimeError("选择过程读取了观察期，禁止晋级")

    baseline_meta = json.loads(BASELINE_META.read_text(encoding="utf-8"))
    data_end = str(baseline_meta["score_data_end"])
    model_columns = {
        "ts_code",
        "trade_date",
        "type",
        "future_ret_5d_rank",
        "future_ret_5d_end_date",
        *FEATURE_COLS,
    }
    model_panel = load_feature_subset(model_columns)
    model_panel = model_panel[model_panel["trade_date"].le(data_end)].copy()
    if data_end >= "20260101":
        recent_scores = fit_predict_lgbm_periodic(
            model_panel,
            5,
            "20260101",
            data_end,
            retrain_months=3,
            score_col=QUARTERLY_SCORE,
        )
        recent_scores = recent_scores[
            recent_scores["trade_date"].between("20260101", data_end)
        ].dropna(subset=[QUARTERLY_SCORE])
    else:
        recent_scores = pd.DataFrame(columns=["ts_code", "trade_date", QUARTERLY_SCORE])
    del model_panel
    gc.collect()

    annual_scores = pd.read_parquet(BASELINE_SCORES)
    annual_col = next(
        column for column in annual_scores.columns if column not in {"ts_code", "trade_date"}
    )
    pre_challenge = annual_scores[annual_scores["trade_date"].lt(CHALLENGE_START)][
        ["ts_code", "trade_date", annual_col]
    ].rename(columns={annual_col: QUARTERLY_SCORE})
    selected_quarterly = pd.read_parquet(
        FREQUENCY_DIR / "QUARTERLY_SCORES_2024_2025.parquet"
    )
    promoted = pd.concat(
        [pre_challenge, selected_quarterly, recent_scores], ignore_index=True
    ).drop_duplicates(["ts_code", "trade_date"], keep="last")
    promoted = promoted.rename(columns={QUARTERLY_SCORE: annual_col}).sort_values(
        ["ts_code", "trade_date"]
    )
    if promoted.duplicated(["ts_code", "trade_date"]).any():
        raise RuntimeError("晋级评分存在重复键")
    if str(promoted["trade_date"].max()) != data_end:
        raise RuntimeError("晋级评分没有覆盖当前数据截止日")

    archive_dir = BASELINE_META.parent / "archive" / f"annual_expanding_{data_end}"
    if archive_dir.exists():
        raise RuntimeError(f"年度评分归档已存在: {archive_dir}")
    archive_dir.mkdir(parents=True)
    shutil.copy2(BASELINE_META, archive_dir / BASELINE_META.name)
    shutil.copy2(BASELINE_SCORES, archive_dir / BASELINE_SCORES.name)

    promoted_meta = {
        **baseline_meta,
        "variant": "expanding_quarterly",
        "selection_period": "2024-2025 model frequency selection",
        "selection_rule": (
            "quarterly must improve total return by >=2pct, not lower Sharpe, "
            "not worsen MDD by >2pct or turnover by >1pct, and both years positive"
        ),
        "retrain_months": 3,
        "quarterly_schedule_effective_from": CHALLENGE_START,
        "pre_challenge_score_source": "annual_expanding",
        "observation_used_for_selection": False,
        "observation_evaluated_for_selection": False,
        "score_data_end": data_end,
        "promotion_decision_path": str(FREQUENCY_DIR / "DECISION.json"),
    }
    with tempfile.TemporaryDirectory(prefix="quarterly-promotion-") as folder:
        staged_scores = Path(folder) / BASELINE_SCORES.name
        staged_meta = Path(folder) / BASELINE_META.name
        promoted.to_parquet(staged_scores, index=False)
        staged_meta.write_text(
            json.dumps(promoted_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staged_scores, BASELINE_SCORES)
        os.replace(staged_meta, BASELINE_META)

    # 晋级事实单独写回决策文件，避免研究结论与当前默认模型不一致。
    decision.update(
        {
            "current_frozen_product_changed": True,
            "promoted": True,
            "promoted_variant": "expanding_quarterly",
            "promoted_score_data_end": data_end,
        }
    )
    (FREQUENCY_DIR / "DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[promoted] expanding_quarterly through {data_end}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="比较或晋级季度扩展窗口LightGBM")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="仅在现有DECISION通过时，将季度评分晋级为默认模型",
    )
    args = parser.parse_args()
    if args.promote:
        promote_quarterly()
    else:
        main()


if __name__ == "__main__":
    cli()
