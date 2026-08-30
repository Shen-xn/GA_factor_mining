#!/usr/bin/env python3
"""验证唯一预声明的DEFENSIVE连续技术仓位候选。"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd

from ...common.paths import ensure_output_dir
from .low_risk import build_low_risk_return_frame
from .product_backtest import (
    PRODUCT_HISTORY_START,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
)
from .run_experiments import (
    FEATURE_PROTOCOL_VERSION,
    TRAIN_END,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_feature_subset,
)
from .strategy import get_strategy_policy


SELECTED_META = Path("outputs/sector/adaptation/SELECTED.json")
SELECTED_SCORES = Path("outputs/sector/adaptation/SELECTED_SCORES.parquet")
FORMAL_SUMMARY = Path("outputs/sector/strategy/SUMMARY.csv")
DEVELOPMENT_START = PRODUCT_HISTORY_START
CANDIDATE = "continuous_defensive_exposure"

DEVELOPMENT_GATE = {
    "cumulative_return_positive": True,
    "positive_years_min": 4,
    "worst_year_improvement_min": 0.04,
    "worst_year_floor": -0.10,
    "max_drawdown_floor": -0.20,
    "average_turnover_max": 0.10,
    "non_crisis_year_lag_max": 0.03,
}

SELECTION_GATE = {
    "cumulative_return_retention_min": 0.80,
    "each_year_positive": True,
    "max_drawdown_floor": -0.12,
    "max_drawdown_worsening_max": 0.02,
    "average_turnover_max": 0.12,
    "thirty_bp_cumulative_return_positive": True,
}


def _annual_returns(daily: pd.DataFrame, start: str, end: str) -> dict[int, float]:
    selected = daily[daily["date"].between(start, end)].copy()
    selected["year"] = selected["date"].str[:4].astype(int)
    return {
        int(year): float((1.0 + frame["net_return"]).prod() - 1.0)
        for year, frame in selected.groupby("year")
    }


def development_gate_failures(
    baseline: dict,
    candidate: dict,
    baseline_years: dict[int, float],
    candidate_years: dict[int, float],
) -> list[str]:
    failures = []
    if candidate["total_ret"] <= 0.0:
        failures.append("development cumulative return is not positive")
    if sum(value > 0 for value in candidate_years.values()) < 4:
        failures.append("fewer than four of six development years are positive")
    baseline_worst = min(baseline_years.values())
    candidate_worst = min(candidate_years.values())
    if candidate_worst < baseline_worst + 0.04:
        failures.append("worst year did not improve by at least four percentage points")
    if candidate_worst < -0.10:
        failures.append("worst year remained below -10%")
    if candidate["max_drawdown"] < -0.20:
        failures.append("development maximum drawdown exceeded 20%")
    if candidate["avg_turnover"] > 0.10:
        failures.append("development average daily turnover exceeded 10%")
    for year in (2019, 2020, 2021, 2023):
        if candidate_years.get(year, -1.0) < baseline_years.get(year, -1.0) - 0.03:
            failures.append(f"{year} lagged baseline by more than three percentage points")
    return failures


def selection_gate_failures(
    baseline: dict,
    candidate: dict,
    candidate_years: dict[int, float],
    candidate_30bp_total: float,
) -> list[str]:
    failures = []
    if candidate["total_ret"] < baseline["total_ret"] * 0.80:
        failures.append("selection cumulative return retained less than 80% of baseline")
    if candidate_years.get(2024, -1.0) <= 0 or candidate_years.get(2025, -1.0) <= 0:
        failures.append("2024 and 2025 were not both positive")
    if candidate["max_drawdown"] < -0.12:
        failures.append("selection maximum drawdown exceeded 12%")
    if candidate["max_drawdown"] < baseline["max_drawdown"] - 0.02:
        failures.append("selection maximum drawdown worsened by more than two points")
    if candidate["avg_turnover"] > 0.12:
        failures.append("selection average daily turnover exceeded 12%")
    if candidate_30bp_total <= 0.0:
        failures.append("30bp selection cumulative return was not positive")
    return failures


def _evaluate(
    panel: pd.DataFrame,
    score_name: str,
    low_risk_frame: pd.DataFrame,
    end: str,
    period_start: str,
    period_end: str,
    technical_defensive: bool,
    cost_bps: float = 20.0,
) -> tuple[dict, dict[int, float], pd.DataFrame]:
    daily, actions, _ = run_product_backtest(
        panel,
        score_name,
        PRODUCT_HISTORY_START,
        end,
        cost_bps=cost_bps,
        strategy_policy=get_strategy_policy("simple_v1"),
        low_risk_frame=low_risk_frame,
        use_continuous_defensive_exposure=technical_defensive,
    )
    _, _, metrics = summarize_backtest_period(daily, actions, period_start, period_end)
    annual = _annual_returns(daily, period_start, period_end)
    del actions
    gc.collect()
    return metrics, annual, daily


def _assert_frozen_baseline(period: str, metrics: dict) -> None:
    formal = pd.read_csv(FORMAL_SUMMARY).set_index("period").loc[period]
    for key in ("total_ret", "max_drawdown", "avg_turnover"):
        if abs(float(metrics[key]) - float(formal[key])) > 1e-12:
            raise RuntimeError(f"{period}基线未与正式账本对齐: {key}")


def _row(
    variant: str,
    metrics: dict,
    annual: dict[int, float],
    failures: list[str],
    cost_bps: float = 20.0,
) -> dict:
    return {
        "variant": variant,
        "cost_bps": cost_bps,
        "passed_gate": not failures,
        "failed_gates": "|".join(failures),
        **metrics,
        **{f"return_{year}": value for year, value in sorted(annual.items())},
    }


def main() -> None:
    output_dir = ensure_output_dir("sector", "defensive_exposure")
    meta = json.loads(SELECTED_META.read_text(encoding="utf-8"))
    feature_signature = current_feature_cache_signature()
    if meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise RuntimeError("冻结模型与当前特征协议不一致")
    if meta.get("feature_cache_signature") != feature_signature:
        raise RuntimeError("冻结模型与当前特征缓存不一致")

    scores = pd.read_parquet(SELECTED_SCORES)
    score_name = next(
        column for column in scores.columns if column not in {"ts_code", "trade_date"}
    )
    panel = load_feature_subset(
        product_feature_columns("score_breakout", external_score=True)
    )
    panel = panel[panel["trade_date"].le(TRAIN_END)].merge(
        scores[scores["trade_date"].le(TRAIN_END)],
        on=["ts_code", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    low_risk_frame = build_low_risk_return_frame(panel)
    baseline_metrics, baseline_years, baseline_daily = _evaluate(
        panel, score_name, low_risk_frame, TRAIN_END, DEVELOPMENT_START, TRAIN_END, False
    )
    _assert_frozen_baseline("development", baseline_metrics)
    candidate_metrics, candidate_years, candidate_daily = _evaluate(
        panel, score_name, low_risk_frame, TRAIN_END, DEVELOPMENT_START, TRAIN_END, True
    )
    failures = development_gate_failures(
        baseline_metrics, candidate_metrics, baseline_years, candidate_years
    )
    development_rows = [
        _row("baseline", baseline_metrics, baseline_years, []),
        _row(CANDIDATE, candidate_metrics, candidate_years, failures),
    ]
    pd.DataFrame(development_rows).to_csv(
        output_dir / "DEVELOPMENT_RESULTS.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "variant": ["baseline", CANDIDATE],
            "average_exposure": [
                float(baseline_daily["exposure"].mean()),
                float(candidate_daily["exposure"].mean()),
            ],
            "defensive_average_exposure": [
                float(baseline_daily.loc[baseline_daily["regime"].eq("DEFENSIVE"), "exposure"].mean()),
                float(candidate_daily.loc[candidate_daily["regime"].eq("DEFENSIVE"), "exposure"].mean()),
            ],
            "zero_exposure_day_share": [
                float(baseline_daily["exposure"].le(1e-12).mean()),
                float(candidate_daily["exposure"].le(1e-12).mean()),
            ],
        }
    ).to_csv(output_dir / "EXPOSURE_DIAGNOSTICS.csv", index=False, encoding="utf-8-sig")
    development_passed = not failures
    del panel, low_risk_frame, baseline_daily, candidate_daily
    gc.collect()

    selection_rows = []
    selection_failures: list[str] = []
    selected_candidate = None
    if development_passed:
        panel = load_feature_subset(
            product_feature_columns("score_breakout", external_score=True)
        )
        panel = panel[panel["trade_date"].le(VAL_END)].merge(
            scores[scores["trade_date"].le(VAL_END)],
            on=["ts_code", "trade_date"],
            how="left",
            validate="many_to_one",
        )
        low_risk_frame = build_low_risk_return_frame(panel)
        baseline_metrics, baseline_years, baseline_daily = _evaluate(
            panel, score_name, low_risk_frame, VAL_END, VAL_START, VAL_END, False
        )
        _assert_frozen_baseline("selection", baseline_metrics)
        candidate_metrics, candidate_years, candidate_daily = _evaluate(
            panel, score_name, low_risk_frame, VAL_END, VAL_START, VAL_END, True
        )
        candidate_30_metrics, _, candidate_30_daily = _evaluate(
            panel,
            score_name,
            low_risk_frame,
            VAL_END,
            VAL_START,
            VAL_END,
            True,
            cost_bps=30.0,
        )
        selection_failures = selection_gate_failures(
            baseline_metrics,
            candidate_metrics,
            candidate_years,
            candidate_30_metrics["total_ret"],
        )
        selection_rows = [
            _row("baseline", baseline_metrics, baseline_years, []),
            _row(CANDIDATE, candidate_metrics, candidate_years, selection_failures),
            _row(
                f"{CANDIDATE}_30bp",
                candidate_30_metrics,
                {},
                [] if candidate_30_metrics["total_ret"] > 0 else ["not positive"],
                cost_bps=30.0,
            ),
        ]
        if not selection_failures:
            selected_candidate = CANDIDATE
        del panel, low_risk_frame, baseline_daily, candidate_daily, candidate_30_daily
        gc.collect()
    selection_frame = pd.DataFrame(selection_rows)
    if selection_frame.empty:
        selection_frame = pd.DataFrame(
            columns=["variant", "cost_bps", "passed_gate", "failed_gates"]
        )
    selection_frame.to_csv(
        output_dir / "SELECTION_RESULTS.csv", index=False, encoding="utf-8-sig"
    )

    (output_dir / "DECISION.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": feature_signature,
                "candidate": CANDIDATE,
                "formula": (
                    "DEFENSIVE exposure = 0.30 * sqrt("
                    "clip((trend60+0.12)/0.12,0,1) * "
                    "clip((breadth20-0.20)/0.25,0,1))"
                ),
                "development_period": [DEVELOPMENT_START, TRAIN_END],
                "selection_period": [VAL_START, VAL_END],
                "observation_evaluated": False,
                "observation_used_for_decision": False,
                "development_gate": DEVELOPMENT_GATE,
                "selection_gate": SELECTION_GATE,
                "frozen_baseline_replay_matched": True,
                "development_passed": development_passed,
                "development_failures": failures,
                "selection_opened": development_passed,
                "selection_failures": selection_failures,
                "selected_candidate_for_future_protocol": selected_candidate,
                "formal_policy_changed": False,
                "promotion_requires_explicit_command": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] development_passed={development_passed} "
        f"selected={selected_candidate}"
    )


if __name__ == "__main__":
    main()
