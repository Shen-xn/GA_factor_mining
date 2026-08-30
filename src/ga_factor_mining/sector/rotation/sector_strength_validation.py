"""在封存2026的前提下验证一个板块自身趋势覆盖候选。"""

from __future__ import annotations

import json

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


def _load_scored_panel() -> tuple[pd.DataFrame, str, pd.DataFrame]:
    panel = load_feature_subset(product_feature_columns("score_breakout", external_score=True))
    score_dir = ensure_output_dir("sector", "adaptation")
    score_meta = json.loads((score_dir / "SELECTED.json").read_text(encoding="utf-8"))
    if (
        score_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION
        or score_meta.get("feature_cache_signature") != current_feature_cache_signature()
    ):
        raise RuntimeError("冻结评分与当前特征协议不一致")
    predictions = pd.read_parquet(score_dir / "SELECTED_SCORES.parquet")
    score_columns = [column for column in predictions if column not in {"ts_code", "trade_date"}]
    if len(score_columns) != 1:
        raise ValueError("冻结评分应只有一个评分字段")
    score_name = score_columns[0]
    scored = panel.merge(predictions, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    return scored, score_name, build_low_risk_return_frame(panel)


def _period_rows(
    variant: str,
    daily: pd.DataFrame,
    actions: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    periods = (
        ("development", PRODUCT_HISTORY_START, TRAIN_END),
        ("selection", VAL_START, VAL_END),
        ("pre_observation", PRODUCT_HISTORY_START, VAL_END),
    )
    for period, start, end in periods:
        period_daily, _, metrics = summarize_backtest_period(daily, actions, start, end)
        rows.append(
            {
                "variant": variant,
                "period": period,
                "start": start,
                "end": end,
                "override_days": int(period_daily["sector_strength_override"].sum()),
                **metrics,
            }
        )
    return rows


def _annual_rows(
    variant: str,
    daily: pd.DataFrame,
    actions: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for year in range(2018, 2026):
        _, _, metrics = summarize_backtest_period(
            daily, actions, f"{year}0101", f"{year}1231"
        )
        rows.append({"variant": variant, "year": year, **metrics})
    return rows


def _gate(periods: pd.DataFrame, annual: pd.DataFrame) -> dict:
    indexed_periods = periods.set_index(["variant", "period"])
    indexed_annual = annual.set_index(["variant", "year"])
    baseline_selection = indexed_periods.loc[("simple_v1", "selection")]
    candidate_selection = indexed_periods.loc[("sector_strength_candidate", "selection")]
    baseline_full = indexed_periods.loc[("simple_v1", "pre_observation")]
    candidate_full = indexed_periods.loc[("sector_strength_candidate", "pre_observation")]
    candidate_years = annual[annual["variant"].eq("sector_strength_candidate")]
    checks = {
        "majority_years_positive": int(candidate_years["total_ret"].gt(0).sum()) >= 5,
        "2018_not_worse": float(indexed_annual.loc[("sector_strength_candidate", 2018), "total_ret"])
        >= float(indexed_annual.loc[("simple_v1", 2018), "total_ret"]),
        "2022_not_worse": float(indexed_annual.loc[("sector_strength_candidate", 2022), "total_ret"])
        >= float(indexed_annual.loc[("simple_v1", 2022), "total_ret"]),
        "2024_positive": float(indexed_annual.loc[("sector_strength_candidate", 2024), "total_ret"]) > 0,
        "2025_positive": float(indexed_annual.loc[("sector_strength_candidate", 2025), "total_ret"]) > 0,
        "full_max_drawdown_within_20pct": float(candidate_full["max_drawdown"]) >= -0.20,
        "turnover_increase_within_1pct": float(candidate_full["avg_turnover"])
        <= float(baseline_full["avg_turnover"]) + 0.01,
        "selection_return_improves_5pct": float(candidate_selection["total_ret"])
        >= float(baseline_selection["total_ret"]) + 0.05,
        "selection_drawdown_not_worse_2pct": float(candidate_selection["max_drawdown"])
        >= float(baseline_selection["max_drawdown"]) - 0.02,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observation_opened": False,
        "candidate_name": "sector_strength_candidate",
        "candidate_rule": {
            "active_regime": "DEFENSIVE only",
            "leader_definition": "at least 3 of score Top5 have ret_20d > 0 and ret_5d_rank >= 0.5",
            "effect": "raise market exposure floor to 70% and retain five positions",
            "drawdown_cap_still_applies": True,
        },
        "selection_metrics": {
            "baseline_total_ret": float(baseline_selection["total_ret"]),
            "candidate_total_ret": float(candidate_selection["total_ret"]),
            "baseline_max_drawdown": float(baseline_selection["max_drawdown"]),
            "candidate_max_drawdown": float(candidate_selection["max_drawdown"]),
        },
        "pre_observation_metrics": {
            "baseline_total_ret": float(baseline_full["total_ret"]),
            "candidate_total_ret": float(candidate_full["total_ret"]),
            "baseline_max_drawdown": float(baseline_full["max_drawdown"]),
            "candidate_max_drawdown": float(candidate_full["max_drawdown"]),
            "baseline_avg_turnover": float(baseline_full["avg_turnover"]),
            "candidate_avg_turnover": float(candidate_full["avg_turnover"]),
        },
    }


def main() -> None:
    scored, score_name, low_risk_frame = _load_scored_panel()
    policy = get_strategy_policy("simple_v1")
    variants = {
        "simple_v1": False,
        "sector_strength_candidate": True,
    }
    period_rows: list[dict] = []
    annual_rows: list[dict] = []
    for variant, use_override in variants.items():
        print(f"[candidate] {variant} through {VAL_END}")
        daily, actions, _ = run_product_backtest(
            scored,
            score_name,
            PRODUCT_HISTORY_START,
            VAL_END,
            strategy_policy=policy,
            cost_bps=20.0,
            low_risk_frame=low_risk_frame,
            use_market_regime=True,
            use_drawdown_cap=True,
            use_sector_strength_override=use_override,
        )
        period_rows.extend(_period_rows(variant, daily, actions))
        annual_rows.extend(_annual_rows(variant, daily, actions))

    periods = pd.DataFrame(period_rows)
    annual = pd.DataFrame(annual_rows)
    decision = _gate(periods, annual)
    output_dir = ensure_output_dir("sector", "sector_strength_candidate")
    periods.to_csv(output_dir / "PERIOD_RESULTS.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output_dir / "ANNUAL_RESULTS.csv", index=False, encoding="utf-8-sig")
    (output_dir / "DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if decision["passed"]:
        # 只有2018—2025闸门通过，才允许计算2026观察段。
        from .run_experiments import OBSERVATION_END, OBSERVATION_START

        daily, actions, _ = run_product_backtest(
            scored,
            score_name,
            PRODUCT_HISTORY_START,
            OBSERVATION_END,
            strategy_policy=policy,
            cost_bps=20.0,
            low_risk_frame=low_risk_frame,
            use_market_regime=True,
            use_drawdown_cap=True,
            use_sector_strength_override=True,
        )
        _, _, observation = summarize_backtest_period(
            daily, actions, OBSERVATION_START, OBSERVATION_END
        )
        pd.DataFrame([observation]).to_csv(
            output_dir / "OBSERVATION_RESULTS.csv", index=False, encoding="utf-8-sig"
        )
        decision["observation_opened"] = True
        (output_dir / "DECISION.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[done] passed={decision['passed']} output={output_dir}")


if __name__ == "__main__":
    main()
