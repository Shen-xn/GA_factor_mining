#!/usr/bin/env python3
"""拆解当前正式产品在2018和2022年的损失来源。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...common.paths import ensure_output_dir
from .run_experiments import FEATURE_PATH, current_feature_cache_signature


YEARS = (2018, 2022)
HISTORY_PATH = Path("outputs/sector/strategy/HISTORY_DAILY.parquet")
ACTIONS_PATH = Path("outputs/sector/strategy/HISTORY_ACTIONS.parquet")
COMPONENTS = ("market", "selection", "low_risk", "cost")


def build_benchmark_returns(
    feature_path: Path,
    years: tuple[int, ...] = YEARS,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    """分批构造I+N板块等权开盘收益，避免载入完整面板。"""
    parquet = pq.ParquetFile(feature_path)
    required = {"type", "return_end_date", "forward_open_ret_1d"}
    missing = required - set(parquet.schema.names)
    if missing:
        raise ValueError(f"特征缓存缺少基准字段: {sorted(missing)}")
    year_prefixes = tuple(str(year) for year in years)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=batch_size):
        frame = batch.to_pandas()
        dates = frame["return_end_date"].fillna("").astype(str)
        selected = frame.loc[
            frame["type"].isin(["I", "N"])
            & dates.str.startswith(year_prefixes)
            & frame["forward_open_ret_1d"].notna(),
            ["return_end_date", "forward_open_ret_1d"],
        ]
        if selected.empty:
            continue
        grouped = selected.groupby("return_end_date")["forward_open_ret_1d"].agg(
            ["sum", "count"]
        )
        for date, row in grouped.iterrows():
            key = str(date)
            totals[key] = totals.get(key, 0.0) + float(row["sum"])
            counts[key] = counts.get(key, 0) + int(row["count"])
    rows = [
        {
            "date": date,
            "benchmark_return": totals[date] / counts[date],
            "benchmark_member_count": counts[date],
        }
        for date in sorted(totals)
    ]
    return pd.DataFrame(rows)


def add_daily_components(
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    attribution_years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """用上一期实际仓位拆分市场、选板块、低风险和成本。"""
    if history["date"].duplicated().any():
        raise ValueError("正式产品账本存在重复收益日")
    out = history.sort_values("date").copy()
    out["held_exposure"] = out["exposure"].shift(1, fill_value=0.0)
    out["held_regime"] = out["regime"].shift(1, fill_value="INITIAL")
    if attribution_years is not None:
        years = out["date"].astype(str).str[:4].astype(int)
        out = out.loc[years.isin(attribution_years)].copy()
    out = out.merge(benchmark, on="date", how="left", validate="one_to_one")
    no_market_move = out["benchmark_return"].isna()
    invalid = no_market_move & out["sector_contribution"].abs().gt(1e-15)
    if invalid.any():
        raise RuntimeError(
            f"有板块收益的日期缺少基准: {out.loc[invalid, 'date'].head().tolist()}"
        )
    out.loc[no_market_move, "benchmark_return"] = 0.0
    out.loc[no_market_move, "benchmark_member_count"] = 0
    out["market_gross_component"] = out["held_exposure"] * out["benchmark_return"]
    out["selection_gross_component"] = (
        out["sector_contribution"] - out["market_gross_component"]
    )
    one_minus_cost = 1.0 - out["cost"]
    out["market_component"] = out["market_gross_component"] * one_minus_cost
    out["selection_component"] = out["selection_gross_component"] * one_minus_cost
    out["low_risk_component"] = out["low_risk_contribution"] * one_minus_cost
    out["cost_component"] = -out["cost"]
    reconstructed = out[[f"{name}_component" for name in COMPONENTS]].sum(axis=1)
    error = float((reconstructed - out["net_return"]).abs().max())
    if error > 1e-12:
        raise RuntimeError(f"日收益归因恒等式失败: {error:.2e}")
    return out


def link_period(frame: pd.DataFrame) -> tuple[dict[str, float], float]:
    """把日收益分量精确链接为期间累计收益分量。"""
    local_equity = (1.0 + frame["net_return"]).cumprod()
    prior_equity = local_equity.shift(1, fill_value=1.0)
    contributions = {
        name: float((prior_equity * frame[f"{name}_component"]).sum())
        for name in COMPONENTS
    }
    total_return = float(local_equity.iloc[-1] - 1.0)
    if abs(sum(contributions.values()) - total_return) > 1e-12:
        raise RuntimeError("期间链接贡献不能还原累计收益")
    return contributions, total_return


def _year_summary(frame: pd.DataFrame, year: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    selected = frame[frame["date"].str.startswith(str(year))].copy()
    if selected.empty:
        raise RuntimeError(f"正式账本缺少{year}年")
    contributions, total_return = link_period(selected)
    local_equity = (1.0 + selected["net_return"]).cumprod()
    prior_equity = local_equity.shift(1, fill_value=1.0)
    for name in COMPONENTS:
        selected[f"year_linked_{name}"] = prior_equity * selected[f"{name}_component"]
    row = {
        "year": year,
        "days": len(selected),
        "total_return": total_return,
        **{f"{name}_contribution": value for name, value in contributions.items()},
        "average_held_exposure": float(selected["held_exposure"].mean()),
        "average_turnover": float(selected["turnover"].mean()),
        "total_nominal_cost": float(selected["cost"].sum()),
        "cash_day_share": float(selected["held_regime"].eq("CASH").mean()),
        "defensive_day_share": float(selected["held_regime"].eq("DEFENSIVE").mean()),
        "neutral_day_share": float(selected["held_regime"].eq("NEUTRAL").mean()),
        "risk_on_day_share": float(selected["held_regime"].eq("RISK_ON").mean()),
    }

    monthly_rows = []
    selected["month"] = selected["date"].str[:6]
    for month, month_frame in selected.groupby("month", sort=True):
        month_contributions, month_return = link_period(month_frame)
        monthly_rows.append(
            {
                "year": year,
                "month": month,
                "total_return": month_return,
                **{
                    f"{name}_contribution": value
                    for name, value in month_contributions.items()
                },
                "average_held_exposure": float(month_frame["held_exposure"].mean()),
                "average_turnover": float(month_frame["turnover"].mean()),
            }
        )

    regime_rows = []
    for regime, regime_frame in selected.groupby("held_regime", sort=True):
        regime_rows.append(
            {
                "year": year,
                "held_regime": regime,
                "days": len(regime_frame),
                "day_share": len(regime_frame) / len(selected),
                **{
                    f"year_linked_{name}_contribution": float(
                        regime_frame[f"year_linked_{name}"].sum()
                    )
                    for name in COMPONENTS
                },
                "average_held_exposure": float(regime_frame["held_exposure"].mean()),
            }
        )
    return row, pd.DataFrame(monthly_rows), pd.DataFrame(regime_rows)


def _action_summary(actions: pd.DataFrame) -> pd.DataFrame:
    selected = actions[
        actions["execution_date"].astype(str).str[:4].astype(int).isin(YEARS)
    ].copy()
    selected["year"] = selected["execution_date"].astype(str).str[:4].astype(int)
    selected["asset_leg"] = np.where(selected["ts_code"].eq("LOW_RISK"), "low_risk", "sector")
    return (
        selected.groupby(["year", "asset_leg", "action", "reason"], dropna=False)
        .agg(
            action_count=("ts_code", "size"),
            absolute_weight_change=("weight_change", lambda values: float(values.abs().sum())),
            allocated_expected_cost=("allocated_expected_cost", "sum"),
            median_held_sessions=("held_sessions", "median"),
        )
        .reset_index()
        .sort_values(["year", "asset_leg", "action_count"], ascending=[True, True, False])
    )


def main() -> None:
    output_dir = ensure_output_dir("sector", "bad_year_attribution")
    history = pd.read_parquet(HISTORY_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    benchmark = build_benchmark_returns(FEATURE_PATH)
    daily = add_daily_components(history, benchmark, attribution_years=YEARS)

    summaries = []
    monthly_frames = []
    regime_frames = []
    diagnostics = {}
    for year in YEARS:
        summary, monthly, regime = _year_summary(daily, year)
        summaries.append(summary)
        monthly_frames.append(monthly)
        regime_frames.append(regime)
        negative_components = {
            name: summary[f"{name}_contribution"]
            for name in COMPONENTS
            if summary[f"{name}_contribution"] < 0
        }
        primary_driver = min(negative_components, key=negative_components.get)
        worst_month = monthly.loc[monthly["total_return"].idxmin()]
        diagnostics[str(year)] = {
            "primary_negative_driver": primary_driver,
            "worst_month": str(worst_month["month"]),
            "worst_month_return": float(worst_month["total_return"]),
            "average_held_exposure": summary["average_held_exposure"],
            "cost_share_of_absolute_loss": abs(summary["cost_contribution"])
            / abs(summary["total_return"]),
        }

    summary_frame = pd.DataFrame(summaries)
    monthly_frame = pd.concat(monthly_frames, ignore_index=True)
    regime_frame = pd.concat(regime_frames, ignore_index=True)
    action_frame = _action_summary(actions)
    summary_frame.to_csv(output_dir / "BAD_YEAR_SUMMARY.csv", index=False, encoding="utf-8-sig")
    monthly_frame.to_csv(output_dir / "BAD_YEAR_MONTHLY.csv", index=False, encoding="utf-8-sig")
    regime_frame.to_csv(output_dir / "BAD_YEAR_REGIME.csv", index=False, encoding="utf-8-sig")
    action_frame.to_csv(output_dir / "BAD_YEAR_ACTIONS.csv", index=False, encoding="utf-8-sig")
    common_driver = len({item["primary_negative_driver"] for item in diagnostics.values()}) == 1
    (output_dir / "DIAGNOSIS.json").write_text(
        json.dumps(
            {
                "feature_cache_signature": current_feature_cache_signature(),
                "product_history_path": str(HISTORY_PATH),
                "years": list(YEARS),
                "benchmark": "daily equal-weight I+N sector open-to-open return",
                "exact_linking_identity": True,
                "observation_used_for_design": False,
                "year_diagnostics": diagnostics,
                "same_primary_negative_driver": common_driver,
                "strategy_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {output_dir} common_driver={common_driver}")


if __name__ == "__main__":
    main()
