"""用真实ETF开盘价复核板块策略的可落地执行路径。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .etf_mapping import (
    ETF_ADJ_PATH,
    ETF_DAILY_PATH,
    build_active_mapping_lookup,
    load_adjusted_etf_prices,
    resolve_target_weights_from_active,
)
from .low_risk import DEFAULT_LOW_RISK_CODE, _load_adjusted_prices
from .run_experiments import annualized_metrics


@dataclass(frozen=True)
class EtfReplayResult:
    daily: pd.DataFrame
    resolved_targets: pd.DataFrame
    summary: dict[str, float]


def _period_metrics(daily: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    period = daily.loc[daily["date"].between(start, end)]
    metrics = annualized_metrics(period.set_index("date")["net_return"])
    active = period.loc[period["sector_target_exposure"].gt(1e-12)]
    return {
        **metrics,
        "average_daily_turnover": float(period["turnover"].mean()),
        "annualized_turnover": float(period["turnover"].mean() * 252.0),
        "trade_day_ratio": float(period["turnover"].gt(1e-12).mean()),
        "average_mapping_coverage": float(active["mapping_coverage"].mean()),
        "minimum_mapping_coverage": float(active["mapping_coverage"].min()),
    }


def write_etf_replay_outputs(
    target_timeline: pd.DataFrame,
    mapping: pd.DataFrame,
    open_prices: pd.DataFrame,
    output_dir: Path,
    *,
    periods: tuple[tuple[str, str, str], ...],
    costs_bps: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0),
) -> dict:
    """输出冻结板块目标的ETF纯翻译回放，不重新排名或改变策略。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    # 冻结目标不会受交易成本反馈影响；映射和行情只解析一次，再精确重算净值。
    default_result = run_resolved_etf_backtest(
        target_timeline,
        mapping,
        open_prices,
        cost_bps=20.0,
    )
    for cost_bps in costs_bps:
        cost_daily = default_result.daily.copy()
        cost_daily["cost"] = cost_daily["turnover"] * cost_bps / 10_000.0
        cost_daily["net_return"] = (
            (1.0 + cost_daily["gross_return"]) * (1.0 - cost_daily["cost"]) - 1.0
        )
        cost_daily["equity"] = (1.0 + cost_daily["net_return"]).cumprod()
        cost_daily["drawdown"] = (
            cost_daily["equity"] / cost_daily["equity"].cummax() - 1.0
        )
        for period_name, start, end in periods:
            summary_rows.append(
                {
                    "period": period_name,
                    "cost_bps": cost_bps,
                    "replay_kind": "frozen_sector_targets_to_real_etf_open",
                    **_period_metrics(cost_daily, start, end),
                }
            )
        if cost_bps == 20.0:
            default_result = EtfReplayResult(
                daily=cost_daily,
                resolved_targets=default_result.resolved_targets,
                summary=default_result.summary,
            )
    if 20.0 not in costs_bps:
        raise ValueError("成本集合必须包含正式20bp路径")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "SUMMARY.csv", index=False, encoding="utf-8-sig")
    default_result.daily.to_csv(
        output_dir / "DAILY_20BP.csv", index=False, encoding="utf-8-sig"
    )
    default_result.resolved_targets.to_csv(
        output_dir / "RESOLVED_TARGETS_20BP.csv", index=False, encoding="utf-8-sig"
    )
    annual_rows = []
    for year, year_daily in default_result.daily.groupby(default_result.daily["date"].str[:4]):
        annual_rows.append(
            {
                "year": int(year),
                "cost_bps": 20.0,
                **annualized_metrics(year_daily.set_index("date")["net_return"]),
            }
        )
    pd.DataFrame(annual_rows).to_csv(
        output_dir / "ANNUAL_RESULTS.csv", index=False, encoding="utf-8-sig"
    )
    full_row = summary.loc[
        summary["period"].eq("full") & summary["cost_bps"].eq(20.0)
    ].iloc[0]
    payload = {
        "status": "completed",
        "replay_kind": "frozen_sector_targets_to_real_etf_open",
        "changes_sector_ranking": False,
        "changes_sector_holding_rules": False,
        "cost_paths_bps": list(costs_bps),
        "full_20bp_total_return": float(full_row["total_ret"]),
        "full_20bp_annualized_return": float(full_row["ann_ret"]),
        "full_20bp_maximum_drawdown": float(full_row["max_drawdown"]),
        "full_20bp_average_mapping_coverage": float(
            full_row["average_mapping_coverage"]
        ),
        "price_gaps_encountered": 0,
        "point_in_time_master_available": False,
        "backtest_promotable": False,
        "blockers": [
            "mapping_coverage_insufficient",
            "etf_master_not_point_in_time",
            "sector_catalog_not_point_in_time",
        ],
    }
    (output_dir / "READINESS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_execution_open_prices() -> pd.DataFrame:
    """合并权益ETF与冻结低风险ETF的复权开盘价。"""
    equity = load_adjusted_etf_prices()[["trade_date", "ts_code", "adj_open"]]
    low_risk = _load_adjusted_prices()
    low_risk = low_risk.loc[
        low_risk["ts_code"].eq(DEFAULT_LOW_RISK_CODE),
        ["trade_date", "ts_code", "adj_open"],
    ]
    prices = pd.concat([equity, low_risk], ignore_index=True)
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["ts_code"] = prices["ts_code"].astype(str)
    prices["adj_open"] = pd.to_numeric(prices["adj_open"], errors="coerce")
    prices = prices.dropna(subset=["adj_open"])
    return prices.drop_duplicates(["trade_date", "ts_code"], keep="last")


def _turnover(pretrade: dict[str, float], target: dict[str, float]) -> float:
    assets = set(pretrade) | set(target)
    return 0.5 * sum(
        abs(float(target.get(code, 0.0)) - float(pretrade.get(code, 0.0)))
        for code in assets
    )


def _drift_weights(
    weights: dict[str, float],
    returns: dict[str, float],
    portfolio_return: float,
) -> dict[str, float]:
    gross = 1.0 + portfolio_return
    if gross <= 0:
        raise RuntimeError("ETF组合净值归零，无法继续回放")
    return {
        code: float(weight) * (1.0 + float(returns[code])) / gross
        for code, weight in weights.items()
    }


def _resolve_timeline_group(
    group: pd.DataFrame,
    active_mapping: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, float, float]:
    sector_rows = group.loc[~group["asset_code"].eq("LOW_RISK")]
    sector_targets = dict(
        zip(
            sector_rows["asset_code"].astype(str),
            sector_rows["target_weight"].astype(float),
            strict=True,
        )
    )
    explicit_low_risk = float(
        group.loc[group["asset_code"].eq("LOW_RISK"), "target_weight"].sum()
    )
    resolved = resolve_target_weights_from_active(sector_targets, active_mapping)
    target: dict[str, float] = {DEFAULT_LOW_RISK_CODE: explicit_low_risk}
    for row in resolved.itertuples(index=False):
        code = str(row.final_asset_code)
        target[code] = target.get(code, 0.0) + float(row.allocated_weight)
    target = {code: weight for code, weight in target.items() if weight > 1e-12}
    if abs(sum(target.values()) - 1.0) > 1e-9:
        raise RuntimeError("ETF目标权重不守恒")
    risk_weight = float(sum(sector_targets.values()))
    mapped_weight = float(
        resolved.loc[
            resolved["allocation_reason"].eq("mapped_equity_etf"), "allocated_weight"
        ].sum()
    )
    return target, resolved, risk_weight, mapped_weight


def run_resolved_etf_backtest(
    target_timeline: pd.DataFrame,
    mapping: pd.DataFrame,
    open_prices: pd.DataFrame,
    *,
    cost_bps: float = 20.0,
) -> EtfReplayResult:
    """将完整板块目标逐日解析为ETF，并按相邻交易日开盘价回放。"""
    required_targets = {
        "signal_date",
        "execution_date",
        "stage",
        "asset_code",
        "target_weight",
    }
    if missing := required_targets - set(target_timeline.columns):
        raise ValueError(f"目标权重时间线缺少字段: {sorted(missing)}")
    if cost_bps < 0:
        raise ValueError("cost_bps 不能为负数")
    timeline = target_timeline.loc[target_timeline["stage"].eq("executed")].copy()
    timeline["signal_date"] = timeline["signal_date"].astype(str)
    timeline["execution_date"] = timeline["execution_date"].astype(str)
    timeline["target_weight"] = pd.to_numeric(timeline["target_weight"], errors="raise")
    timeline = timeline.sort_values(["execution_date", "asset_code"])
    if timeline.empty:
        raise ValueError("没有可回放的已执行目标")
    sums = timeline.groupby("execution_date")["target_weight"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-9):
        raise RuntimeError("板块目标时间线存在权重不守恒")

    prices = open_prices.copy()
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices["ts_code"] = prices["ts_code"].astype(str)
    price_pivot = prices.pivot(index="trade_date", columns="ts_code", values="adj_open")
    groups = list(timeline.groupby("execution_date", sort=True))
    active_lookup = build_active_mapping_lookup(
        mapping, [str(execution_date) for execution_date, _ in groups]
    )
    cost_rate = cost_bps / 10_000.0
    rows: list[dict] = []
    resolved_frames: list[pd.DataFrame] = []
    equity = 1.0
    peak = 1.0
    live_weights = {DEFAULT_LOW_RISK_CODE: 1.0}

    first_date, first_group = groups[0]
    first_target, first_resolved, first_risk, first_mapped = _resolve_timeline_group(
        first_group, active_lookup.get(str(first_date), mapping.iloc[0:0])
    )
    if str(first_date) not in price_pivot.index:
        raise RuntimeError(f"{first_date} 缺少ETF开盘行情")
    if price_pivot.loc[str(first_date)].reindex(first_target).isna().any():
        raise RuntimeError(f"{first_date} 首日目标ETF缺少开盘价")
    first_turnover = _turnover(live_weights, first_target)
    first_cost = first_turnover * cost_rate
    equity *= 1.0 - first_cost
    peak = max(peak, equity)
    live_weights = first_target
    first_resolved = first_resolved.copy()
    first_resolved.insert(0, "execution_date", str(first_date))
    first_resolved.insert(0, "signal_date", str(first_group["signal_date"].iloc[0]))
    if not first_resolved.empty:
        resolved_frames.append(first_resolved)
    rows.append(
        {
            "date": str(first_date),
            "signal_date": str(first_group["signal_date"].iloc[0]),
            "gross_return": 0.0,
            "turnover": first_turnover,
            "cost": first_cost,
            "net_return": -first_cost,
            "equity": equity,
            "drawdown": equity / peak - 1.0,
            "sector_target_exposure": first_risk,
            "mapped_equity_exposure": first_mapped,
            "mapping_coverage": first_mapped / first_risk if first_risk > 1e-12 else 1.0,
            "etf_count": len(live_weights),
        }
    )

    for (prior_date, _), (execution_date, group) in zip(groups[:-1], groups[1:], strict=True):
        prior_date = str(prior_date)
        execution_date = str(execution_date)
        if prior_date not in price_pivot.index or execution_date not in price_pivot.index:
            raise RuntimeError(f"ETF行情缺少区间 {prior_date}->{execution_date}")
        start_prices = price_pivot.loc[prior_date].reindex(live_weights)
        end_prices = price_pivot.loc[execution_date].reindex(live_weights)
        if start_prices.isna().any() or end_prices.isna().any():
            missing = sorted(
                set(start_prices[start_prices.isna()].index)
                | set(end_prices[end_prices.isna()].index)
            )
            raise RuntimeError(f"{prior_date}->{execution_date} 持仓ETF缺少开盘价: {missing}")
        asset_returns = (end_prices / start_prices - 1.0).to_dict()
        gross_return = float(
            sum(live_weights[code] * asset_returns[code] for code in live_weights)
        )
        pretrade = _drift_weights(live_weights, asset_returns, gross_return)
        target, resolved, risk_weight, mapped_weight = _resolve_timeline_group(
            group, active_lookup.get(execution_date, mapping.iloc[0:0])
        )
        if price_pivot.loc[execution_date].reindex(target).isna().any():
            missing = price_pivot.loc[execution_date].reindex(target)
            raise RuntimeError(
                f"{execution_date} 新目标ETF缺少开盘价: {sorted(missing[missing.isna()].index)}"
            )
        turnover = _turnover(pretrade, target)
        cost = turnover * cost_rate
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        live_weights = target
        resolved = resolved.copy()
        resolved.insert(0, "execution_date", execution_date)
        resolved.insert(0, "signal_date", str(group["signal_date"].iloc[0]))
        if not resolved.empty:
            resolved_frames.append(resolved)
        rows.append(
            {
                "date": execution_date,
                "signal_date": str(group["signal_date"].iloc[0]),
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "sector_target_exposure": risk_weight,
                "mapped_equity_exposure": mapped_weight,
                "mapping_coverage": mapped_weight / risk_weight if risk_weight > 1e-12 else 1.0,
                "etf_count": len(live_weights),
            }
        )

    daily = pd.DataFrame(rows)
    resolved_targets = (
        pd.concat(resolved_frames, ignore_index=True)
        if resolved_frames
        else pd.DataFrame(
            columns=[
                "signal_date",
                "execution_date",
                "sector_code",
                "requested_weight",
                "mapped_etf_code",
                "fallback_code",
                "final_asset_code",
                "etf_code",
                "allocated_weight",
                "allocation_reason",
            ]
        )
    )
    metrics = annualized_metrics(daily.set_index("date")["net_return"])
    summary = {
        **metrics,
        "average_daily_turnover": float(daily["turnover"].mean()),
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "trade_day_ratio": float(daily["turnover"].gt(1e-12).mean()),
        "average_mapping_coverage": float(
            daily.loc[daily["sector_target_exposure"].gt(1e-12), "mapping_coverage"].mean()
        ),
        "minimum_mapping_coverage": float(
            daily.loc[daily["sector_target_exposure"].gt(1e-12), "mapping_coverage"].min()
        ),
    }
    return EtfReplayResult(daily=daily, resolved_targets=resolved_targets, summary=summary)


def main() -> None:
    """在独立小进程中生成ETF纯翻译回放与最新执行安全门。"""
    from ...common.paths import ensure_output_dir
    from .etf_mapping import write_latest_execution_readiness
    from .product_backtest import PRODUCT_HISTORY_START
    from .run_experiments import (
        OBSERVATION_END,
        OBSERVATION_START,
        TRAIN_END,
        VAL_END,
        VAL_START,
    )

    strategy_dir = ensure_output_dir("sector", "strategy")
    mapping_dir = ensure_output_dir("sector", "etf_mapping")
    timeline_path = strategy_dir / "TARGET_WEIGHT_TIMELINE.csv"
    mapping_path = mapping_dir / "MONTHLY_MAPPING.parquet"
    if not timeline_path.exists():
        raise FileNotFoundError("ETF回放需要产品流程生成的目标权重时间线")
    optional_inputs = {
        "monthly_mapping_missing": mapping_path,
        "equity_etf_daily_missing": ETF_DAILY_PATH,
        "equity_etf_adjustment_missing": ETF_ADJ_PATH,
    }
    missing = [name for name, path in optional_inputs.items() if not path.exists()]
    if missing:
        # 最小历史数据包可以完成板块回放；ETF实施数据缺失时只阻止执行，不拖垮主流程。
        output_dir = ensure_output_dir("sector", "etf_backtest")
        unavailable = {
            "status": "not_available",
            "replay_kind": "frozen_sector_targets_to_real_etf_open",
            "backtest_promotable": False,
            "blockers": missing,
        }
        (output_dir / "READINESS.json").write_text(
            json.dumps(unavailable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        readiness = write_latest_execution_readiness(strategy_dir, mapping_dir)
        print(
            "[etf-replay] 可选执行层数据不完整，历史板块回放已保留；"
            f"ETF执行={readiness['overall_status']}"
        )
        return
    payload = write_etf_replay_outputs(
        pd.read_csv(timeline_path, dtype={"signal_date": str, "execution_date": str}),
        pd.read_parquet(mapping_path),
        load_execution_open_prices(),
        ensure_output_dir("sector", "etf_backtest"),
        periods=(
            ("development", PRODUCT_HISTORY_START, TRAIN_END),
            ("selection", VAL_START, VAL_END),
            ("full", PRODUCT_HISTORY_START, VAL_END),
            ("observation", OBSERVATION_START, OBSERVATION_END),
        ),
    )
    readiness = write_latest_execution_readiness(strategy_dir, mapping_dir)
    print(
        "[etf-replay] "
        f"full={payload['full_20bp_total_return']:.2%}, "
        f"coverage={payload['full_20bp_average_mapping_coverage']:.2%}, "
        f"latest={readiness['overall_status']}"
    )


if __name__ == "__main__":
    main()
