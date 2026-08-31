#!/usr/bin/env python3
"""生成不改变正式交易路径的大盘与ETF参考数据。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import DATA_ROOT, ensure_output_dir
from .etf_mapping import (
    ETF_ADJ_PATH,
    ETF_BASIC_PATH,
    ETF_DAILY_PATH,
    MappingPolicy,
    _pair_metrics,
    load_adjusted_etf_prices,
    normalize_theme_name,
)
from .risk import REGIME_ORDER, classify_market, market_risk_components
from .run_experiments import load_feature_subset


INDEX_DAILY_PATH = DATA_ROOT / "sector" / "market_index_daily.parquet"
SW_L1_DAILY_PATH = DATA_ROOT / "sector" / "sw_l1_daily.parquet"
SECTOR_CATALOG_PATH = DATA_ROOT / "sector" / "ths_index.parquet"


def build_broad_market_diagnostic_state(
    index_daily: pd.DataFrame,
    industry_daily: pd.DataFrame,
) -> pd.DataFrame:
    """用宽基趋势/波动和申万一级宽度构造独立大盘诊断。"""
    required = {"ts_code", "trade_date", "close"}
    if missing := required - set(index_daily.columns):
        raise ValueError(f"宽基行情缺少字段: {sorted(missing)}")
    if missing := required - set(industry_daily.columns):
        raise ValueError(f"申万行业行情缺少字段: {sorted(missing)}")
    index_close = index_daily.pivot(
        index="trade_date", columns="ts_code", values="close"
    ).sort_index()
    industry_close = industry_daily.pivot(
        index="trade_date", columns="ts_code", values="close"
    ).sort_index()
    dates = index_close.index.intersection(industry_close.index)
    if dates.empty:
        return pd.DataFrame()
    index_close = index_close.reindex(dates)
    industry_close = industry_close.reindex(dates)
    index_ret1 = index_close.pct_change(fill_method=None)
    industry_ret20 = industry_close / industry_close.shift(20) - 1.0
    industry_ret60 = industry_close / industry_close.shift(60) - 1.0
    daily = pd.DataFrame(index=dates)
    daily["benchmark_ret_1d"] = index_ret1.mean(axis=1)
    daily["breadth_positive_20d"] = (
        industry_ret20.gt(0).sum(axis=1) / industry_ret20.notna().sum(axis=1)
    )
    daily["breadth_positive_60d"] = (
        industry_ret60.gt(0).sum(axis=1) / industry_ret60.notna().sum(axis=1)
    )
    daily["risk_breadth_positive_20d"] = daily["breadth_positive_20d"]
    daily["risk_breadth_positive_60d"] = daily["breadth_positive_60d"]
    daily["breadth_20d_valid_count"] = industry_ret20.notna().sum(axis=1)
    daily["breadth_60d_valid_count"] = industry_ret60.notna().sum(axis=1)
    daily["sector_count"] = industry_close.notna().sum(axis=1)
    daily["breadth_20d_coverage"] = daily["breadth_20d_valid_count"] / daily["sector_count"]
    daily["breadth_60d_coverage"] = daily["breadth_60d_valid_count"] / daily["sector_count"]
    daily["benchmark_equity"] = (1.0 + daily["benchmark_ret_1d"].fillna(0.0)).cumprod()
    daily["benchmark_trend_60d"] = daily["benchmark_equity"] / daily["benchmark_equity"].shift(60) - 1.0
    daily["market_volatility_20d"] = daily["benchmark_ret_1d"].rolling(
        20, min_periods=15
    ).std()

    def last_percentile(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        return float((valid <= valid[-1]).mean()) if len(valid) else np.nan

    daily["market_vol_percentile"] = daily["market_volatility_20d"].rolling(
        252, min_periods=60
    ).apply(last_percentile, raw=True)
    components = daily.apply(market_risk_components, axis=1, result_type="expand")
    daily = pd.concat([daily, components], axis=1).reset_index()
    daily["raw_regime"] = daily.apply(classify_market, axis=1)
    return daily


def build_latest_proxy_candidates(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    etf_basic: pd.DataFrame,
    sector_catalog: pd.DataFrame,
    target_weights: dict[str, float],
    asof_date: str,
    *,
    top_n: int = 5,
    policy: MappingPolicy | None = None,
) -> pd.DataFrame:
    """为最新目标列出人工复核候选，永远不自动转成正式映射。"""
    if top_n <= 0:
        raise ValueError("top_n 必须大于0")
    policy = policy or MappingPolicy()
    targets = {
        str(code): float(weight)
        for code, weight in target_weights.items()
        if str(code) != "LOW_RISK" and float(weight) > 0.0
    }
    if not targets:
        return pd.DataFrame()
    required_frames = (
        ("板块面板", {"trade_date", "ts_code", "type", "ret_1d"}, panel),
        ("ETF行情", {"trade_date", "ts_code", "etf_ret_1d", "amount"}, prices),
        (
            "ETF目录",
            {"ts_code", "csname", "index_name", "list_date", "list_status", "etf_type"},
            etf_basic,
        ),
        ("板块目录", {"ts_code", "name"}, sector_catalog),
    )
    for label, required, frame in required_frames:
        if missing := required - set(frame.columns):
            raise ValueError(f"{label}缺少字段: {sorted(missing)}")

    sector_returns = panel.loc[
        panel["type"].isin(["I", "N"])
        & panel["trade_date"].astype(str).le(asof_date),
        ["trade_date", "ts_code", "ret_1d"],
    ].copy()
    sector_returns["trade_date"] = sector_returns["trade_date"].astype(str)
    market = sector_returns.groupby("trade_date", sort=True)["ret_1d"].mean().rename(
        "market_ret_1d"
    )
    window_dates = market.index.astype(str).tolist()[-policy.lookback_sessions :]
    sector_by_code = {
        str(code): group.drop_duplicates("trade_date").set_index("trade_date")
        for code, group in sector_returns[sector_returns["ts_code"].isin(targets)].groupby(
            "ts_code", sort=False
        )
    }
    catalog_names = sector_catalog.drop_duplicates("ts_code").set_index("ts_code")["name"]
    basic = etf_basic.loc[
        etf_basic["etf_type"].eq("纯境内")
        & etf_basic["list_status"].eq("L")
        & etf_basic["list_date"].fillna("99999999").astype(str).le(asof_date)
    ].drop_duplicates("ts_code")
    prices = prices.loc[
        prices["trade_date"].astype(str).le(asof_date)
        & prices["ts_code"].isin(basic["ts_code"]),
        ["trade_date", "ts_code", "etf_ret_1d", "amount"],
    ].copy()
    prices["trade_date"] = prices["trade_date"].astype(str)
    price_by_code = {
        str(code): group.drop_duplicates("trade_date").set_index("trade_date")
        for code, group in prices.groupby("ts_code", sort=False)
    }
    rows: list[dict] = []
    for sector_code, target_weight in targets.items():
        sector_frame = sector_by_code.get(sector_code)
        if sector_frame is None:
            continue
        sector_name = str(catalog_names.get(sector_code, ""))
        sector_normalized = normalize_theme_name(sector_name)
        for etf in basic.itertuples(index=False):
            etf_frame = price_by_code.get(str(etf.ts_code))
            if etf_frame is None:
                continue
            joined = pd.DataFrame(index=window_dates)
            joined["sector_ret_1d"] = sector_frame["ret_1d"].reindex(window_dates)
            joined["etf_ret_1d"] = etf_frame["etf_ret_1d"].reindex(window_dates)
            joined["amount"] = etf_frame["amount"].reindex(window_dates)
            joined["market_ret_1d"] = market.reindex(window_dates)
            metrics = _pair_metrics(joined, policy)
            if metrics.get("n_obs", 0) < policy.minimum_observations:
                continue
            recent = joined.tail(20)
            completeness = float(recent["etf_ret_1d"].notna().mean())
            median_amount = float(recent["amount"].median())
            index_name = str(etf.index_name)
            index_normalized = normalize_theme_name(index_name)
            semantic_overlap = bool(
                min(len(sector_normalized), len(index_normalized)) >= 2
                and (sector_normalized in index_normalized or index_normalized in sector_normalized)
            )
            statistical_gate = bool(
                completeness >= policy.minimum_recent_completeness
                and np.isfinite(median_amount)
                and median_amount >= policy.minimum_median_amount_thousand_rmb
                and metrics.get("corr120", -np.inf) >= policy.minimum_corr_120
                and metrics.get("corr60", -np.inf) >= policy.minimum_corr_60
                and policy.minimum_beta
                <= metrics.get("beta", np.nan)
                <= policy.maximum_beta
            )
            if statistical_gate and semantic_overlap:
                review_status = "strong_candidate_requires_manual_review"
            elif metrics.get("corr120", -np.inf) >= 0.40 and metrics.get("corr60", -np.inf) >= 0.30:
                review_status = "weak_candidate"
            else:
                review_status = "no_reliable_proxy"
            rows.append(
                {
                    "signal_date": asof_date,
                    "sector_code": sector_code,
                    "sector_name": sector_name,
                    "target_weight": target_weight,
                    "etf_code": str(etf.ts_code),
                    "etf_name": str(etf.csname),
                    "index_name": index_name,
                    **metrics,
                    "median_amount20": median_amount,
                    "recent_completeness": completeness,
                    "semantic_name_overlap": semantic_overlap,
                    "statistical_gate_pass": statistical_gate,
                    "manual_review_status": review_status,
                    "automatic_mapping_eligible": False,
                    "point_in_time_backtest_eligible": False,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    display = result.loc[
        result["recent_completeness"].ge(policy.minimum_recent_completeness)
        & result["median_amount20"].ge(policy.minimum_median_amount_thousand_rmb)
        & result["manual_review_status"].ne("no_reliable_proxy")
        & result["semantic_name_overlap"]
    ].copy()
    if display.empty:
        return display.reset_index(drop=True)
    display = display.sort_values(
        ["sector_code", "corr120", "corr60", "median_amount20"],
        ascending=[True, False, False, False],
    )
    display["display_rank"] = display.groupby("sector_code", sort=False).cumcount() + 1
    return display.loc[display["display_rank"].le(top_n)].reset_index(drop=True)


def write_latest_broad_market_risk(strategy_dir: Path) -> dict:
    """输出真正宽基含义的大盘诊断；不反向改写正式仓位。"""
    paths = (INDEX_DAILY_PATH, SW_L1_DAILY_PATH)
    if any(not path.exists() for path in paths):
        payload = {
            "status": "not_available",
            "diagnostic_only": True,
            "missing": [str(path) for path in paths if not path.exists()],
        }
        (strategy_dir / "LATEST_BROAD_MARKET_RISK.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    indices = pd.read_parquet(
        INDEX_DAILY_PATH, columns=["ts_code", "trade_date", "close"]
    )
    industries = pd.read_parquet(
        SW_L1_DAILY_PATH, columns=["ts_code", "trade_date", "close"]
    )
    state = build_broad_market_diagnostic_state(indices, industries)
    if state.empty:
        raise RuntimeError("宽基与申万一级行情没有共同交易日")
    latest = state.iloc[-1]
    formal_path = strategy_dir / "LATEST_MARKET_RISK.json"
    formal = (
        json.loads(formal_path.read_text(encoding="utf-8"))
        if formal_path.exists()
        else {}
    )
    formal_regime = str(formal.get("confirmed_regime", ""))
    broad_regime = str(latest["raw_regime"])
    if formal_regime in REGIME_ORDER:
        level_gap = REGIME_ORDER[broad_regime] - REGIME_ORDER[formal_regime]
        relation = "more_optimistic" if level_gap > 0 else "more_defensive" if level_gap < 0 else "agrees"
    else:
        level_gap = None
        relation = "formal_state_unavailable"
    payload = {
        "status": "ready" if latest["risk_data_quality"] == "complete" else "insufficient",
        "diagnostic_only": True,
        "formal_exposure_effect": "none_after_development_replacement_and_overlay_failed",
        "risk_asof_date": str(latest["trade_date"]),
        "risk_score": float(latest["risk_score"]),
        "risk_data_quality": str(latest["risk_data_quality"]),
        "raw_regime": broad_regime,
        "benchmark_trend_60d": float(latest["benchmark_trend_60d"]),
        "breadth_positive_20d": float(latest["breadth_positive_20d"]),
        "breadth_positive_60d": float(latest["breadth_positive_60d"]),
        "market_vol_percentile": float(latest["market_vol_percentile"]),
        "industry_count": int(latest["sector_count"]),
        "breadth_20d_coverage": float(latest["breadth_20d_coverage"]),
        "breadth_60d_coverage": float(latest["breadth_60d_coverage"]),
        "formal_confirmed_regime": formal_regime or None,
        "formal_risk_target_exposure": formal.get("risk_target_exposure"),
        "relation_to_formal_state": relation,
        "regime_level_gap": level_gap,
        "reason": "五个宽基指数衡量趋势和波动，31个申万一级行业衡量上涨宽度；仅供人工复核，不生成或放大交易指令。",
    }
    pd.DataFrame([payload]).to_csv(
        strategy_dir / "LATEST_BROAD_MARKET_RISK.csv", index=False, encoding="utf-8-sig"
    )
    (strategy_dir / "LATEST_BROAD_MARKET_RISK.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_latest_proxy_reference(strategy_dir: Path, mapping_dir: Path) -> dict:
    """输出最新目标的统计代理候选，正式映射和订单安全门保持不变。"""
    plan_path = strategy_dir / "LATEST_PLAN.json"
    paths = (plan_path, ETF_BASIC_PATH, ETF_DAILY_PATH, ETF_ADJ_PATH, SECTOR_CATALOG_PATH)
    if any(not path.exists() for path in paths):
        payload = {
            "status": "not_available",
            "reference_only": True,
            "missing": [str(path) for path in paths if not path.exists()],
            "automatic_mapping_changed": False,
            "orders_generated": False,
        }
        (mapping_dir / "LATEST_PROXY_SUMMARY.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    signal_date = str(plan["signal_date"])
    panel = load_feature_subset({"trade_date", "ts_code", "type", "ret_1d"})
    candidates = build_latest_proxy_candidates(
        panel,
        load_adjusted_etf_prices(),
        pd.read_parquet(ETF_BASIC_PATH),
        pd.read_parquet(SECTOR_CATALOG_PATH),
        plan.get("target_weights", {}),
        signal_date,
    )
    candidate_path = mapping_dir / "LATEST_PROXY_CANDIDATES.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    target_count = sum(
        str(code) != "LOW_RISK" and float(weight) > 0.0
        for code, weight in plan.get("target_weights", {}).items()
    )
    strong = (
        candidates["manual_review_status"].eq("strong_candidate_requires_manual_review")
        if not candidates.empty
        else pd.Series(dtype=bool)
    )
    represented = set(candidates["sector_code"].astype(str)) if not candidates.empty else set()
    target_codes = [
        str(code)
        for code, weight in plan.get("target_weights", {}).items()
        if str(code) != "LOW_RISK" and float(weight) > 0.0
    ]
    payload = {
        "status": "reference_only",
        "reference_only": True,
        "signal_date": signal_date,
        "target_sector_count": int(target_count),
        "candidate_rows": int(len(candidates)),
        "sectors_with_strong_manual_review_candidate": int(
            candidates.loc[strong, "sector_code"].nunique() if not candidates.empty else 0
        ),
        "sectors_without_displayable_proxy": [
            code for code in target_codes if code not in represented
        ],
        "automatic_mapping_changed": False,
        "orders_generated": False,
        "latest_actions_modified": False,
        "warning": "统计相似不等于跟踪同一资产；必须核对ETF成分、指数定义和折溢价，不能把本表直接当作订单。",
    }
    (mapping_dir / "LATEST_PROXY_SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    strategy_dir = ensure_output_dir("sector", "strategy")
    mapping_dir = ensure_output_dir("sector", "etf_mapping")
    broad = write_latest_broad_market_risk(strategy_dir)
    proxy = write_latest_proxy_reference(strategy_dir, mapping_dir)
    print(
        f"[reference] broad={broad['status']} proxy={proxy['status']} "
        f"candidates={proxy.get('candidate_rows', 0)}"
    )


if __name__ == "__main__":
    main()
