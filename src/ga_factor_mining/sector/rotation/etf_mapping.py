#!/usr/bin/env python3
"""严格构造板块到可交易ETF的月度冻结映射。

当前只把名称一致视为“候选关系”，最终映射仍必须通过历史收益、
流动性和稳定性门槛。没有可靠映射的板块权重转入低风险ETF。
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import DATA_ROOT, OUTPUT_ROOT, ensure_output_dir
from .low_risk import DEFAULT_LOW_RISK_CODE
from .run_experiments import load_feature_subset


ETF_BASIC_PATH = DATA_ROOT / "sector" / "etf_basic.parquet"
ETF_DAILY_PATH = DATA_ROOT / "sector" / "equity_etf_daily.parquet"
ETF_ADJ_PATH = DATA_ROOT / "sector" / "equity_etf_adj.parquet"
MAPPING_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MappingPolicy:
    """所有阈值在观察2026结果之前冻结。"""

    lookback_sessions: int = 120
    stability_sessions: int = 60
    minimum_observations: int = 100
    minimum_listing_sessions: int = 120
    minimum_recent_completeness: float = 0.95
    minimum_median_amount_thousand_rmb: float = 50_000.0
    minimum_corr_120: float = 0.60
    minimum_corr_60: float = 0.50
    minimum_beta: float = 0.50
    maximum_beta: float = 1.50
    minimum_distinct_index_gap: float = 0.05


_DROP_WORDS = re.compile(
    r"中证|国证|上证|深证|同花顺|申万|指数|主题|行业|概念|板块|"
    r"交易型开放式|证券投资|ETF|基金|全指|精选|领先",
    flags=re.IGNORECASE,
)
_PUNCTUATION = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_LEVEL_SUFFIX = re.compile(r"(?:Ⅰ|Ⅱ|Ⅲ|I{1,3})$")


def normalize_theme_name(value: object) -> str:
    """仅做保守的公共前后缀清理，不用模糊相似度自动配对。"""
    text = "" if pd.isna(value) else str(value).strip()
    text = _PUNCTUATION.sub("", text)
    text = _DROP_WORDS.sub("", text)
    text = _LEVEL_SUFFIX.sub("", text)
    return text.lower()


def _data_signature(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            digest.update(f"missing:{path.name}".encode("utf-8"))
            continue
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_strict_candidates(
    sector_catalog: pd.DataFrame,
    etf_basic: pd.DataFrame,
    asof_date: str,
) -> pd.DataFrame:
    """生成严格名称候选；候选本身不等于已经认可的映射。"""
    required_sector = {"ts_code", "name", "type"}
    required_etf = {
        "ts_code",
        "index_code",
        "index_name",
        "list_date",
        "list_status",
        "etf_type",
    }
    if missing := required_sector - set(sector_catalog.columns):
        raise ValueError(f"板块目录缺少字段: {sorted(missing)}")
    if missing := required_etf - set(etf_basic.columns):
        raise ValueError(f"ETF目录缺少字段: {sorted(missing)}")

    sectors = sector_catalog.loc[
        sector_catalog["type"].isin(["I", "N"]), ["ts_code", "name", "type"]
    ].copy()
    sectors["normalized_name"] = sectors["name"].map(normalize_theme_name)
    sectors = sectors[sectors["normalized_name"].str.len().ge(2)]

    etfs = etf_basic.loc[
        etf_basic["etf_type"].eq("纯境内")
        & etf_basic["list_date"].fillna("99999999").astype(str).le(asof_date)
        & etf_basic["list_status"].eq("L"),
        ["ts_code", "csname", "index_code", "index_name", "list_date", "list_status"],
    ].copy()
    etfs["normalized_name"] = etfs["index_name"].map(normalize_theme_name)
    etfs = etfs[etfs["normalized_name"].str.len().ge(2)]

    candidates = sectors.merge(etfs, on="normalized_name", suffixes=("_sector", "_etf"))
    candidates = candidates.rename(
        columns={
            "ts_code_sector": "sector_code",
            "name": "sector_name",
            "type": "sector_type",
            "ts_code_etf": "etf_code",
            "csname": "etf_name",
        }
    )
    candidates["candidate_method"] = "strict_normalized_index_name"
    candidates["candidate_asof_date"] = asof_date
    columns = [
        "sector_code",
        "sector_name",
        "sector_type",
        "etf_code",
        "etf_name",
        "index_code",
        "index_name",
        "list_date",
        "list_status",
        "normalized_name",
        "candidate_method",
        "candidate_asof_date",
    ]
    return candidates[columns].sort_values(
        ["sector_code", "index_code", "list_date", "etf_code"]
    ).reset_index(drop=True)


def load_adjusted_etf_prices() -> pd.DataFrame:
    """读取权益ETF行情并构造复权开收盘价。"""
    if not ETF_DAILY_PATH.exists() or not ETF_ADJ_PATH.exists():
        missing = [str(path) for path in (ETF_DAILY_PATH, ETF_ADJ_PATH) if not path.exists()]
        raise FileNotFoundError(f"缺少权益ETF行情: {missing}")
    daily = pd.read_parquet(ETF_DAILY_PATH)
    adj = pd.read_parquet(ETF_ADJ_PATH)
    prices = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if prices["adj_factor"].isna().any():
        missing = prices.loc[prices["adj_factor"].isna(), "ts_code"].value_counts().to_dict()
        raise RuntimeError(f"权益ETF缺少复权因子: {missing}")
    for column in ("open", "close", "amount", "adj_factor"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices["adj_open"] = prices["open"] * prices["adj_factor"]
    prices["adj_close"] = prices["close"] * prices["adj_factor"]
    prices["etf_ret_1d"] = prices.groupby("ts_code", sort=False)["adj_close"].pct_change()
    return prices.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _residual(values: pd.Series, market: pd.Series) -> pd.Series:
    valid = values.notna() & market.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() < 3:
        return result
    x = np.column_stack([np.ones(int(valid.sum())), market.loc[valid].to_numpy(dtype=float)])
    y = values.loc[valid].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    result.loc[valid] = y - x @ beta
    return result


def _pair_metrics(window: pd.DataFrame, policy: MappingPolicy) -> dict[str, float]:
    valid = window[["sector_ret_1d", "etf_ret_1d", "market_ret_1d"]].dropna()
    n_obs = len(valid)
    if n_obs < policy.minimum_observations:
        return {"n_obs": n_obs}
    sector_resid = _residual(valid["sector_ret_1d"], valid["market_ret_1d"])
    etf_resid = _residual(valid["etf_ret_1d"], valid["market_ret_1d"])
    paired = pd.DataFrame({"sector": sector_resid, "etf": etf_resid}).dropna()
    if len(paired) < policy.minimum_observations or paired["sector"].std(ddof=0) == 0:
        return {"n_obs": len(paired)}
    corr120 = float(paired["sector"].corr(paired["etf"]))
    recent = paired.tail(policy.stability_sessions)
    corr60 = float(recent["sector"].corr(recent["etf"])) if len(recent) >= 40 else np.nan
    beta = float(np.cov(paired["sector"], paired["etf"], ddof=0)[0, 1] / np.var(paired["sector"]))
    fitted = beta * paired["sector"]
    residual = paired["etf"] - fitted
    total_var = float(np.var(paired["etf"]))
    r2 = 1.0 - float(np.var(residual)) / total_var if total_var > 0 else np.nan
    tracking_error = float((paired["etf"] - paired["sector"]).std(ddof=0) * np.sqrt(252.0))
    return {
        "n_obs": len(paired),
        "corr120": corr120,
        "corr60": corr60,
        "beta": beta,
        "r2": r2,
        "tracking_error": tracking_error,
    }


def build_monthly_mapping(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    candidates: pd.DataFrame,
    policy: MappingPolicy | None = None,
) -> pd.DataFrame:
    """只用月末当时可见数据构建下月生效的映射。"""
    policy = policy or MappingPolicy()
    sector_returns = panel.loc[
        panel["type"].isin(["I", "N"]), ["trade_date", "ts_code", "ret_1d"]
    ].rename(columns={"ts_code": "sector_code", "ret_1d": "sector_ret_1d"})
    market = sector_returns.groupby("trade_date", sort=True)["sector_ret_1d"].mean().rename(
        "market_ret_1d"
    )
    trading_dates = market.index.astype(str).tolist()
    month_ends = (
        pd.Series(trading_dates)
        .groupby(pd.Series(trading_dates).str[:6], sort=True)
        .last()
        .tolist()
    )
    next_date = {date: trading_dates[index + 1] for index, date in enumerate(trading_dates[:-1])}
    trading_position = {date: index for index, date in enumerate(trading_dates)}
    price_columns = ["trade_date", "ts_code", "etf_ret_1d", "amount"]
    prices = prices[price_columns].rename(columns={"ts_code": "etf_code"})
    # 预先按代码建立索引，避免每个月、每个候选反复扫描整张行情表。
    sector_by_code = {
        str(code): group.drop_duplicates("trade_date").set_index("trade_date")
        for code, group in sector_returns.groupby("sector_code", sort=False)
    }
    prices_by_code = {
        str(code): group.drop_duplicates("trade_date").set_index("trade_date")
        for code, group in prices.groupby("etf_code", sort=False)
    }
    price_dates_by_code = {
        code: group.index.astype(str).to_numpy() for code, group in prices_by_code.items()
    }
    rows: list[dict] = []

    for asof_date in month_ends:
        if asof_date not in next_date:
            continue
        end_index = trading_position[asof_date]
        start_index = max(0, end_index - policy.lookback_sessions + 1)
        window_dates = trading_dates[start_index : end_index + 1]
        effective_from = next_date[asof_date]
        candidate_now = candidates[candidates["list_date"].astype(str).le(asof_date)]
        for candidate in candidate_now.itertuples(index=False):
            sector_frame = sector_by_code.get(str(candidate.sector_code))
            etf_frame = prices_by_code.get(str(candidate.etf_code))
            if sector_frame is None or etf_frame is None:
                continue
            joined = pd.DataFrame(index=window_dates)
            joined["sector_ret_1d"] = sector_frame["sector_ret_1d"].reindex(window_dates)
            joined["etf_ret_1d"] = etf_frame["etf_ret_1d"].reindex(window_dates)
            joined["amount"] = etf_frame["amount"].reindex(window_dates)
            joined["market_ret_1d"] = market.reindex(window_dates)
            recent = joined.tail(20)
            completeness = float(recent["etf_ret_1d"].notna().mean()) if len(recent) else 0.0
            median_amount = float(recent["amount"].median()) if recent["amount"].notna().any() else np.nan
            listing_sessions = int(
                np.searchsorted(
                    price_dates_by_code[str(candidate.etf_code)], asof_date, side="right"
                )
            )
            metrics = _pair_metrics(joined, policy)
            eligible = (
                listing_sessions >= policy.minimum_listing_sessions
                and completeness >= policy.minimum_recent_completeness
                and np.isfinite(median_amount)
                and median_amount >= policy.minimum_median_amount_thousand_rmb
                and metrics.get("n_obs", 0) >= policy.minimum_observations
                and metrics.get("corr120", -np.inf) >= policy.minimum_corr_120
                and metrics.get("corr60", -np.inf) >= policy.minimum_corr_60
                and policy.minimum_beta <= metrics.get("beta", np.nan) <= policy.maximum_beta
            )
            reject_reasons = []
            if listing_sessions < policy.minimum_listing_sessions:
                reject_reasons.append("insufficient_listing_history")
            if completeness < policy.minimum_recent_completeness:
                reject_reasons.append("recent_prices_incomplete")
            if not np.isfinite(median_amount) or median_amount < policy.minimum_median_amount_thousand_rmb:
                reject_reasons.append("insufficient_liquidity")
            if metrics.get("n_obs", 0) < policy.minimum_observations:
                reject_reasons.append("insufficient_pair_observations")
            if metrics.get("corr120", -np.inf) < policy.minimum_corr_120:
                reject_reasons.append("corr120_below_threshold")
            if metrics.get("corr60", -np.inf) < policy.minimum_corr_60:
                reject_reasons.append("corr60_below_threshold")
            beta = metrics.get("beta", np.nan)
            if not np.isfinite(beta) or not policy.minimum_beta <= beta <= policy.maximum_beta:
                reject_reasons.append("beta_outside_range")
            score = (
                0.50 * metrics.get("corr120", np.nan)
                + 0.25 * metrics.get("corr60", np.nan)
                + 0.15 * metrics.get("r2", np.nan)
            )
            rows.append(
                {
                    "asof_date": asof_date,
                    "effective_from": effective_from,
                    "sector_code": candidate.sector_code,
                    "sector_name": candidate.sector_name,
                    "etf_code": candidate.etf_code,
                    "etf_name": candidate.etf_name,
                    "index_code": candidate.index_code,
                    "index_name": candidate.index_name,
                    "listing_sessions": listing_sessions,
                    "recent_completeness": completeness,
                    "median_amount20": median_amount,
                    **metrics,
                    "score_before_tracking_penalty": score,
                    "eligible_before_index_gap": bool(eligible),
                    "reject_reason": "|".join(reject_reasons),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["tracking_error_rank"] = result.groupby(
        ["asof_date", "sector_code"], sort=False
    )["tracking_error"].rank(pct=True, ascending=True)
    result["mapping_score"] = (
        result["score_before_tracking_penalty"] - 0.10 * result["tracking_error_rank"]
    )
    result["selected"] = False
    result["runner_up_index_gap"] = np.nan

    for (_, _), group in result[result["eligible_before_index_gap"]].groupby(
        ["asof_date", "sector_code"], sort=False
    ):
        best_by_index = (
            group.sort_values(["mapping_score", "median_amount20"], ascending=False)
            .drop_duplicates("index_code")
            .sort_values("mapping_score", ascending=False)
        )
        best = best_by_index.iloc[0]
        gap = (
            float(best["mapping_score"] - best_by_index.iloc[1]["mapping_score"])
            if len(best_by_index) > 1
            else np.nan
        )
        same_index = group[group["index_code"].eq(best["index_code"])].sort_values(
            ["mapping_score", "median_amount20", "etf_code"], ascending=[False, False, True]
        )
        chosen_index = same_index.index[0]
        result.loc[group.index, "runner_up_index_gap"] = gap
        if np.isnan(gap) or gap >= policy.minimum_distinct_index_gap:
            result.loc[chosen_index, "selected"] = True
        else:
            result.loc[group.index, "reject_reason"] = result.loc[
                group.index, "reject_reason"
            ].str.cat(pd.Series("distinct_index_gap_too_small", index=group.index), sep="|").str.strip("|")

    # 每月都重新验收；本月失败时不得沿用上一次成功映射。
    effective_dates = sorted(result["effective_from"].dropna().astype(str).unique())
    next_effective = {
        date: effective_dates[index + 1]
        for index, date in enumerate(effective_dates[:-1])
    }
    result["effective_to"] = result["effective_from"].astype(str).map(next_effective)
    return result.sort_values(
        ["asof_date", "sector_code", "selected", "mapping_score"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def build_active_mapping_lookup(
    mapping: pd.DataFrame,
    execution_dates: list[str],
) -> dict[str, pd.DataFrame]:
    """一次性展开执行日映射，避免逐日重复扫描整张历史表。"""
    dates = sorted(set(str(value) for value in execution_dates if value))
    buckets: dict[str, list[int]] = {date: [] for date in dates}
    if not dates or mapping.empty:
        return {date: mapping.iloc[0:0].copy() for date in dates}
    selected_flag = mapping.get("selected", pd.Series(False, index=mapping.index))
    selected = mapping.loc[selected_flag.fillna(False).astype(bool)].copy()
    for index, row in selected.iterrows():
        effective_from = str(row["effective_from"])
        effective_to = None if pd.isna(row.get("effective_to")) else str(row["effective_to"])
        asof_value = row.get("asof_date")
        asof = pd.to_datetime(
            str(asof_value), format="%Y%m%d", errors="coerce"
        ) if pd.notna(asof_value) else pd.NaT
        review_due = (
            (asof + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
            if pd.notna(asof)
            else None
        )
        start_index = bisect_left(dates, effective_from)
        stop_index = len(dates)
        if effective_to:
            stop_index = min(stop_index, bisect_left(dates, effective_to))
        if review_due:
            stop_index = min(stop_index, bisect_right(dates, review_due))
        for date in dates[start_index:stop_index]:
            buckets[date].append(index)
    lookup: dict[str, pd.DataFrame] = {}
    for date, indices in buckets.items():
        active = selected.loc[indices].copy() if indices else selected.iloc[0:0].copy()
        if not active.empty:
            active = active.sort_values(
                ["mapping_score", "median_amount20"], ascending=False
            ).drop_duplicates("sector_code")
        lookup[date] = active
    return lookup


def active_mapping_for_date(mapping: pd.DataFrame, execution_date: str) -> pd.DataFrame:
    """返回执行日有效且当月重新验收通过的唯一板块映射。"""
    return build_active_mapping_lookup(mapping, [execution_date]).get(
        execution_date, mapping.iloc[0:0].copy()
    )


def resolve_target_weights_from_active(
    target_weights: dict[str, float],
    active_mapping: pd.DataFrame,
    low_risk_code: str = DEFAULT_LOW_RISK_CODE,
) -> pd.DataFrame:
    """使用已经冻结到执行日的映射解析目标权重。"""
    active = active_mapping.set_index("sector_code", drop=False)
    # 同一ETF冲突由映射质量决定，不能依赖target_weights的插入顺序。
    mapped_candidates = []
    for sector_code, requested_weight in target_weights.items():
        if sector_code in active.index:
            row = active.loc[sector_code]
            mapped_candidates.append(
                (
                    str(sector_code),
                    float(requested_weight),
                    str(row["etf_code"]),
                    float(row["mapping_score"]),
                    float(row["median_amount20"]),
                )
            )
    mapped_candidates.sort(key=lambda item: (-item[3], -item[4], item[0]))
    mapped_winners: dict[str, str] = {}
    used_for_selection: set[str] = set()
    for sector_code, _, etf_code, _, _ in mapped_candidates:
        if etf_code not in used_for_selection:
            mapped_winners[sector_code] = etf_code
            used_for_selection.add(etf_code)

    used_etfs: set[str] = set()
    rows: list[dict] = []
    low_risk_weight = 0.0
    for sector_code, requested_weight in sorted(target_weights.items()):
        weight = float(requested_weight)
        if sector_code not in active.index:
            low_risk_weight += weight
            rows.append(
                {
                    "sector_code": sector_code,
                    "requested_weight": weight,
                    "mapped_etf_code": None,
                    "fallback_code": low_risk_code,
                    "final_asset_code": low_risk_code,
                    "etf_code": low_risk_code,
                    "allocated_weight": weight,
                    "allocation_reason": "unmapped_to_low_risk",
                }
            )
            continue
        etf_code = str(active.loc[sector_code, "etf_code"])
        if mapped_winners.get(str(sector_code)) != etf_code or etf_code in used_etfs:
            low_risk_weight += weight
            rows.append(
                {
                    "sector_code": sector_code,
                    "requested_weight": weight,
                    "mapped_etf_code": etf_code,
                    "fallback_code": low_risk_code,
                    "final_asset_code": low_risk_code,
                    "etf_code": low_risk_code,
                    "allocated_weight": weight,
                    "allocation_reason": "duplicate_etf_to_low_risk",
                }
            )
            continue
        used_etfs.add(etf_code)
        rows.append(
            {
                "sector_code": sector_code,
                "requested_weight": weight,
                "mapped_etf_code": etf_code,
                "fallback_code": None,
                "final_asset_code": etf_code,
                "etf_code": etf_code,
                "allocated_weight": weight,
                "allocation_reason": "mapped_equity_etf",
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "sector_code",
            "requested_weight",
            "mapped_etf_code",
            "fallback_code",
            "final_asset_code",
            "etf_code",
            "allocated_weight",
            "allocation_reason",
        ],
    )
    result.attrs["low_risk_weight_from_unmapped"] = low_risk_weight
    return result


def resolve_target_weights(
    target_weights: dict[str, float],
    mapping: pd.DataFrame,
    execution_date: str,
    low_risk_code: str = DEFAULT_LOW_RISK_CODE,
) -> pd.DataFrame:
    """将未映射或冲突的板块权重透明地转入低风险ETF。"""
    return resolve_target_weights_from_active(
        target_weights,
        active_mapping_for_date(mapping, execution_date),
        low_risk_code,
    )


def build_latest_execution_readiness(
    plan: dict,
    mapping: pd.DataFrame,
    *,
    equity_quote_date: str | None,
    low_risk_quote_date: str | None,
    reference_date: str | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """把指数目标解析成ETF参考组合，并对所有执行前提做硬阻断。"""
    signal_date = str(plan.get("signal_date") or "")
    execution_date = str(plan.get("planned_execution_date") or "")
    data_end = str(plan.get("market_data_asof") or "")
    reference = pd.Timestamp(reference_date or pd.Timestamp.today()).normalize()
    data_age = (
        int((reference - pd.to_datetime(data_end, format="%Y%m%d")).days)
        if data_end
        else None
    )
    targets = {str(code): float(weight) for code, weight in plan.get("target_weights", {}).items()}
    original_low_risk = float(targets.pop("LOW_RISK", 0.0))
    risk_weight = float(sum(targets.values()))

    selected = mapping[mapping.get("selected", False).astype(bool)].copy()
    latest_mapping_asof = str(selected["asof_date"].astype(str).max()) if not selected.empty else None
    latest_mapping = (
        selected[selected["asof_date"].astype(str).eq(latest_mapping_asof)].copy()
        if latest_mapping_asof
        else selected
    )
    review_due = (
        (
            pd.to_datetime(latest_mapping_asof, format="%Y%m%d") + pd.offsets.MonthEnd(1)
        ).strftime("%Y%m%d")
        if latest_mapping_asof
        else None
    )
    resolution_date = execution_date or signal_date
    resolved = (
        resolve_target_weights(targets, mapping, resolution_date)
        if targets
        else pd.DataFrame()
    )
    metadata_columns = [
        "sector_code",
        "sector_name",
        "etf_name",
        "asof_date",
        "effective_from",
        "effective_to",
        "corr120",
        "corr60",
        "beta",
        "tracking_error",
        "median_amount20",
    ]
    available_metadata = [column for column in metadata_columns if column in latest_mapping.columns]
    if not resolved.empty and available_metadata:
        latest_by_sector = latest_mapping[available_metadata].drop_duplicates("sector_code")
        resolution = resolved.merge(latest_by_sector, on="sector_code", how="left")
    else:
        resolution = resolved.copy()
    if resolution.empty:
        resolution = pd.DataFrame(
            columns=[
                "sector_code",
                "requested_weight",
                "mapped_etf_code",
                "fallback_code",
                "final_asset_code",
                "allocation_reason",
            ]
        )
    resolution.insert(0, "planned_execution_date", plan.get("planned_execution_date"))
    resolution.insert(0, "signal_date", signal_date)
    resolution["mapping_review_due"] = review_due
    resolution["latest_quote_date"] = equity_quote_date
    resolution["mapping_status"] = resolution.get(
        "allocation_reason", pd.Series(index=resolution.index, dtype=str)
    )

    portfolio_rows: list[dict] = []
    for row in resolution.to_dict("records"):
        final_code = str(row["final_asset_code"])
        fallback_weight = (
            float(row["allocated_weight"])
            if row["allocation_reason"] != "mapped_equity_etf"
            else 0.0
        )
        portfolio_rows.append(
            {
                "etf_code": final_code,
                "equity_mapping_weight": float(row["allocated_weight"]) - fallback_weight,
                "original_low_risk_weight": 0.0,
                "fallback_weight": fallback_weight,
                "source_sector": str(row["sector_code"]),
            }
        )
    if original_low_risk > 1e-12 or not portfolio_rows:
        portfolio_rows.append(
            {
                "etf_code": DEFAULT_LOW_RISK_CODE,
                "equity_mapping_weight": 0.0,
                "original_low_risk_weight": original_low_risk if portfolio_rows else 1.0,
                "fallback_weight": 0.0,
                "source_sector": "LOW_RISK",
            }
        )
    raw_portfolio = pd.DataFrame(portfolio_rows)
    portfolio = (
        raw_portfolio.groupby("etf_code", as_index=False)
        .agg(
            equity_mapping_weight=("equity_mapping_weight", "sum"),
            original_low_risk_weight=("original_low_risk_weight", "sum"),
            fallback_weight=("fallback_weight", "sum"),
            source_sectors=("source_sector", lambda values: "|".join(sorted(set(values)))),
        )
        .sort_values("etf_code")
    )
    portfolio["final_target_weight"] = portfolio[
        ["equity_mapping_weight", "original_low_risk_weight", "fallback_weight"]
    ].sum(axis=1)
    weights_valid = bool(
        (portfolio["final_target_weight"] >= -1e-12).all()
        and abs(float(portfolio["final_target_weight"].sum()) - 1.0) <= 1e-9
        and portfolio["etf_code"].is_unique
    )
    mapped_weight = float(
        resolution.loc[
            resolution.get("allocation_reason", pd.Series(index=resolution.index)).eq(
                "mapped_equity_etf"
            ),
            "allocated_weight",
        ].sum()
    ) if not resolution.empty else 0.0
    coverage = mapped_weight / risk_weight if risk_weight > 1e-12 else 1.0

    operational_blockers = []
    if data_age is None or data_age > 7:
        operational_blockers.append("strategy_data_stale")
    if signal_date != data_end:
        operational_blockers.append("signal_not_aligned_with_market_data")
    if plan.get("stage") != "planned" or not execution_date:
        operational_blockers.append("next_trade_calendar_missing")
    if risk_weight > 1e-12 and (review_due is None or execution_date > review_due):
        operational_blockers.append("mapping_stale")
    if risk_weight > 1e-12 and (not equity_quote_date or equity_quote_date < signal_date):
        operational_blockers.append("equity_etf_quotes_stale")
    if not low_risk_quote_date or low_risk_quote_date < signal_date:
        operational_blockers.append("low_risk_quote_stale")
    if risk_weight > 1e-12 and coverage < 1.0 - 1e-12:
        operational_blockers.append("mapping_coverage_incomplete")
    if not weights_valid:
        operational_blockers.append("resolved_weights_invalid")
    replay_path = OUTPUT_ROOT / "sector" / "etf_backtest" / "READINESS.json"
    research_blockers = [
        "etf_master_not_point_in_time",
        "sector_catalog_not_point_in_time",
    ]
    if not replay_path.exists():
        research_blockers.append("resolved_etf_backtest_missing")
    else:
        try:
            replay_status = json.loads(replay_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            replay_status = {}
        if replay_status.get("status") != "completed":
            research_blockers.append("resolved_etf_backtest_incomplete")
        elif not replay_status.get("backtest_promotable", False):
            research_blockers.append("resolved_etf_backtest_not_promotable")
    overall_status = "blocked" if operational_blockers else "reference_only"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "execution_ready": False,
        "strategy_data_end": data_end,
        "strategy_data_age_days": data_age,
        "signal_date": signal_date,
        "planned_execution_date": plan.get("planned_execution_date"),
        "mapping_asof": latest_mapping_asof,
        "mapping_review_due": review_due,
        "equity_quote_date": equity_quote_date,
        "low_risk_quote_date": low_risk_quote_date,
        "current_selected_sectors": int(latest_mapping["sector_code"].nunique()) if not latest_mapping.empty else 0,
        "current_distinct_etfs": int(latest_mapping["etf_code"].nunique()) if not latest_mapping.empty else 0,
        "historical_selected_rows": int(len(selected)),
        "risk_target_weight": risk_weight,
        "mapped_equity_weight": mapped_weight,
        "current_target_coverage": coverage,
        "weights_valid": weights_valid,
        "operational_blockers": operational_blockers,
        "research_blockers": research_blockers,
        "blockers": operational_blockers + research_blockers,
    }
    portfolio["status"] = overall_status
    return payload, resolution, portfolio


def write_latest_execution_readiness(
    strategy_output_dir: Path,
    mapping_output_dir: Path,
    *,
    reference_date: str | None = None,
) -> dict:
    """根据当前结构化产物生成ETF执行层状态，不重跑历史映射。"""
    plan_path = strategy_output_dir / "LATEST_PLAN.json"
    mapping_path = mapping_output_dir / "MONTHLY_MAPPING.parquet"
    if not plan_path.exists() or not mapping_path.exists():
        payload = {
            "overall_status": "blocked",
            "execution_ready": False,
            "blockers": ["latest_plan_or_mapping_missing"],
        }
        (mapping_output_dir / "ETF_EXECUTION_READINESS.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mapping = pd.read_parquet(mapping_path)
    equity_quote_date = (
        str(pd.read_parquet(ETF_DAILY_PATH, columns=["trade_date"])["trade_date"].astype(str).max())
        if ETF_DAILY_PATH.exists()
        else None
    )
    low_risk_path = DATA_ROOT / "sector" / "low_risk_fund_daily.parquet"
    low_risk_quote_date = (
        str(pd.read_parquet(low_risk_path, columns=["trade_date"])["trade_date"].astype(str).max())
        if low_risk_path.exists()
        else None
    )
    payload, resolution, portfolio = build_latest_execution_readiness(
        plan,
        mapping,
        equity_quote_date=equity_quote_date,
        low_risk_quote_date=low_risk_quote_date,
        reference_date=reference_date,
    )
    mapping_output_dir.mkdir(parents=True, exist_ok=True)
    resolution.to_csv(
        mapping_output_dir / "LATEST_MAPPING_RESOLUTION.csv", index=False, encoding="utf-8-sig"
    )
    portfolio.to_csv(
        mapping_output_dir / "RESOLVED_ETF_TARGET_PORTFOLIO.csv",
        index=False,
        encoding="utf-8-sig",
    )
    preview = portfolio.copy() if payload["overall_status"] == "ready_for_manual_review" else portfolio.iloc[0:0]
    blocked = portfolio.copy() if payload["overall_status"] != "ready_for_manual_review" else portfolio.iloc[0:0]
    preview.to_csv(mapping_output_dir / "ORDER_PREVIEW.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(mapping_output_dir / "BLOCKED_ORDERS.csv", index=False, encoding="utf-8-sig")
    (mapping_output_dir / "ETF_EXECUTION_READINESS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def build_strategy_coverage_audit(
    actions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """重建每日目标仓位，量化正式板块建议能落到ETF的比例。"""
    required = {"signal_date", "execution_date", "ts_code", "target_weight"}
    if missing := required - set(actions.columns):
        raise ValueError(f"产品动作缺少字段: {sorted(missing)}")
    rows: list[dict] = []
    for signal_date, execution_date, positions in _target_position_snapshots(actions):
        risk_weight = float(sum(positions.values()))
        resolved = resolve_target_weights(positions, mapping, str(execution_date))
        mapped_weight = float(
            resolved.loc[
                resolved["allocation_reason"].eq("mapped_equity_etf"), "allocated_weight"
            ].sum()
        )
        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": str(execution_date),
                "risk_weight": risk_weight,
                "mapped_equity_etf_weight": mapped_weight,
                "unmapped_to_low_risk_weight": risk_weight - mapped_weight,
                "risk_weight_coverage": (
                    mapped_weight / risk_weight if risk_weight > 1e-12 else np.nan
                ),
                "target_sector_count": len(positions),
                "mapped_sector_count": int(
                    resolved["allocation_reason"].eq("mapped_equity_etf").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_position_snapshots(
    actions: pd.DataFrame,
):
    positions: dict[str, float] = {}
    ordered = actions.sort_values(["execution_date", "ts_code"])
    for execution_date, day_actions in ordered.groupby("execution_date", sort=True):
        signal_date = str(day_actions["signal_date"].iloc[-1])
        for action in day_actions.itertuples(index=False):
            if action.ts_code == "LOW_RISK":
                continue
            target_weight = float(action.target_weight)
            if target_weight <= 1e-12:
                positions.pop(str(action.ts_code), None)
            else:
                positions[str(action.ts_code)] = target_weight
        yield signal_date, str(execution_date), dict(positions)


def build_strategy_allocation_audit(
    actions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """输出每个目标板块最终落到权益ETF还是低风险ETF。"""
    frames: list[pd.DataFrame] = []
    for signal_date, execution_date, positions in _target_position_snapshots(actions):
        resolved = resolve_target_weights(positions, mapping, execution_date)
        if resolved.empty:
            continue
        resolved.insert(0, "execution_date", execution_date)
        resolved.insert(0, "signal_date", signal_date)
        frames.append(resolved)
    columns = [
        "signal_date",
        "execution_date",
        "sector_code",
        "requested_weight",
        "etf_code",
        "allocated_weight",
        "allocation_reason",
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _save_fetched_frames(daily_frames: list[pd.DataFrame], adj_frames: list[pd.DataFrame]) -> None:
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    adj = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame()
    if daily.empty or adj.empty:
        raise RuntimeError("ETF行情和复权因子必须同时存在")
    daily = daily.drop_duplicates(["ts_code", "trade_date"], keep="last")
    daily = daily.sort_values(["ts_code", "trade_date"])
    adj = adj.drop_duplicates(["ts_code", "trade_date"], keep="last")
    adj = adj.sort_values(["ts_code", "trade_date"])
    # 两份行情先完整落到临时目录，再一起替换，避免中断后版本不一致。
    from .refresh_data import _transactional_replace

    with tempfile.TemporaryDirectory(prefix="etf-refresh-", dir=ETF_DAILY_PATH.parent) as temp_dir:
        staging = Path(temp_dir)
        staged_daily = staging / ETF_DAILY_PATH.name
        staged_adj = staging / ETF_ADJ_PATH.name
        daily.to_parquet(staged_daily, index=False)
        adj.to_parquet(staged_adj, index=False)
        _transactional_replace(
            [(ETF_DAILY_PATH, staged_daily), (ETF_ADJ_PATH, staged_adj)]
        )


def refresh_etf_basic(token_file: Path) -> None:
    """更新ETF目录；凭据只从仓库外文件读取。"""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - 仅数据准备环境需要
        raise RuntimeError("抓取ETF目录需要单独安装 tushare") from exc
    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Tushare token 文件为空")
    basic = ts.pro_api(token).etf_basic()
    required = {"ts_code", "index_code", "index_name", "list_date", "list_status"}
    if basic.empty or required - set(basic.columns):
        raise RuntimeError("ETF目录为空或字段不完整")
    ETF_BASIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".etf-basic-",
        suffix=".parquet",
        dir=ETF_BASIC_PATH.parent,
        delete=False,
    )
    staged = Path(handle.name)
    handle.close()
    try:
        basic.to_parquet(staged, index=False)
        os.replace(staged, ETF_BASIC_PATH)
    finally:
        staged.unlink(missing_ok=True)


def candidate_fetch_ranges(
    candidates: pd.DataFrame,
    existing_daily: pd.DataFrame,
    existing_adj: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str]]:
    """为每只ETF计算断点续传起点，并识别旧接口截断造成的头部缺口。"""
    daily_first = (
        existing_daily.assign(trade_date=existing_daily["trade_date"].astype(str))
        .groupby("ts_code")["trade_date"]
        .min()
        .to_dict()
        if not existing_daily.empty
        else {}
    )
    daily_latest = (
        existing_daily.assign(trade_date=existing_daily["trade_date"].astype(str))
        .groupby("ts_code")["trade_date"]
        .max()
        .to_dict()
        if not existing_daily.empty
        else {}
    )
    adj_first = (
        existing_adj.assign(trade_date=existing_adj["trade_date"].astype(str))
        .groupby("ts_code")["trade_date"]
        .min()
        .to_dict()
        if not existing_adj.empty
        else {}
    )
    adj_latest = (
        existing_adj.assign(trade_date=existing_adj["trade_date"].astype(str))
        .groupby("ts_code")["trade_date"]
        .max()
        .to_dict()
        if not existing_adj.empty
        else {}
    )
    ranges = []
    candidate_rows = candidates.copy()
    candidate_rows["etf_code"] = candidate_rows["etf_code"].astype(str)
    listing_dates = (
        candidate_rows.dropna(subset=["list_date"])
        .groupby("etf_code")["list_date"]
        .min()
        .astype(str)
        .to_dict()
        if "list_date" in candidate_rows.columns
        else {}
    )
    for code in sorted(set(candidate_rows["etf_code"])):
        expected_start = max(str(start_date), listing_dates.get(code, str(start_date)))
        # 任一配对文件起点偏晚都代表复权开盘序列存在头部缺口。
        first_available = max(
            daily_first.get(code, "99999999"),
            adj_first.get(code, "99999999"),
        )
        expected_timestamp = pd.to_datetime(expected_start, format="%Y%m%d", errors="coerce")
        first_timestamp = pd.to_datetime(first_available, format="%Y%m%d", errors="coerce")
        leading_gap = code in listing_dates and (
            pd.isna(first_timestamp)
            or pd.isna(expected_timestamp)
            or (first_timestamp - expected_timestamp).days > 14
        )
        overlap = min(daily_latest.get(code, expected_start), adj_latest.get(code, expected_start))
        fetch_start = expected_start if leading_gap else max(expected_start, str(overlap))
        if fetch_start <= end_date:
            ranges.append((code, fetch_start))
    return ranges


def calendar_year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """把长区间拆为自然年，避开行情接口单次返回行数上限。"""
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    ranges = []
    cursor = start
    while cursor <= end:
        stop = min(pd.Timestamp(year=cursor.year, month=12, day=31), end)
        ranges.append((cursor.strftime("%Y%m%d"), stop.strftime("%Y%m%d")))
        cursor = stop + pd.Timedelta(days=1)
    return ranges


def fetch_candidate_history(
    candidates: pd.DataFrame,
    token_file: Path,
    start_date: str,
    end_date: str,
    call_interval: float = 0.25,
) -> None:
    """分批抓取严格候选ETF行情，支持已有文件断点续抓。"""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - 仅数据准备环境需要
        raise RuntimeError("抓取ETF数据需要单独安装 tushare") from exc
    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Tushare token 文件为空")
    pro = ts.pro_api(token)
    existing_daily = pd.read_parquet(ETF_DAILY_PATH) if ETF_DAILY_PATH.exists() else pd.DataFrame()
    existing_adj = pd.read_parquet(ETF_ADJ_PATH) if ETF_ADJ_PATH.exists() else pd.DataFrame()
    daily_frames = [existing_daily] if not existing_daily.empty else []
    adj_frames = [existing_adj] if not existing_adj.empty else []
    fetch_ranges = candidate_fetch_ranges(
        candidates,
        existing_daily,
        existing_adj,
        start_date,
        end_date,
    )
    for index, (code, fetch_start) in enumerate(fetch_ranges, start=1):
        daily_parts = []
        adj_parts = []
        for chunk_start, chunk_end in calendar_year_ranges(fetch_start, end_date):
            daily_chunk = pro.fund_daily(
                ts_code=code, start_date=chunk_start, end_date=chunk_end
            )
            time.sleep(call_interval)
            adj_chunk = pro.fund_adj(
                ts_code=code, start_date=chunk_start, end_date=chunk_end
            )
            time.sleep(call_interval)
            if not daily_chunk.empty:
                daily_parts.append(daily_chunk)
            if not adj_chunk.empty:
                adj_parts.append(adj_chunk)
        daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
        adj = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()
        if daily.empty or adj.empty:
            raise RuntimeError(f"ETF {code} 行情或复权因子为空")
        daily_frames.append(daily)
        adj_frames.append(adj)
        if index % 20 == 0 or index == len(fetch_ranges):
            _save_fetched_frames(daily_frames, adj_frames)
            print(f"[etf-data] {index}/{len(fetch_ranges)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof-date", default="20260529")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default="20260529")
    args = parser.parse_args()

    if args.fetch:
        if args.token_file is None:
            parser.error("--fetch 时必须提供 --token-file")
        refresh_etf_basic(args.token_file)
    sector_catalog = pd.read_parquet(DATA_ROOT / "sector" / "ths_index.parquet")
    etf_basic = pd.read_parquet(ETF_BASIC_PATH)
    candidates = build_strict_candidates(sector_catalog, etf_basic, args.asof_date)
    if args.fetch:
        fetch_candidate_history(candidates, args.token_file, args.start_date, args.end_date)

    output_dir = ensure_output_dir("sector", "etf_mapping")
    candidates.to_csv(output_dir / "CANDIDATES.csv", index=False, encoding="utf-8-sig")
    policy = MappingPolicy()
    readiness = {
        "mapping_protocol_version": MAPPING_PROTOCOL_VERSION,
        "candidate_asof_date": args.asof_date,
        "candidate_method": "strict normalized tracked-index name; candidate is not an accepted mapping",
        "candidate_sectors": int(candidates["sector_code"].nunique()),
        "candidate_etfs": int(candidates["etf_code"].nunique()),
        "equity_etf_daily_available": ETF_DAILY_PATH.exists(),
        "equity_etf_adj_available": ETF_ADJ_PATH.exists(),
        "etf_master_pit_complete": False,
        "sector_catalog_pit_complete": False,
        "backtest_promotable": False,
        "unmapped_policy": f"allocate to {DEFAULT_LOW_RISK_CODE}",
        "policy": asdict(policy),
        "data_signature": _data_signature((ETF_BASIC_PATH, ETF_DAILY_PATH, ETF_ADJ_PATH)),
    }
    if ETF_DAILY_PATH.exists() and ETF_ADJ_PATH.exists():
        # 映射只需要日收益，禁止加载整张特征面板造成无意义的内存峰值。
        panel = load_feature_subset({"trade_date", "ts_code", "type", "ret_1d"})
        prices = load_adjusted_etf_prices()
        mapping = build_monthly_mapping(panel, prices, candidates, policy)
        mapping.to_parquet(output_dir / "MONTHLY_MAPPING.parquet", index=False)
        mapping[mapping["selected"]].to_csv(
            output_dir / "SELECTED_MAPPING.csv", index=False, encoding="utf-8-sig"
        )
        readiness["selected_mappings"] = int(mapping["selected"].sum())
        readiness["selected_sectors"] = int(
            mapping.loc[mapping["selected"], "sector_code"].nunique()
        )
        action_paths = (
            OUTPUT_ROOT / "sector" / "strategy" / "selection_actions.parquet",
            OUTPUT_ROOT / "sector" / "strategy" / "observation_actions.parquet",
        )
        if all(path.exists() for path in action_paths):
            actions = pd.concat([pd.read_parquet(path) for path in action_paths], ignore_index=True)
            coverage = build_strategy_coverage_audit(actions, mapping)
            coverage.to_csv(output_dir / "STRATEGY_COVERAGE.csv", index=False, encoding="utf-8-sig")
            allocations = build_strategy_allocation_audit(actions, mapping)
            allocations.to_parquet(output_dir / "ALLOCATION_AUDIT.parquet", index=False)
            positive_risk = coverage[coverage["risk_weight"].gt(1e-12)]
            readiness["strategy_coverage_period"] = [
                str(coverage["execution_date"].min()),
                str(coverage["execution_date"].max()),
            ]
            readiness["strategy_risk_weighted_mapping_coverage"] = float(
                positive_risk["mapped_equity_etf_weight"].sum()
                / positive_risk["risk_weight"].sum()
            )
            readiness["strategy_zero_mapping_days"] = int(
                positive_risk["mapped_equity_etf_weight"].le(1e-12).sum()
            )
            readiness["strategy_positive_risk_days"] = int(len(positive_risk))
    (output_dir / "READINESS.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] candidate_sectors={readiness['candidate_sectors']}")


if __name__ == "__main__":
    main()
