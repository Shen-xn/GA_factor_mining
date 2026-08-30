#!/usr/bin/env python3
"""严格构造板块到可交易ETF的月度冻结映射。

当前只把名称一致视为“候选关系”，最终映射仍必须通过历史收益、
流动性和稳定性门槛。没有可靠映射的板块权重转入低风险ETF。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import DATA_ROOT, OUTPUT_ROOT, ensure_output_dir
from .low_risk import DEFAULT_LOW_RISK_CODE
from .run_experiments import load_or_build_features


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


def resolve_target_weights(
    target_weights: dict[str, float],
    mapping: pd.DataFrame,
    signal_date: str,
    low_risk_code: str = DEFAULT_LOW_RISK_CODE,
) -> pd.DataFrame:
    """将未映射或冲突的板块权重透明地转入低风险ETF。"""
    active = mapping[
        mapping.get("selected", False).astype(bool)
        & mapping["effective_from"].astype(str).le(signal_date)
        & (mapping["effective_to"].isna() | mapping["effective_to"].astype(str).gt(signal_date))
    ].copy()
    active = active.sort_values(["mapping_score", "median_amount20"], ascending=False)
    active = active.drop_duplicates("sector_code").set_index("sector_code", drop=False)
    used_etfs: set[str] = set()
    rows: list[dict] = []
    low_risk_weight = 0.0
    for sector_code, requested_weight in target_weights.items():
        weight = float(requested_weight)
        if sector_code not in active.index:
            low_risk_weight += weight
            rows.append(
                {
                    "sector_code": sector_code,
                    "requested_weight": weight,
                    "etf_code": low_risk_code,
                    "allocated_weight": weight,
                    "allocation_reason": "unmapped_to_low_risk",
                }
            )
            continue
        etf_code = str(active.loc[sector_code, "etf_code"])
        if etf_code in used_etfs:
            low_risk_weight += weight
            rows.append(
                {
                    "sector_code": sector_code,
                    "requested_weight": weight,
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
            "etf_code",
            "allocated_weight",
            "allocation_reason",
        ],
    )
    result.attrs["low_risk_weight_from_unmapped"] = low_risk_weight
    return result


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
    if not daily.empty:
        daily = daily.drop_duplicates(["ts_code", "trade_date"], keep="last")
        daily = daily.sort_values(["ts_code", "trade_date"])
        daily.to_parquet(ETF_DAILY_PATH, index=False)
    if not adj.empty:
        adj = adj.drop_duplicates(["ts_code", "trade_date"], keep="last")
        adj = adj.sort_values(["ts_code", "trade_date"])
        adj.to_parquet(ETF_ADJ_PATH, index=False)


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
    complete_codes = set(existing_daily.get("ts_code", pd.Series(dtype=str)).astype(str)) & set(
        existing_adj.get("ts_code", pd.Series(dtype=str)).astype(str)
    )
    codes = sorted(set(candidates["etf_code"].astype(str)) - complete_codes)
    for index, code in enumerate(codes, start=1):
        daily = pro.fund_daily(ts_code=code, start_date=start_date, end_date=end_date)
        time.sleep(call_interval)
        adj = pro.fund_adj(ts_code=code, start_date=start_date, end_date=end_date)
        time.sleep(call_interval)
        if daily.empty or adj.empty:
            raise RuntimeError(f"ETF {code} 行情或复权因子为空")
        daily_frames.append(daily)
        adj_frames.append(adj)
        if index % 20 == 0 or index == len(codes):
            _save_fetched_frames(daily_frames, adj_frames)
            print(f"[etf-data] {index}/{len(codes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof-date", default="20260529")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default="20260529")
    args = parser.parse_args()

    sector_catalog = pd.read_parquet(DATA_ROOT / "sector" / "ths_index.parquet")
    etf_basic = pd.read_parquet(ETF_BASIC_PATH)
    candidates = build_strict_candidates(sector_catalog, etf_basic, args.asof_date)
    if args.fetch:
        if args.token_file is None:
            parser.error("--fetch 时必须提供 --token-file")
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
        panel = load_or_build_features()
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
