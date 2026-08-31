"""预登记语义代理ETF研究；结果不会自动进入正式映射。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ...common.paths import CONFIG_ROOT, DATA_ROOT, ensure_output_dir
from .etf_mapping import (
    ETF_BASIC_PATH,
    MappingPolicy,
    build_monthly_mapping,
    fetch_candidate_history,
    load_adjusted_etf_prices,
    normalize_theme_name,
)
from .run_experiments import load_feature_subset


CONFIG_PATH = CONFIG_ROOT / "sector" / "etf_proxy_hypotheses.json"
SELECTION_END = "20251231"


def load_registered_proxy_candidates(config_path: Path = CONFIG_PATH) -> tuple[dict, pd.DataFrame]:
    """读取并核对预登记代理；退市、代码或名称不一致时立即失败。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hypotheses = pd.DataFrame(config.get("hypotheses", []))
    required = {"sector_code", "sector_name", "etf_code", "etf_name", "semantic_basis", "status"}
    if hypotheses.empty or (missing := required - set(hypotheses.columns)):
        raise ValueError(f"代理ETF配置为空或缺少字段: {sorted(missing) if hypotheses.size else sorted(required)}")
    if hypotheses[["sector_code", "etf_code"]].duplicated().any():
        raise ValueError("代理ETF配置存在重复板块-ETF关系")

    sector_catalog = pd.read_parquet(DATA_ROOT / "sector" / "ths_index.parquet")
    sector_catalog = sector_catalog[["ts_code", "name", "type"]].rename(
        columns={"ts_code": "sector_code", "name": "catalog_sector_name", "type": "sector_type"}
    )
    etf_basic = pd.read_parquet(ETF_BASIC_PATH)[
        ["ts_code", "csname", "index_code", "index_name", "list_date", "list_status", "etf_type"]
    ].rename(columns={"ts_code": "etf_code", "csname": "catalog_etf_name"})
    candidates = hypotheses.merge(sector_catalog, on="sector_code", how="left", validate="many_to_one")
    candidates = candidates.merge(etf_basic, on="etf_code", how="left", validate="many_to_one")
    invalid = candidates[
        candidates["catalog_sector_name"].isna()
        | candidates["catalog_etf_name"].isna()
        | ~candidates["list_status"].eq("L")
        | ~candidates["etf_type"].eq("纯境内")
        | ~candidates["sector_name"].eq(candidates["catalog_sector_name"])
        | ~candidates["etf_name"].eq(candidates["catalog_etf_name"])
    ]
    if not invalid.empty:
        raise RuntimeError(f"代理ETF目录核验失败: {invalid[['sector_code', 'etf_code']].to_dict('records')}")
    candidates["normalized_name"] = candidates["sector_name"].map(normalize_theme_name)
    candidates["candidate_method"] = "registered_semantic_proxy"
    candidates["candidate_asof_date"] = str(config["registered_at"]).replace("-", "")
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
        "semantic_basis",
        "status",
    ]
    return config, candidates[columns].sort_values(["sector_code", "etf_code"]).reset_index(drop=True)


def _latest_pair_snapshot(mapping: pd.DataFrame, cutoff: str, label: str) -> pd.DataFrame:
    eligible = mapping.loc[mapping["asof_date"].astype(str).le(cutoff)].copy()
    if eligible.empty:
        return eligible
    latest = eligible.groupby(["sector_code", "etf_code"])["asof_date"].transform("max")
    result = eligible.loc[eligible["asof_date"].eq(latest)].copy()
    result.insert(0, "snapshot", label)
    return result


def run_research(output_dir: Path) -> dict:
    config, candidates = load_registered_proxy_candidates()
    panel = load_feature_subset({"trade_date", "ts_code", "type", "ret_1d"})
    prices = load_adjusted_etf_prices()
    mapping = build_monthly_mapping(panel, prices, candidates, MappingPolicy())
    if mapping.empty:
        raise RuntimeError("代理ETF没有形成任何可验证月份")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "REGISTERED_HYPOTHESES.csv", index=False, encoding="utf-8-sig")
    mapping.to_parquet(output_dir / "MONTHLY_DIAGNOSTIC.parquet", index=False)
    mapping.to_csv(output_dir / "MONTHLY_DIAGNOSTIC.csv", index=False, encoding="utf-8-sig")
    snapshots = pd.concat(
        [
            _latest_pair_snapshot(mapping, SELECTION_END, "through_2025_diagnostic"),
            _latest_pair_snapshot(mapping, str(mapping["asof_date"].max()), "latest_observation"),
        ],
        ignore_index=True,
    )
    snapshot_columns = [
        "snapshot",
        "asof_date",
        "sector_code",
        "sector_name",
        "etf_code",
        "etf_name",
        "corr120",
        "corr60",
        "beta",
        "tracking_error",
        "median_amount20",
        "eligible_before_index_gap",
        "selected",
        "reject_reason",
    ]
    snapshots[snapshot_columns].to_csv(
        output_dir / "LATEST_DIAGNOSTICS.csv", index=False, encoding="utf-8-sig"
    )
    historical = snapshots.loc[snapshots["snapshot"].eq("through_2025_diagnostic")]
    payload = {
        "status": "completed",
        "registered_at": config["registered_at"],
        "registered_after_observation": bool(config["registered_after_observation"]),
        "selection_end": SELECTION_END,
        "historical_results_are_diagnostic_only": True,
        "forward_evidence_eligible_after": config["forward_evidence_eligible_after"],
        "hypothesis_count": int(len(candidates)),
        "historical_pair_gate_pass_count": int(historical["eligible_before_index_gap"].sum()),
        "historical_selected_pair_count": int(historical["selected"].sum()),
        "default_mapping_modified": False,
        "promotion": "rejected_pending_forward_and_point_in_time_audit",
        "blockers": [
            "hypotheses_registered_after_2026_observation",
            "index_constituent_overlap_not_verified",
            "etf_master_not_point_in_time",
            "sector_catalog_not_point_in_time",
            "independent_forward_evidence_missing",
        ],
    }
    (output_dir / "DECISION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="研究预登记的板块ETF语义代理")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    args = parser.parse_args()
    output_dir = ensure_output_dir("sector", "etf_proxy_research")
    _, candidates = load_registered_proxy_candidates()
    if args.fetch_only:
        if args.token_file is None:
            parser.error("--fetch-only 时必须提供 --token-file")
        fetch_candidate_history(candidates, args.token_file, args.start_date, args.end_date)
        print(f"[proxy-data] 已更新{candidates['etf_code'].nunique()}只预登记ETF")
        return
    payload = run_research(output_dir)
    print(
        f"[proxy-research] 历史通过={payload['historical_pair_gate_pass_count']}/"
        f"{payload['hypothesis_count']}，正式映射未修改"
    )


if __name__ == "__main__":
    main()
