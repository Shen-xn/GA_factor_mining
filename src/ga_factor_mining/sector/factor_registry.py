#!/usr/bin/env python3
"""维护因子依赖关系，并输出机器可读的重复检查结果。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..common.paths import CONFIG_ROOT, ensure_output_dir


REGISTRY_PATH = CONFIG_ROOT / "sector" / "factor_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry(registry: dict) -> list[str]:
    """校验唯一性、依赖闭包和循环依赖，返回因子顺序。"""
    factors = registry.get("factors", [])
    ids = [factor["factor_id"] for factor in factors]
    if len(ids) != len(set(ids)):
        raise ValueError("因子注册表存在重复 factor_id")
    external = set(registry.get("external_sources", []))
    known = set(ids) | external
    dependencies = {factor["factor_id"]: factor.get("dependencies", []) for factor in factors}
    unknown = {
        dependency
        for values in dependencies.values()
        for dependency in values
        if dependency not in known
    }
    if unknown:
        raise ValueError(f"因子注册表存在未知依赖: {sorted(unknown)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(factor_id: str) -> None:
        if factor_id in visiting:
            raise ValueError(f"因子注册表存在循环依赖: {factor_id}")
        if factor_id in visited:
            return
        visiting.add(factor_id)
        for dependency in dependencies[factor_id]:
            if dependency in dependencies:
                visit(dependency)
        visiting.remove(factor_id)
        visited.add(factor_id)

    for factor_id in ids:
        visit(factor_id)
    return ids


def dependency_profiles(registry: dict) -> dict[str, dict]:
    """展开完整依赖链，区分已有因子表示和真正的原始信息源。"""
    ids = validate_registry(registry)
    factor_ids = set(ids)
    dependencies = {
        factor["factor_id"]: list(factor.get("dependencies", []))
        for factor in registry["factors"]
    }
    cache: dict[str, dict] = {}

    def expand(factor_id: str) -> dict:
        if factor_id in cache:
            return cache[factor_id]
        factor_dependencies: set[str] = set()
        raw_sources: set[str] = set()
        depth = 0
        for dependency in dependencies[factor_id]:
            if dependency in factor_ids:
                child = expand(dependency)
                factor_dependencies.add(dependency)
                factor_dependencies.update(child["factor_dependencies"])
                raw_sources.update(child["raw_sources"])
                depth = max(depth, 1 + int(child["dependency_depth"]))
            else:
                raw_sources.add(dependency)
        profile = {
            "direct_dependencies": dependencies[factor_id],
            "factor_dependencies": sorted(factor_dependencies),
            "raw_sources": sorted(raw_sources),
            "dependency_depth": depth,
            "derived_only_from_registered_factors": bool(dependencies[factor_id])
            and all(item in factor_ids for item in dependencies[factor_id]),
        }
        cache[factor_id] = profile
        return profile

    return {factor_id: expand(factor_id) for factor_id in ids}


def _empty_moments(size: int) -> dict[str, np.ndarray]:
    shape = (size, size)
    return {
        "n": np.zeros(shape, dtype="float64"),
        "sum_x": np.zeros(shape, dtype="float64"),
        "sum_x2": np.zeros(shape, dtype="float64"),
        "sum_xy": np.zeros(shape, dtype="float64"),
    }


def _update_moments(state: dict[str, np.ndarray], values: np.ndarray) -> None:
    """分批累计两两相关所需统计量，不在内存中保留完整面板。"""
    if values.size == 0:
        return
    valid = np.isfinite(values).astype("float64")
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    state["n"] += valid.T @ valid
    state["sum_x"] += filled.T @ valid
    state["sum_x2"] += (filled * filled).T @ valid
    state["sum_xy"] += filled.T @ filled


def _moments_correlation(state: dict[str, np.ndarray], min_periods: int) -> np.ndarray:
    n = state["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = state["sum_xy"] - state["sum_x"] * state["sum_x"].T / n
        variance_x = state["sum_x2"] - state["sum_x"] ** 2 / n
        variance_y = state["sum_x2"].T - state["sum_x"].T ** 2 / n
        correlation = covariance / np.sqrt(variance_x * variance_y)
    correlation[(n < min_periods) | ~np.isfinite(correlation)] = np.nan
    return correlation


def _pair_relation(left: str, right: str, profiles: dict[str, dict]) -> str:
    if left in profiles[right]["factor_dependencies"]:
        return "right_derived_from_left"
    if right in profiles[left]["factor_dependencies"]:
        return "left_derived_from_right"
    if profiles[left]["raw_sources"] == profiles[right]["raw_sources"]:
        return "same_raw_sources"
    if set(profiles[left]["raw_sources"]) & set(profiles[right]["raw_sources"]):
        return "overlapping_raw_sources"
    return "empirical_only"


def build_streaming_redundancy_report(
    feature_path: Path,
    registry: dict,
    periods: dict[str, tuple[str, str]],
    decision_period: str = "development",
    threshold: float = 0.80,
    batch_size: int = 65_536,
    sector_min_periods: int = 1_000,
    context_min_periods: int = 100,
) -> pd.DataFrame:
    """分批计算多期间相关；只有开发期相关性参与去重决策。"""
    ids = validate_registry(registry)
    profiles = dependency_profiles(registry)
    specs = {factor["factor_id"]: factor for factor in registry["factors"]}
    factor_columns = {
        factor_id: specs[factor_id].get("model_column", f"{factor_id}_rank")
        for factor_id in ids
    }
    sector_ids = [factor_id for factor_id in ids if specs[factor_id]["family"] != "market_context"]
    context_ids = [factor_id for factor_id in ids if specs[factor_id]["family"] == "market_context"]
    sector_columns = [factor_columns[factor_id] for factor_id in sector_ids]
    context_columns = [factor_columns[factor_id] for factor_id in context_ids]
    requested = ["trade_date", *sector_columns, *context_columns]
    parquet = pq.ParquetFile(feature_path)
    missing = sorted(set(requested) - set(parquet.schema.names))
    if missing:
        raise ValueError(f"特征面板缺少注册因子: {missing}")

    states = {
        period: {
            "sector": _empty_moments(len(sector_ids)),
            "context": _empty_moments(len(context_ids)),
        }
        for period in periods
    }
    context_dates: dict[str, set[str]] = {period: set() for period in periods}
    for batch in parquet.iter_batches(columns=requested, batch_size=batch_size):
        frame = batch.to_pandas()
        dates = frame["trade_date"].astype(str)
        for period, (start, end) in periods.items():
            selected = frame.loc[dates.between(start, end)]
            if selected.empty:
                continue
            _update_moments(
                states[period]["sector"],
                selected[sector_columns].to_numpy(dtype="float64", copy=False),
            )
            context = selected[["trade_date", *context_columns]].drop_duplicates("trade_date")
            context["trade_date"] = context["trade_date"].astype(str)
            context = context.loc[~context["trade_date"].isin(context_dates[period])]
            if not context.empty:
                context_dates[period].update(context["trade_date"].tolist())
                _update_moments(
                    states[period]["context"],
                    context[context_columns].to_numpy(dtype="float64", copy=False),
                )

    correlations = {
        period: {
            "sector": _moments_correlation(state["sector"], sector_min_periods),
            "context": _moments_correlation(state["context"], context_min_periods),
        }
        for period, state in states.items()
    }
    index_by_scope = {
        "sector": {factor_id: index for index, factor_id in enumerate(sector_ids)},
        "context": {factor_id: index for index, factor_id in enumerate(context_ids)},
    }
    rows: list[dict] = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            left_scope = "context" if left in context_ids else "sector"
            right_scope = "context" if right in context_ids else "sector"
            if left_scope != right_scope:
                continue
            scope = left_scope
            i, j = index_by_scope[scope][left], index_by_scope[scope][right]
            period_values = {
                period: float(correlations[period][scope][i, j])
                for period in periods
            }
            decision_value = period_values[decision_period]
            high_periods = [
                period
                for period, value in period_values.items()
                if np.isfinite(value) and abs(value) >= threshold
            ]
            relation = _pair_relation(left, right, profiles)
            if np.isfinite(decision_value) and abs(decision_value) >= threshold:
                action = (
                    "ablate_derived_representation"
                    if "derived_from" in relation
                    else "controlled_pair_ablation"
                )
            else:
                action = "keep"
            row = {
                "left_factor": left,
                "right_factor": right,
                "scope": scope,
                "relation": relation,
                "same_family": specs[left]["family"] == specs[right]["family"],
                "flagged_by_development": bool(
                    np.isfinite(decision_value) and abs(decision_value) >= threshold
                ),
                "high_correlation_periods": "|".join(high_periods),
                "high_correlation_period_count": len(high_periods),
                "suggested_action": action,
            }
            for period, value in period_values.items():
                row[f"correlation_{period}"] = value
                row[f"abs_correlation_{period}"] = abs(value)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        f"abs_correlation_{decision_period}", ascending=False
    ).reset_index(drop=True)


def build_factor_catalog(registry: dict, redundancy: pd.DataFrame) -> pd.DataFrame:
    """汇总每个因子的结构来源和开发期重复风险。"""
    profiles = dependency_profiles(registry)
    rows = []
    for spec in registry["factors"]:
        factor_id = spec["factor_id"]
        pairs = redundancy.loc[
            redundancy["left_factor"].eq(factor_id)
            | redundancy["right_factor"].eq(factor_id)
        ]
        flagged = pairs.loc[pairs["flagged_by_development"]]
        max_corr = (
            float(pairs["abs_correlation_development"].max()) if not pairs.empty else np.nan
        )
        profile = profiles[factor_id]
        rows.append(
            {
                "factor_id": factor_id,
                "model_column": spec.get("model_column", f"{factor_id}_rank"),
                "family": spec["family"],
                "kind": spec["kind"],
                "status": spec["status"],
                "direct_dependencies": "|".join(profile["direct_dependencies"]),
                "factor_dependencies": "|".join(profile["factor_dependencies"]),
                "raw_sources": "|".join(profile["raw_sources"]),
                "dependency_depth": profile["dependency_depth"],
                "derived_only_from_registered_factors": profile[
                    "derived_only_from_registered_factors"
                ],
                "flagged_pair_count": int(len(flagged)),
                "max_abs_correlation_development": max_corr,
                "review_required": bool(len(flagged)),
            }
        )
    return pd.DataFrame(rows)


def build_redundancy_report(
    panel: pd.DataFrame,
    registry: dict,
    start: str,
    end: str,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """在固定期间比较排名因子；这里只做筛查，不代替模型增量消融。"""
    ids = validate_registry(registry)
    specs = {factor["factor_id"]: factor for factor in registry["factors"]}
    factor_columns = {
        factor_id: specs[factor_id].get("model_column", f"{factor_id}_rank")
        for factor_id in ids
    }
    columns = list(factor_columns.values())
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        raise ValueError(f"特征面板缺少注册因子: {missing}")
    sample = panel.loc[
        (panel["trade_date"] >= start) & (panel["trade_date"] <= end),
        ["trade_date", *columns],
    ]
    sector_ids = [factor_id for factor_id in ids if specs[factor_id]["family"] != "market_context"]
    context_ids = [factor_id for factor_id in ids if specs[factor_id]["family"] == "market_context"]
    sector_columns = [factor_columns[factor_id] for factor_id in sector_ids]
    context_columns = [factor_columns[factor_id] for factor_id in context_ids]
    sector_correlation = sample[sector_columns].corr(method="pearson", min_periods=1_000)
    # 市场状态列在同一天对所有板块相同，先按日期去重，避免板块数量变化给日期加权。
    context_correlation = sample.drop_duplicates("trade_date")[context_columns].corr(
        method="pearson", min_periods=100
    )
    rows = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            left_scope = "context" if left in context_ids else "sector"
            right_scope = "context" if right in context_ids else "sector"
            if left_scope != right_scope:
                continue
            correlation = context_correlation if left_scope == "context" else sector_correlation
            value = float(correlation.loc[factor_columns[left], factor_columns[right]])
            left_dependencies = set(specs[left].get("dependencies", []))
            right_dependencies = set(specs[right].get("dependencies", []))
            rows.append(
                {
                    "left_factor": left,
                    "right_factor": right,
                    "scope": left_scope,
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "same_family": specs[left]["family"] == specs[right]["family"],
                    "direct_dependency": left in right_dependencies or right in left_dependencies,
                    "flagged": bool(np.isfinite(value) and abs(value) >= threshold),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def main() -> None:
    from .rotation.run_experiments import (
        FEATURE_META_PATH,
        FEATURE_PATH,
        feature_cache_is_current,
    )

    registry = load_registry()
    factor_ids = validate_registry(registry)
    if not feature_cache_is_current(FEATURE_PATH, FEATURE_META_PATH):
        raise RuntimeError("特征缓存未通过原始数据和协议签名检查")
    periods = {
        "development": ("20150101", "20231231"),
        "selection": ("20240101", "20251231"),
        "observation": ("20260101", "20260810"),
    }
    report = build_streaming_redundancy_report(
        FEATURE_PATH,
        registry,
        periods,
        decision_period="development",
    )
    catalog = build_factor_catalog(registry, report)
    flagged = report.loc[report["flagged_by_development"]]
    top_pair = report.iloc[0]
    output_dir = ensure_output_dir("sector", "factors")
    report.to_csv(output_dir / "REDUNDANCY.csv", index=False, encoding="utf-8-sig")
    catalog.to_csv(output_dir / "FACTOR_CATALOG.csv", index=False, encoding="utf-8-sig")
    (output_dir / "VALIDATION.json").write_text(
        json.dumps(
            {
                "registry_version": registry["version"],
                "factor_count": len(factor_ids),
                "periods": periods,
                "decision_period": "development",
                "selection_safe": True,
                "observation_used_for_decision": False,
                "streaming_read": True,
                "cross_scope_pairs_excluded": True,
                "correlation_threshold": 0.80,
                "flagged_pairs": int(len(flagged)),
                "derived_representation_pairs": int(
                    flagged["suggested_action"].eq("ablate_derived_representation").sum()
                ),
                "controlled_pair_ablation_pairs": int(
                    flagged["suggested_action"].eq("controlled_pair_ablation").sum()
                ),
                "most_redundant_pair": {
                    "left_factor": str(top_pair["left_factor"]),
                    "right_factor": str(top_pair["right_factor"]),
                    "abs_correlation_development": float(
                        top_pair["abs_correlation_development"]
                    ),
                },
                "formal_feature_set_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {output_dir / 'REDUNDANCY.csv'} rows={len(report)}")


if __name__ == "__main__":
    main()
