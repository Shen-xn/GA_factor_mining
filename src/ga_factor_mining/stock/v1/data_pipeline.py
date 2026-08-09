"""独立因子策略的数据读取、标签构造与横截面统计工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXCLUDED_TERMINALS = {
    "is_basic_missing", "is_moneyflow_missing", "is_tech_warmup",
    "is_recent_listing", "valid_feature_ratio", "valuation_missing_count",
    "is_loss_or_pe_missing",
}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    base = config_path.parent
    for key in (
        "features_path", "feature_meta_path", "price_path",
        "prepared_data_path", "output_dir",
    ):
        if key in config["data"]:
            config["data"][key] = str((base / config["data"][key]).resolve())
    Path(config["data"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    return config, config_path


def load_feature_names(config: dict[str, Any]) -> list[str]:
    with open(config["data"]["feature_meta_path"], "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return [x for x in metadata["feature_columns"] if x not in EXCLUDED_TERMINALS]


def prepare_data(config: dict[str, Any], force: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """合并特征和前复权价格，并按股票构造严格的未来 5 日收益。"""
    # 缓存可放在项目之外，迁移代码时无需复制数 GB 的预处理数据。
    cache_path = Path(config["data"].get(
        "prepared_data_path",
        Path(config["data"]["output_dir"]) / "prepared_data.parquet",
    ))
    feature_names = load_feature_names(config)
    target_name = config["target"]["name"]
    if cache_path.exists() and not force:
        print(f"[data] 读取缓存: {cache_path}")
        return pd.read_parquet(cache_path), feature_names
    print("[data] 读取基础特征与前复权价格")
    features = pd.read_parquet(
        config["data"]["features_path"],
        columns=["ts_code", "trade_date", *feature_names],
    )
    price_column = config["data"]["price_column"]
    prices = pd.read_parquet(
        config["data"]["price_path"],
        columns=["ts_code", "trade_date", price_column],
    ).sort_values(["ts_code", "trade_date"])
    horizon = int(config["target"]["horizon"])
    prices[target_name] = (
        prices.groupby("ts_code", sort=False)[price_column].shift(-horizon)
        / prices[price_column] - 1.0
    )
    data = features.merge(
        prices[["ts_code", "trade_date", price_column, target_name]],
        on=["ts_code", "trade_date"], how="inner", validate="one_to_one",
    )
    data["trade_date"] = data["trade_date"].astype(str)
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    data.to_parquet(cache_path, index=False)
    print(f"[data] 缓存完成: rows={len(data):,} path={cache_path}")
    return data, feature_names


def daily_rank_ic(factor: pd.Series, target: pd.Series, dates: pd.Series, min_stocks: int) -> pd.Series:
    """逐日横截面 Spearman IC。"""
    frame = pd.DataFrame({"date": dates, "factor": factor, "target": target}).dropna()
    counts = frame.groupby("date", sort=False).size()
    frame = frame[frame["date"].isin(counts[counts >= min_stocks].index)]
    if frame.empty:
        return pd.Series(dtype=np.float64)
    frame["factor_rank"] = frame.groupby("date", sort=False)["factor"].rank(pct=True)
    frame["target_rank"] = frame.groupby("date", sort=False)["target"].rank(pct=True)
    grouped = frame.groupby("date", sort=False)
    return grouped.apply(
        lambda x: x["factor_rank"].corr(x["target_rank"]),
        include_groups=False,
    )
