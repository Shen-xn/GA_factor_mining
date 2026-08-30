#!/usr/bin/env python3
"""
板块基金假设下的短周期轮动研究。

假设每个同花顺板块指数都有一个无跟踪误差、可按开盘价交易的基金。
信号在 t 日收盘后生成，组合在 t+1 日开盘成交，按配置的 rebalance_days 调仓。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import DATA_ROOT, ensure_output_dir

DATA_DIR = DATA_ROOT / "sector"
OUT_DIR = ensure_output_dir("sector", "rotation")
FEATURE_PATH = OUT_DIR / "sector_feature_panel.parquet"
FEATURE_META_PATH = OUT_DIR / "sector_feature_panel.meta.json"
FEATURE_PROTOCOL_VERSION = 4
FEATURE_LOGIC_SIGNATURE = "open-to-open-unified-calendar-universe-rank-context-v4"

TRAIN_END = "20231231"
VAL_START = "20240101"
VAL_END = "20251231"
OBSERVATION_START = "20260101"
OBSERVATION_END = "20261231"


UNIVERSES = {
    "industry": ["I"],
    "industry_concept": ["I", "N"],
    "tradable_theme": ["I", "N", "R", "TH", "ST"],
}

RANK_SOURCE_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "volatility_10d",
    "volatility_20d",
    "risk_adj_5_20",
    "risk_adj_10_20",
    "risk_adj_20_60",
    "close_pos_20d",
    "drawdown_60d",
    "ma_gap_5_20",
    "ma_gap_10_60",
    "volume_z_20d",
    "turnover_z_20d",
    "range_1d",
    "future_ret_5d",
    "future_ret_10d",
]

MODEL_FEATURE_COLS = [
    "ret_1d_rank", "ret_3d_rank", "ret_5d_rank", "ret_10d_rank", "ret_20d_rank", "ret_60d_rank",
    "volatility_10d_rank", "volatility_20d_rank",
    "risk_adj_5_20_rank", "risk_adj_10_20_rank", "risk_adj_20_60_rank",
    "close_pos_20d_rank", "drawdown_60d_rank", "ma_gap_5_20_rank", "ma_gap_10_60_rank",
    "volume_z_20d_rank", "turnover_z_20d_rank", "range_1d_rank",
]

MARKET_CONTEXT_COLS = [
    "market_breadth_20d",
    "market_breadth_60d",
    "market_dispersion_20d",
    "market_trend_60d",
    "market_vol_percentile",
]


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    universe: str
    score_name: str
    top_k: int
    rebalance_days: int
    hold_days: int


def rank_cs(s: pd.Series, ascending: bool = True) -> pd.Series:
    return s.rank(pct=True, method="average", ascending=ascending)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_cross_sectional_ranks(df: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """在实际投资宇宙内部重算横截面排名。"""
    out = df[df["type"].isin(types)].copy()
    for col in RANK_SOURCE_COLS:
        if col in out.columns:
            out[f"{col}_rank"] = out.groupby("trade_date")[col].transform(rank_cs)
    return out


def matured_training_mask(df: pd.DataFrame, cutoff: str, horizon: int) -> pd.Series:
    """训练样本的特征日期和标签兑现日期都必须早于截止日。"""
    target = f"future_ret_{horizon}d_rank"
    label_end = f"future_ret_{horizon}d_end_date"
    return (
        (df["trade_date"] <= cutoff)
        & (df[label_end] <= cutoff)
        & df[target].notna()
    )


def add_forward_open_returns(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    """按统一交易日历计算可成交的未来开盘到开盘收益。

    第 t 日收盘产生信号，第 t+1 日开盘进入；h日标签在第 t+h+1 日
    开盘退出。板块自身缺少某个交易日时返回缺失，不允许自动跨日补齐。
    """
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("板块日行情存在重复的 ts_code/trade_date")
    out = df.copy()
    calendar = pd.DataFrame({"trade_date": sorted(out["trade_date"].dropna().astype(str).unique())})
    calendar["next_open_date"] = calendar["trade_date"].shift(-1)
    for horizon in horizons:
        end_col = "return_end_date" if horizon == 1 else f"future_ret_{horizon}d_end_date"
        calendar[end_col] = calendar["trade_date"].shift(-(horizon + 1))
    out = out.merge(calendar, on="trade_date", how="left", validate="many_to_one")

    open_lookup = out.set_index(["ts_code", "trade_date"])["open"]

    def lookup_open(date_column: str) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays(
            [out["ts_code"].to_numpy(), out[date_column].to_numpy()],
            names=["ts_code", "trade_date"],
        )
        return open_lookup.reindex(keys).to_numpy(dtype=float)

    entry_open = lookup_open("next_open_date")
    for horizon in horizons:
        end_col = "return_end_date" if horizon == 1 else f"future_ret_{horizon}d_end_date"
        realized = lookup_open(end_col) / entry_open - 1.0
        if horizon == 1:
            out["forward_open_ret_1d"] = realized
        else:
            out[f"future_ret_{horizon}d"] = realized
    return out


def _file_fingerprint(path: Path) -> dict[str, int | str]:
    """用相对名称、大小和内容哈希绑定原始数据。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        # 不记录机器绝对路径和mtime，数据包复制到另一台机器后仍可校验。
        "path": f"data/sector/{path.name}",
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def source_data_fingerprints() -> dict[str, dict[str, int | str]]:
    return {
        name: _file_fingerprint(DATA_DIR / name)
        for name in ("ths_daily.parquet", "ths_index.parquet")
    }


def _metadata_signature(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _portable_source_identity(sources: dict) -> dict:
    """兼容旧元数据，但只用可跨机器复核的内容字段比较原始数据。"""
    return {
        name: {
            "size": fingerprint.get("size"),
            "sha256": fingerprint.get("sha256"),
        }
        for name, fingerprint in sources.items()
    }


def feature_cache_is_current(feature_path: Path, meta_path: Path) -> bool:
    """协议、构建逻辑和原始数据均一致时才复用特征缓存。"""
    if not feature_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("feature_protocol_version") == FEATURE_PROTOCOL_VERSION
        and metadata.get("feature_logic_signature") == FEATURE_LOGIC_SIGNATURE
        and _portable_source_identity(metadata.get("sources", {}))
        == _portable_source_identity(source_data_fingerprints())
        and metadata.get("feature_cache_signature")
        == _metadata_signature({key: value for key, value in metadata.items() if key != "feature_cache_signature"})
    )


def current_feature_cache_signature() -> str:
    """返回已经通过完整性检查的特征缓存签名。"""
    if not feature_cache_is_current(FEATURE_PATH, FEATURE_META_PATH):
        raise RuntimeError("特征缓存未通过完整性检查")
    metadata = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    return str(metadata["feature_cache_signature"])


def build_feature_frame(daily: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """从给定行情构建特征；增量更新只需传入带足够预热期的尾部行情。"""
    index = index[["ts_code", "name", "type", "count", "list_date"]].drop_duplicates("ts_code")

    df = daily.merge(index, on="ts_code", how="left")
    df = df[df["type"].isin(sorted({t for types in UNIVERSES.values() for t in types}))].copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "pre_close", "avg_price", "vol", "turnover_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("ts_code", sort=False)
    # 缺失收盘价不做隐式前向填充，避免把停牌或缺测误当成零波动。
    df["ret_1d"] = g["close"].pct_change(fill_method=None)
    for w in [3, 5, 10, 20, 60]:
        df[f"ret_{w}d"] = df["close"] / g["close"].shift(w) - 1.0
    df = add_forward_open_returns(df)

    g = df.groupby("ts_code", sort=False)
    for w in [5, 10, 20, 60]:
        df[f"volatility_{w}d"] = (
            g["ret_1d"].rolling(w, min_periods=max(3, w // 2)).std().reset_index(level=0, drop=True)
        )
        df[f"ma_{w}d"] = (
            g["close"].rolling(w, min_periods=max(3, w // 2)).mean().reset_index(level=0, drop=True)
        )

    roll_high_20 = g["high"].rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    roll_low_20 = g["low"].rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
    roll_max_60 = g["close"].rolling(60, min_periods=20).max().reset_index(level=0, drop=True)
    df["close_pos_20d"] = safe_div(df["close"] - roll_low_20, roll_high_20 - roll_low_20)
    df["drawdown_60d"] = df["close"] / roll_max_60 - 1.0
    df["ma_gap_5_20"] = df["ma_5d"] / df["ma_20d"] - 1.0
    df["ma_gap_10_60"] = df["ma_10d"] / df["ma_60d"] - 1.0
    df["range_1d"] = df["high"] / df["low"] - 1.0

    vol_mean_20 = g["vol"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    vol_std_20 = g["vol"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    df["volume_z_20d"] = safe_div(df["vol"] - vol_mean_20, vol_std_20)

    to_mean_20 = g["turnover_rate"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    to_std_20 = g["turnover_rate"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    df["turnover_z_20d"] = safe_div(df["turnover_rate"] - to_mean_20, to_std_20)

    df["risk_adj_5_20"] = safe_div(df["ret_5d"], df["volatility_20d"])
    df["risk_adj_10_20"] = safe_div(df["ret_10d"], df["volatility_20d"])
    df["risk_adj_20_60"] = safe_div(df["ret_20d"], df["volatility_60d"])

    from .risk import build_market_state

    market_state = build_market_state(df).rename(
        columns={
            "breadth_positive_20d": "market_breadth_20d",
            "breadth_positive_60d": "market_breadth_60d",
            "cross_section_dispersion_20d": "market_dispersion_20d",
            "benchmark_trend_60d": "market_trend_60d",
        }
    )
    df = df.merge(
        market_state[["trade_date"] + MARKET_CONTEXT_COLS],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )

    # 缓存中的默认排名服务于主研究宇宙 I+N；其他宇宙在回测缓存中重新计算。
    ranked_main = add_cross_sectional_ranks(df, UNIVERSES["industry_concept"])
    for col in [f"{name}_rank" for name in RANK_SOURCE_COLS]:
        df[col] = np.nan
        if col in ranked_main.columns:
            df.loc[ranked_main.index, col] = ranked_main[col]

    keep = [
        "ts_code", "trade_date", "name", "type", "count", "open", "close", "ret_1d",
        "forward_open_ret_1d", "next_open_date", "return_end_date", "future_ret_5d",
        "future_ret_10d", "future_ret_5d_end_date", "future_ret_10d_end_date",
    ]
    raw_feature_cols = [column for column in RANK_SOURCE_COLS if not column.startswith("future_ret_")]
    feature_cols = [c for c in df.columns if c.endswith("_rank") or c in raw_feature_cols]
    out = df[list(dict.fromkeys(keep + feature_cols + MARKET_CONTEXT_COLS))].copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    out[float_cols] = out[float_cols].astype("float32")
    return out


def write_feature_metadata(feature_columns: list[str]) -> None:
    """写入与原始数据绑定的特征缓存元数据。"""
    metadata = {
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "feature_logic_signature": FEATURE_LOGIC_SIGNATURE,
        "signal_time": "close_t",
        "entry_time": "open_t_plus_1",
        "label": "open_t_plus_h_plus_1_over_open_t_plus_1",
        "rank_universe": "recomputed_per_investment_universe",
        "sources": source_data_fingerprints(),
        "feature_columns": feature_columns,
    }
    metadata["feature_cache_signature"] = _metadata_signature(metadata)
    FEATURE_META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_feature_cache(out: pd.DataFrame) -> None:
    """写入完整特征缓存；日常更新使用流式尾部替换。"""
    out.to_parquet(FEATURE_PATH, index=False)
    write_feature_metadata(list(out.columns))


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not force and feature_cache_is_current(FEATURE_PATH, FEATURE_META_PATH):
        cached = pd.read_parquet(FEATURE_PATH)
        required = {
            "open",
            "forward_open_ret_1d",
            "next_open_date",
            "return_end_date",
            "future_ret_5d",
            "future_ret_5d_end_date",
            "future_ret_10d",
            "future_ret_10d_end_date",
        }
        if required.issubset(cached.columns):
            print(f"[features] 读取缓存: {FEATURE_PATH}")
            return cached
    elif FEATURE_PATH.exists() and not force:
        print("[features] 缓存协议已升级，重新构建")

    print("[features] 构建板块特征...")
    daily = pd.read_parquet(DATA_DIR / "ths_daily.parquet")
    index = pd.read_parquet(DATA_DIR / "ths_index.parquet")
    out = build_feature_frame(daily, index)
    write_feature_cache(out)
    print(f"[features] 保存: {FEATURE_PATH} rows={len(out):,} cols={len(out.columns)}")
    return out


def add_formula_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    neutral = 0.5
    r = lambda c: out[c].fillna(neutral)
    out["score_mom_5_10"] = 0.6 * r("ret_5d_rank") + 0.4 * r("ret_10d_rank")
    out["score_mom_10_20"] = 0.35 * r("ret_5d_rank") + 0.40 * r("ret_10d_rank") + 0.25 * r("ret_20d_rank")
    out["score_risk_adj"] = (
        0.35 * r("risk_adj_5_20_rank")
        + 0.35 * r("risk_adj_10_20_rank")
        + 0.20 * r("risk_adj_20_60_rank")
        - 0.10 * r("volatility_20d_rank")
    )
    out["score_breakout"] = (
        0.35 * r("close_pos_20d_rank")
        + 0.35 * r("ma_gap_5_20_rank")
        + 0.20 * r("ret_20d_rank")
        - 0.10 * r("range_1d_rank")
    )
    out["score_pullback_trend"] = (
        0.35 * (1.0 - r("ret_3d_rank"))
        + 0.35 * r("ret_10d_rank")
        + 0.20 * r("ret_20d_rank")
        + 0.10 * r("close_pos_20d_rank")
    )
    out["score_volume_confirm"] = (
        0.30 * r("ret_5d_rank")
        + 0.30 * r("ret_20d_rank")
        + 0.20 * r("volume_z_20d_rank")
        + 0.10 * r("turnover_z_20d_rank")
        - 0.10 * r("volatility_20d_rank")
    )
    out["score_low_vol_mom"] = (
        0.45 * r("ret_10d_rank")
        + 0.35 * r("ret_20d_rank")
        + 0.20 * (1.0 - r("volatility_20d_rank"))
    )
    return out


def train_lightgbm_scores(df: pd.DataFrame, universe_name: str, target_h: int = 5) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except Exception as exc:
        print(f"[lgbm] lightgbm 不可用: {exc}")
        return df

    print(f"[lgbm] 训练 {universe_name} horizon={target_h}d")
    types = UNIVERSES[universe_name]
    sub = add_cross_sectional_ranks(df, types)
    feature_cols = MODEL_FEATURE_COLS
    target = f"future_ret_{target_h}d_rank"
    train_mask = matured_training_mask(sub, TRAIN_END, target_h)
    valid_mask = (sub["trade_date"] >= VAL_START) & (sub["trade_date"] <= OBSERVATION_END)
    train = sub.loc[train_mask, feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 50_000:
        print(f"[lgbm] 训练样本太少: {len(train):,}")
        return df
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train[feature_cols], train[target])
    pred_frame = sub.loc[valid_mask, ["ts_code", "trade_date"] + feature_cols].copy()
    x = pred_frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    pred_frame[f"score_lgbm_{target_h}d"] = model.predict(x).astype("float32")
    out = df.merge(pred_frame[["ts_code", "trade_date", f"score_lgbm_{target_h}d"]], on=["ts_code", "trade_date"], how="left")

    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(OUT_DIR / f"lgbm_{universe_name}_{target_h}d_importance.csv", index=False, encoding="utf-8-sig")
    return out


def load_feature_subset(columns: set[str] | list[str] | tuple[str, ...]) -> pd.DataFrame:
    """只投影读取指定特征列，供回测等低内存流程使用。"""
    if not feature_cache_is_current(FEATURE_PATH, FEATURE_META_PATH):
        raise RuntimeError(
            "特征缓存不存在或已过期；为避免产品流程意外构建全量面板，"
            "请先单独运行特征构建任务"
        )
    requested = list(dict.fromkeys(columns))
    try:
        cached = pd.read_parquet(FEATURE_PATH, columns=requested)
    except Exception as exc:
        raise RuntimeError(f"特征缓存缺少产品流程所需列: {requested}") from exc
    print(f"[features] 低内存投影读取 {len(requested)} 列: {FEATURE_PATH}")
    return cached


def make_configs(include_lgbm: bool = False) -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []
    score_names = [
        "score_mom_5_10",
        "score_mom_10_20",
        "score_risk_adj",
        "score_breakout",
        "score_pullback_trend",
        "score_volume_confirm",
        "score_low_vol_mom",
    ]
    if include_lgbm:
        score_names += ["score_lgbm_5d", "score_lgbm_10d"]
    for universe in UNIVERSES:
        for score in score_names:
            for top_k in [3, 5, 10, 20]:
                for rb in [1, 5, 10]:
                    configs.append(
                        StrategyConfig(
                            strategy_id=f"{universe}__{score}__top{top_k}__rb{rb}",
                            universe=universe,
                            score_name=score,
                            top_k=top_k,
                            rebalance_days=rb,
                            hold_days=rb,
                        )
                    )
    return configs


def annualized_metrics(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return {
            "days": 0, "total_ret": np.nan, "ann_ret": np.nan, "ann_vol": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan, "win_rate": np.nan,
        }
    curve = (1.0 + ret).cumprod()
    total = float(curve.iloc[-1] - 1.0)
    ann_ret = float(curve.iloc[-1] ** (252 / len(ret)) - 1.0) if curve.iloc[-1] > 0 else -1.0
    ann_vol = float(ret.std(ddof=0) * math.sqrt(252))
    sharpe = float(ret.mean() / ret.std(ddof=0) * math.sqrt(252)) if ret.std(ddof=0) > 0 else np.nan
    dd = curve / curve.cummax() - 1.0
    return {
        "days": int(len(ret)),
        "total_ret": total,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "win_rate": float((ret > 0).mean()),
    }


def prepare_universe_cache(df: pd.DataFrame, universe: str, score_names: list[str]) -> dict:
    """每个宇宙只构造一次矩阵，避免每个配置重复 pivot。"""
    sub = add_cross_sectional_ranks(df, UNIVERSES[universe])
    sub = add_formula_scores(sub)
    sub = sub.sort_values(["trade_date", "ts_code"])
    ret_pivot = sub.pivot(index="trade_date", columns="ts_code", values="forward_open_ret_1d").sort_index()
    score_pivots = {}
    for score_name in score_names:
        if score_name in sub.columns:
            score_pivots[score_name] = sub.pivot(index="trade_date", columns="ts_code", values=score_name).sort_index()
    name_map = (
        sub.dropna(subset=["name"])
        .drop_duplicates("ts_code", keep="last")
        .set_index("ts_code")["name"]
        .to_dict()
    )
    return_date_map = (
        sub[["trade_date", "return_end_date"]]
        .dropna()
        .drop_duplicates("trade_date")
        .set_index("trade_date")["return_end_date"]
        .to_dict()
    )
    execution_date_map = (
        sub[["trade_date", "next_open_date"]]
        .dropna()
        .drop_duplicates("trade_date")
        .set_index("trade_date")["next_open_date"]
        .to_dict()
    )
    benchmark_signal = ret_pivot.mean(axis=1)
    benchmark = pd.Series(
        {
            str(return_date_map[date]): float(value)
            for date, value in benchmark_signal.items()
            if date in return_date_map
        }
    ).sort_index()
    return {
        "ret_pivot": ret_pivot,
        "score_pivots": score_pivots,
        "name_map": name_map,
        "benchmark": benchmark,
        "return_date_map": return_date_map,
        "execution_date_map": execution_date_map,
    }


def backtest_one_cached(cache: dict, cfg: StrategyConfig, start: str, end: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    score_all = cache["score_pivots"].get(cfg.score_name)
    if score_all is None:
        return pd.Series(dtype=float), pd.DataFrame(), {"avg_turnover": np.nan, "rebalance_count": 0}
    score_pivot = score_all.loc[(score_all.index >= start) & (score_all.index <= end)]
    ret_pivot = cache["ret_pivot"]
    if score_pivot.empty:
        return pd.Series(dtype=float), pd.DataFrame(), {"avg_turnover": np.nan, "rebalance_count": 0}
    return_date_map = cache["return_date_map"]
    execution_date_map = cache["execution_date_map"]
    all_dates = [
        date for date in score_pivot.index
        if date in return_date_map and str(return_date_map[date]) <= end
    ]
    daily_rets = {}
    selections = []
    turnovers = []
    live_weights: dict[str, float] = {}

    for i in range(0, len(all_dates), cfg.rebalance_days):
        signal_date = all_dates[i]
        score = score_pivot.loc[signal_date].dropna().sort_values(ascending=False)
        # 只有能形成完整次日开盘收益的板块才进入本次理论组合。
        # 这是数据完整性门，不用缺失收益伪造零收益。
        valid_return = ret_pivot.loc[signal_date].notna()
        tradable_mask = valid_return.reindex(score.index).eq(True).to_numpy(dtype=bool)
        score = score[tradable_mask]
        if len(score) < cfg.top_k:
            continue
        hold = list(score.head(cfg.top_k).index)
        target_weights = {code: 1.0 / cfg.top_k for code in hold}
        all_assets = set(live_weights) | set(target_weights)
        asset_change = sum(abs(target_weights.get(code, 0.0) - live_weights.get(code, 0.0)) for code in all_assets)
        old_cash = 1.0 - sum(live_weights.values())
        new_cash = 1.0 - sum(target_weights.values())
        turnovers.append(0.5 * (asset_change + abs(new_cash - old_cash)))
        live_weights = target_weights

        for rank, code in enumerate(hold, start=1):
            selections.append({
                "strategy_id": cfg.strategy_id,
                "signal_date": signal_date,
                "execution_date": execution_date_map.get(signal_date, ""),
                "rank": rank,
                "ts_code": code,
                "name": cache["name_map"].get(code, ""),
                "score": float(score.loc[code]),
            })

        end_i = min(i + cfg.hold_days, len(all_dates))
        for j in range(i, end_i):
            signal_for_return = all_dates[j]
            return_date = str(return_date_map[signal_for_return])
            asset_returns = ret_pivot.loc[signal_for_return].reindex(live_weights)
            if asset_returns.isna().any():
                missing_codes = asset_returns[asset_returns.isna()].index.tolist()
                raise RuntimeError(
                    f"{signal_for_return} 持仓缺少次日开盘收益: {missing_codes}"
                )
            portfolio_return = float(sum(live_weights[code] * asset_returns.loc[code] for code in live_weights))
            daily_rets[return_date] = portfolio_return
            gross = 1.0 + portfolio_return
            if gross > 0:
                live_weights = {
                    code: weight * (1.0 + float(asset_returns.loc[code])) / gross
                    for code, weight in live_weights.items()
                }

    ret = pd.Series(daily_rets).sort_index()
    positions = pd.DataFrame(selections)
    aux = {
        "avg_turnover": float(np.mean(turnovers)) if turnovers else np.nan,
        "rebalance_count": int(len(positions["signal_date"].unique())) if len(positions) else 0,
    }
    return ret, positions, aux


def evaluate_configs(df: pd.DataFrame, configs: list[StrategyConfig]) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame]]:
    rows = []
    curves: dict[str, pd.Series] = {}
    positions: dict[str, pd.DataFrame] = {}
    periods = {
        "val": (VAL_START, VAL_END),
        "observation": (OBSERVATION_START, OBSERVATION_END),
    }
    score_names = sorted({c.score_name for c in configs})
    universe_cache = {}
    for universe in sorted({c.universe for c in configs}):
        print(f"[cache] universe={universe}")
        universe_cache[universe] = prepare_universe_cache(df, universe, score_names)

    start_time = time.time()
    for n, cfg in enumerate(configs, start=1):
        if n % 50 == 0:
            print(f"[backtest] {n}/{len(configs)} elapsed={(time.time()-start_time)/60:.1f}m")
        row = asdict(cfg)
        full_key = cfg.strategy_id
        all_pos = []
        cache = universe_cache[cfg.universe]
        for period_name, (start, end) in periods.items():
            ret, pos, aux = backtest_one_cached(cache, cfg, start, end)
            bench = cache["benchmark"].loc[(cache["benchmark"].index >= start) & (cache["benchmark"].index <= end)]
            bench = bench.reindex(ret.index)
            if bench.isna().any():
                missing_dates = bench[bench.isna()].index.tolist()
                raise RuntimeError(f"基准缺少策略收益日: {missing_dates[:5]}")
            excess = ret - bench
            m = annualized_metrics(ret)
            b = annualized_metrics(bench)
            e = annualized_metrics(excess)
            for k, v in m.items():
                row[f"{period_name}_{k}"] = v
            row[f"{period_name}_bench_ann_ret"] = b["ann_ret"]
            row[f"{period_name}_excess_ann_ret"] = e["ann_ret"]
            row[f"{period_name}_excess_sharpe"] = e["sharpe"]
            row[f"{period_name}_avg_turnover"] = aux["avg_turnover"]
            row[f"{period_name}_rebalance_count"] = aux["rebalance_count"]
            if period_name == "observation":
                curves[full_key] = ret
                positions[full_key] = pos
            if len(pos):
                pos = pos.copy()
                pos["period"] = period_name
                all_pos.append(pos)
        rows.append(row)
    return pd.DataFrame(rows), curves, positions


def select_best_result(results: pd.DataFrame) -> pd.DataFrame:
    """只按验证期指标选型；观察期字段不得参与排序。"""
    return results.sort_values(
        ["val_sharpe", "val_ann_ret", "val_excess_ann_ret", "strategy_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def write_artifacts(results: pd.DataFrame, curves: dict[str, pd.Series], positions: dict[str, pd.DataFrame]) -> None:
    """保存实验结果，报告展示由仓库外的按需工具负责。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = select_best_result(results)
    results.to_csv(OUT_DIR / "EXPERIMENT_RESULTS.csv", index=False, encoding="utf-8-sig")
    best = results.iloc[0].to_dict()
    best_id = best["strategy_id"]
    if best_id in curves:
        curve = (1.0 + curves[best_id].fillna(0.0)).cumprod()
        curve.rename_axis("date").to_csv(OUT_DIR / "best_observation_equity_curve.csv", header=["equity"])
    if best_id in positions:
        positions[best_id].to_parquet(OUT_DIR / "best_observation_positions.parquet", index=False)

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--with-lgbm", action="store_true")
    args = parser.parse_args()

    df = load_or_build_features(force=args.force_features)
    if args.with_lgbm:
        # 只在主宇宙上训练，避免过度拉长实验时间。
        df = train_lightgbm_scores(df, "industry_concept", target_h=5)
        df = train_lightgbm_scores(df, "industry_concept", target_h=10)

    include_lgbm = args.with_lgbm and "score_lgbm_5d" in df.columns
    configs = make_configs(include_lgbm=include_lgbm)
    # LGBM 分数只对 industry_concept 有预测，过滤掉其他宇宙下的 LGBM 配置。
    configs = [
        c for c in configs
        if not c.score_name.startswith("score_lgbm") or c.universe == "industry_concept"
    ]
    print(f"[run] configs={len(configs)}")
    results, curves, positions = evaluate_configs(df, configs)
    write_artifacts(results, curves, positions)
    print(f"[done] {OUT_DIR / 'EXPERIMENT_RESULTS.csv'}")


if __name__ == "__main__":
    main()
