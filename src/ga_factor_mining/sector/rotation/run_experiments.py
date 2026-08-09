#!/usr/bin/env python3
"""
板块基金假设下的短周期轮动研究。

假设每个同花顺板块指数都有一个无跟踪误差、可按收盘价交易的基金。
信号在 t 日收盘后生成，组合从 t+1 日开始持有，按配置的 rebalance_days 调仓。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import DATA_ROOT, ensure_project_dirs

DATA_DIR = DATA_ROOT / "sector"
OUT_DIR, REPORT_DIR = ensure_project_dirs("sector", "rotation")
FEATURE_PATH = OUT_DIR / "sector_feature_panel.parquet"

TRAIN_END = "20231231"
VAL_START = "20240101"
VAL_END = "20251231"
TEST_START = "20260101"
TEST_END = "20260529"


UNIVERSES = {
    "industry": ["I"],
    "industry_concept": ["I", "N"],
    "tradable_theme": ["I", "N", "R", "TH", "ST"],
}


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


def load_or_build_features(force: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_PATH.exists() and not force:
        print(f"[features] 读取缓存: {FEATURE_PATH}")
        return pd.read_parquet(FEATURE_PATH)

    print("[features] 构建板块特征...")
    daily = pd.read_parquet(DATA_DIR / "ths_daily.parquet")
    index = pd.read_parquet(DATA_DIR / "ths_index.parquet")
    index = index[["ts_code", "name", "type", "count", "list_date"]].drop_duplicates("ts_code")

    df = daily.merge(index, on="ts_code", how="left")
    df = df[df["type"].isin(sorted({t for types in UNIVERSES.values() for t in types}))].copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "pre_close", "avg_price", "vol", "turnover_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("ts_code", sort=False)
    df["ret_1d"] = g["close"].pct_change()
    for w in [3, 5, 10, 20, 60]:
        df[f"ret_{w}d"] = df["close"] / g["close"].shift(w) - 1.0
    for h in [5, 10]:
        df[f"future_ret_{h}d"] = g["close"].shift(-h) / df["close"] - 1.0

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

    # 日内同截面 rank，后续策略只组合 rank，避免跨年份指数点位漂移。
    rank_cols = [
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
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("trade_date")[col].transform(rank_cs)

    keep = [
        "ts_code",
        "trade_date",
        "name",
        "type",
        "count",
        "close",
        "ret_1d",
        "future_ret_5d",
        "future_ret_10d",
    ]
    feature_cols = [c for c in df.columns if c.endswith("_rank") or c in [
        "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "volatility_10d", "volatility_20d", "volatility_60d",
        "volume_z_20d", "turnover_z_20d", "range_1d",
    ]]
    out = df[list(dict.fromkeys(keep + feature_cols))].copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    out[float_cols] = out[float_cols].astype("float32")
    out.to_parquet(FEATURE_PATH, index=False)
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
    sub = df[df["type"].isin(types)].copy()
    feature_cols = [
        "ret_1d_rank", "ret_3d_rank", "ret_5d_rank", "ret_10d_rank", "ret_20d_rank", "ret_60d_rank",
        "volatility_10d_rank", "volatility_20d_rank",
        "risk_adj_5_20_rank", "risk_adj_10_20_rank", "risk_adj_20_60_rank",
        "close_pos_20d_rank", "drawdown_60d_rank", "ma_gap_5_20_rank", "ma_gap_10_60_rank",
        "volume_z_20d_rank", "turnover_z_20d_rank", "range_1d_rank",
    ]
    target = f"future_ret_{target_h}d_rank"
    train_mask = (sub["trade_date"] <= TRAIN_END) & sub[target].notna()
    valid_mask = (sub["trade_date"] >= VAL_START) & (sub["trade_date"] <= TEST_END)
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
    sub = df[df["type"].isin(UNIVERSES[universe])].copy()
    sub = sub.sort_values(["trade_date", "ts_code"])
    ret_pivot = sub.pivot(index="trade_date", columns="ts_code", values="ret_1d").sort_index()
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
    benchmark = ret_pivot.mean(axis=1)
    return {
        "ret_pivot": ret_pivot,
        "score_pivots": score_pivots,
        "name_map": name_map,
        "benchmark": benchmark,
    }


def backtest_one_cached(cache: dict, cfg: StrategyConfig, start: str, end: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    score_all = cache["score_pivots"].get(cfg.score_name)
    if score_all is None:
        return pd.Series(dtype=float), pd.DataFrame(), {"avg_turnover": np.nan, "rebalance_count": 0}
    score_pivot = score_all.loc[(score_all.index >= start) & (score_all.index <= end)]
    ret_pivot = cache["ret_pivot"]
    if score_pivot.empty:
        return pd.Series(dtype=float), pd.DataFrame(), {"avg_turnover": np.nan, "rebalance_count": 0}
    all_dates = list(score_pivot.index)
    daily_rets = {}
    selections = []
    prev_hold: set[str] | None = None
    turnovers = []

    for i in range(0, len(all_dates) - 1, cfg.rebalance_days):
        signal_date = all_dates[i]
        score = score_pivot.loc[signal_date].dropna().sort_values(ascending=False)
        if len(score) < cfg.top_k:
            continue
        hold = list(score.head(cfg.top_k).index)
        hold_set = set(hold)
        if prev_hold is not None:
            turnovers.append(1.0 - len(hold_set & prev_hold) / max(1, len(hold_set | prev_hold)))
        prev_hold = hold_set

        for rank, code in enumerate(hold, start=1):
            selections.append({
                "strategy_id": cfg.strategy_id,
                "signal_date": signal_date,
                "rank": rank,
                "ts_code": code,
                "name": cache["name_map"].get(code, ""),
                "score": float(score.loc[code]),
            })

        end_i = min(i + cfg.hold_days, len(all_dates) - 1)
        for j in range(i + 1, end_i + 1):
            date = all_dates[j]
            vals = ret_pivot.loc[date, hold].dropna()
            daily_rets[date] = float(vals.mean()) if len(vals) else 0.0

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
        "test": (TEST_START, TEST_END),
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
            bench = bench.reindex(ret.index).fillna(0.0)
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
            if period_name == "test":
                curves[full_key] = ret
                positions[full_key] = pos
            if len(pos):
                pos = pos.copy()
                pos["period"] = period_name
                all_pos.append(pos)
        rows.append(row)
    return pd.DataFrame(rows), curves, positions


def write_report(results: pd.DataFrame, curves: dict[str, pd.Series], positions: dict[str, pd.DataFrame], df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = results.sort_values(
        ["val_sharpe", "val_ann_ret", "val_excess_ann_ret", "test_sharpe"],
        ascending=[False, False, False, False],
    )
    results.to_csv(OUT_DIR / "EXPERIMENT_RESULTS.csv", index=False, encoding="utf-8-sig")
    best = results.iloc[0].to_dict()
    best_id = best["strategy_id"]
    if best_id in curves:
        curve = (1.0 + curves[best_id].fillna(0.0)).cumprod()
        curve.to_csv(OUT_DIR / "best_test_equity_curve.csv", header=["equity"])
    if best_id in positions:
        positions[best_id].to_parquet(OUT_DIR / "best_test_positions.parquet", index=False)

    top = results.head(15)
    lines = [
        "# 板块基金轮动策略研究报告",
        "",
        "## 实验设定",
        "",
        "- 假设每个同花顺板块指数都有无跟踪误差基金，可按收盘价交易。",
        "- 信号在当日收盘后生成，收益从下一交易日开始计算。",
        f"- 验证集：{VAL_START}~{VAL_END}；确认测试：{TEST_START}~{TEST_END}。",
        "- 策略只做多 TopK 板块基金，等权持有；暂不考虑手续费、冲击成本、涨跌停和容量。",
        "- 参数按验证集夏普排序选择，测试集只用于确认表现。",
        "",
        "## 最佳验证集方案",
        "",
        f"- 策略：`{best_id}`",
        f"- 验证集年化收益：{best['val_ann_ret']:.2%}",
        f"- 验证集夏普：{best['val_sharpe']:.3f}",
        f"- 验证集最大回撤：{best['val_max_drawdown']:.2%}",
        f"- 验证集相对等权基准年化超额：{best['val_excess_ann_ret']:.2%}",
        f"- 测试集年化收益：{best['test_ann_ret']:.2%}",
        f"- 测试集夏普：{best['test_sharpe']:.3f}",
        f"- 测试集最大回撤：{best['test_max_drawdown']:.2%}",
        f"- 测试集相对等权基准年化超额：{best['test_excess_ann_ret']:.2%}",
        "",
        "## 验证集前 15 名",
        "",
        "| 排名 | 策略 | Val年化 | Val夏普 | Val回撤 | Val超额年化 | Test年化 | Test夏普 | Test回撤 | Test超额年化 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {i} | `{r['strategy_id']}` | {r['val_ann_ret']:.2%} | {r['val_sharpe']:.2f} | {r['val_max_drawdown']:.2%} | {r['val_excess_ann_ret']:.2%} | {r['test_ann_ret']:.2%} | {r['test_sharpe']:.2f} | {r['test_max_drawdown']:.2%} | {r['test_excess_ann_ret']:.2%} |"
        )

    if best_id in positions and len(positions[best_id]):
        latest = positions[best_id]["signal_date"].max()
        last_pos = positions[best_id][positions[best_id]["signal_date"] == latest].sort_values("rank")
        lines += [
            "",
            f"## 最佳策略最近一次测试期持仓信号：{latest}",
            "",
            "| 排名 | 板块代码 | 板块名称 | 分数 |",
            "| ---: | --- | --- | ---: |",
        ]
        for _, r in last_pos.iterrows():
            lines.append(f"| {int(r['rank'])} | `{r['ts_code']}` | {r['name']} | {r['score']:.4f} |")

    lines += [
        "",
        "## 初步判断",
        "",
        "- 如果验证集和 2026 测试集都能跑赢同宇宙等权基准，说明短周期板块强度有可利用信号。",
        "- 如果验证好、测试弱，优先怀疑板块风格切换和参数过拟合，应改成更粗的行业宇宙、更低换手或集成多个分数。",
        "- 这只是板块基金假设，不含真实 ETF 跟踪误差、申赎、流动性、交易成本。后续若实盘化，需要把可交易 ETF 映射和成本模型加进去。",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "BEST_STRATEGY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--with-lgbm", action="store_true")
    args = parser.parse_args()

    df = load_or_build_features(force=args.force_features)
    df = add_formula_scores(df)
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
    write_report(results, curves, positions, df)
    print(f"[done] {REPORT_DIR / 'BEST_STRATEGY_REPORT.md'}")


if __name__ == "__main__":
    main()
