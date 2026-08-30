from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .run_experiments import (
    OUT_DIR,
    OBSERVATION_END,
    UNIVERSES,
    StrategyConfig,
    annualized_metrics,
    backtest_one_cached,
    load_or_build_features,
    prepare_universe_cache,
)


ROLLING_RESULTS = OUT_DIR / "ROLLING_VALIDATION_RESULTS.csv"
ROLLING_CURVES = OUT_DIR / "ROLLING_VALIDATION_EQUITY_CURVES.csv"

FEATURE_COLS = [
    "ret_1d_rank", "ret_3d_rank", "ret_5d_rank", "ret_10d_rank", "ret_20d_rank",
    "ret_60d_rank", "volatility_10d_rank", "volatility_20d_rank", "risk_adj_5_20_rank",
    "risk_adj_10_20_rank", "risk_adj_20_60_rank", "close_pos_20d_rank",
    "drawdown_60d_rank", "ma_gap_5_20_rank", "ma_gap_10_60_rank", "volume_z_20d_rank",
    "turnover_z_20d_rank", "range_1d_rank",
]


def make_lgbm_model(horizon: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression", n_estimators=350, learning_rate=0.035, num_leaves=31,
        min_child_samples=160, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2,
        # 本地研究默认单线程，避免LightGBM与NumPy同时抢占内存导致进程崩溃。
        reg_lambda=1.0, random_state=42 + horizon, n_jobs=1, verbose=-1,
        deterministic=True, force_col_wise=True,
    )

ML_STRATEGIES = [
    {
        "strategy_id": "rolling_lgbm5_top5",
        "display_name": "滚动 LGBM5 Top5",
        "horizon": 5,
        "top_k": 5,
        "rebalance_days": 1,
        "hold_days": 1,
    },
    {
        "strategy_id": "rolling_lgbm10_top10",
        "display_name": "滚动 LGBM10 Top10",
        "horizon": 10,
        "top_k": 10,
        "rebalance_days": 1,
        "hold_days": 1,
    },
]

FORMULA_STRATEGIES = [
    {
        "strategy_id": "formula_industry_breakout_top3",
        "display_name": "公式 行业突破 Top3",
        "config": StrategyConfig(
            "formula_industry_breakout_top3", "industry", "score_breakout", 3, 1, 1
        ),
    },
    {
        "strategy_id": "formula_low_vol_top20",
        "display_name": "公式 低波动动量 Top20",
        "config": StrategyConfig(
            "formula_low_vol_top20", "industry_concept", "score_low_vol_mom", 20, 1, 1
        ),
    },
]


def year_bounds(df: pd.DataFrame, start_year: int, end_year: int) -> dict[int, tuple[str, str]]:
    dates = df[["trade_date"]].drop_duplicates().sort_values("trade_date")
    dates["year"] = dates["trade_date"].astype(str).str[:4].astype(int)
    bounds: dict[int, tuple[str, str]] = {}
    for year in range(start_year, end_year + 1):
        sub = dates[dates["year"].eq(year)]
        if not sub.empty:
            bounds[year] = (str(sub["trade_date"].min()), str(sub["trade_date"].max()))
    if 2026 in bounds:
        start, _ = bounds[2026]
        bounds[2026] = (start, OBSERVATION_END)
    return bounds


def training_window_mask(
    df: pd.DataFrame,
    train_end: str,
    horizon: int,
    training_years: int | None,
) -> pd.Series:
    from .run_experiments import matured_training_mask

    mask = matured_training_mask(df, train_end, horizon)
    if training_years is not None:
        start_year = int(train_end[:4]) - training_years + 1
        mask &= df["trade_date"] >= f"{start_year}0101"
    return mask


def recency_sample_weights(
    trade_dates: pd.Series,
    train_end: str,
    half_life_years: float | None,
) -> np.ndarray | None:
    if half_life_years is None:
        return None
    if half_life_years <= 0:
        raise ValueError("half_life_years 必须大于0")
    end = pd.Timestamp(train_end)
    dates = pd.to_datetime(trade_dates, format="%Y%m%d")
    age_days = (end - dates).dt.days.clip(lower=0)
    return np.power(0.5, age_days.to_numpy() / (365.25 * half_life_years))


def fit_predict_lgbm(
    df: pd.DataFrame,
    horizon: int,
    fold_years: list[int],
    training_years: int | None = None,
    recency_half_life_years: float | None = None,
    score_col: str | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    target = f"future_ret_{horizon}d_rank"
    score_col = score_col or f"score_rolling_lgbm_{horizon}d"
    features = feature_cols or FEATURE_COLS
    sub = df[df["type"].isin(UNIVERSES["industry_concept"])].copy()
    sub[score_col] = np.nan
    for year in fold_years:
        train_end = f"{year - 1}1231"
        pred_start = f"{year}0101"
        pred_end = OBSERVATION_END if year == 2026 else f"{year}1231"
        train_mask = training_window_mask(sub, train_end, horizon, training_years)
        pred_mask = (sub["trade_date"] >= pred_start) & (sub["trade_date"] <= pred_end)
        train = sub.loc[train_mask, ["trade_date"] + features + [target]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        pred = sub.loc[pred_mask, features].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        if train.empty or pred.empty:
            continue
        print(
            f"[rolling-lgbm] horizon={horizon} validate={year} "
            f"train_end={train_end} years={training_years or 'all'} "
            f"half_life={recency_half_life_years or 'none'} "
            f"train_rows={len(train):,} pred_rows={len(pred):,}"
        )
        model = make_lgbm_model(horizon)
        sample_weight = recency_sample_weights(
            train["trade_date"], train_end, recency_half_life_years
        )
        model.fit(train[features], train[target], sample_weight=sample_weight)
        sub.loc[pred_mask, score_col] = model.predict(pred).astype("float32")
    return sub[["ts_code", "trade_date", score_col]]


def periodic_prediction_bounds(start: str, end: str, retrain_months: int) -> list[tuple[str, str, str]]:
    """返回 (train_end, predict_start, predict_end)，各预测区间互不重叠。"""
    if retrain_months <= 0:
        raise ValueError("retrain_months 必须大于0")
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{retrain_months}MS")
    bounds = []
    final_end = pd.Timestamp(end)
    for pred_start in starts:
        pred_end = min(pred_start + pd.DateOffset(months=retrain_months) - pd.Timedelta(days=1), final_end)
        train_end = pred_start - pd.Timedelta(days=1)
        bounds.append(
            (train_end.strftime("%Y%m%d"), pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))
        )
    return bounds


def fit_predict_lgbm_periodic(
    df: pd.DataFrame,
    horizon: int,
    start: str,
    end: str,
    retrain_months: int = 3,
    training_years: int | None = None,
    recency_half_life_years: float | None = None,
    score_col: str | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """按固定月份间隔重训；每次仍只使用标签已兑现的数据。"""
    target = f"future_ret_{horizon}d_rank"
    score_col = score_col or f"score_periodic_lgbm_{horizon}d"
    features = feature_cols or FEATURE_COLS
    sub = df[df["type"].isin(UNIVERSES["industry_concept"])].copy()
    sub[score_col] = np.nan
    for train_end, pred_start, pred_end in periodic_prediction_bounds(start, end, retrain_months):
        train_mask = training_window_mask(sub, train_end, horizon, training_years)
        pred_mask = (sub["trade_date"] >= pred_start) & (sub["trade_date"] <= pred_end)
        train = sub.loc[train_mask, ["trade_date"] + features + [target]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        pred = sub.loc[pred_mask, features].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        if train.empty or pred.empty:
            continue
        print(
            f"[periodic-lgbm] horizon={horizon} train_end={train_end} "
            f"predict={pred_start}:{pred_end} train_rows={len(train):,} pred_rows={len(pred):,}"
        )
        model = make_lgbm_model(horizon)
        sample_weight = recency_sample_weights(
            train["trade_date"], train_end, recency_half_life_years
        )
        model.fit(train[features], train[target], sample_weight=sample_weight)
        sub.loc[pred_mask, score_col] = model.predict(pred).astype("float32")
    return sub[["ts_code", "trade_date", score_col]]


def evaluate_year(cache: dict, cfg: StrategyConfig, year: int, start: str, end: str) -> tuple[dict, pd.Series]:
    ret, _pos, aux = backtest_one_cached(cache, cfg, start, end)
    bench = cache["benchmark"].loc[(cache["benchmark"].index >= start) & (cache["benchmark"].index <= end)]
    bench = bench.reindex(ret.index)
    if bench.isna().any():
        missing_dates = bench[bench.isna()].index.tolist()
        raise RuntimeError(f"基准缺少策略收益日: {missing_dates[:5]}")
    excess = ret - bench
    metrics = annualized_metrics(ret)
    bench_metrics = annualized_metrics(bench)
    excess_metrics = annualized_metrics(excess)
    return {
        "year": year, "start": start, "end": end, **metrics,
        "bench_ann_ret": bench_metrics["ann_ret"],
        "excess_ann_ret": excess_metrics["ann_ret"],
        "excess_sharpe": excess_metrics["sharpe"],
        "avg_turnover": aux["avg_turnover"],
        "rebalance_count": aux["rebalance_count"],
    }, ret


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_features(force=False)
    bounds = year_bounds(df, 2016, 2026)
    fold_years = [year for year in range(2018, 2027) if year in bounds]

    for spec in ML_STRATEGIES:
        scores = fit_predict_lgbm(df, spec["horizon"], fold_years)
        df = df.merge(scores, on=["ts_code", "trade_date"], how="left")

    score_names = [f"score_rolling_lgbm_{spec['horizon']}d" for spec in ML_STRATEGIES]
    score_names += [spec["config"].score_name for spec in FORMULA_STRATEGIES]
    caches = {
        "industry_concept": prepare_universe_cache(df, "industry_concept", score_names),
        "industry": prepare_universe_cache(df, "industry", score_names),
    }

    eval_specs = []
    for spec in ML_STRATEGIES:
        cfg = StrategyConfig(
            spec["strategy_id"], "industry_concept", f"score_rolling_lgbm_{spec['horizon']}d",
            spec["top_k"], spec["rebalance_days"], spec["hold_days"],
        )
        eval_specs.append((spec["strategy_id"], spec["display_name"], "ml_walk_forward", cfg, fold_years))
    formula_years = [year for year in range(2016, 2027) if year in bounds]
    for spec in FORMULA_STRATEGIES:
        eval_specs.append((spec["strategy_id"], spec["display_name"], "formula_all_years", spec["config"], formula_years))

    rows: list[dict] = []
    curve_rows: list[dict] = []
    for strategy_id, display_name, validation_kind, cfg, years in eval_specs:
        cache = caches[cfg.universe]
        combined = []
        positive_years = 0
        strategy_year_rows = []
        for year in years:
            start, end = bounds[year]
            row, ret = evaluate_year(cache, cfg, year, start, end)
            row.update({
                "row_type": "year", "strategy_id": strategy_id, "display_name": display_name,
                "validation_kind": validation_kind, "universe": cfg.universe,
                "score_name": cfg.score_name, "top_k": cfg.top_k,
                "rebalance_days": cfg.rebalance_days,
                "train_end": "" if validation_kind.startswith("formula") else f"{year - 1}1231",
            })
            rows.append(row)
            strategy_year_rows.append(row)
            positive_years += int(row["ann_ret"] > 0)
            if not ret.empty:
                combined.append(ret.copy())

        combined_ret = pd.concat(combined).sort_index() if combined else pd.Series(dtype=float)
        metrics = annualized_metrics(combined_ret)
        for date, equity in (1.0 + combined_ret.fillna(0.0)).cumprod().items():
            curve_rows.append({"strategy_id": strategy_id, "date": date, "equity": float(equity)})
        rows.append({
            "row_type": "summary", "strategy_id": strategy_id, "display_name": display_name,
            "validation_kind": validation_kind, "universe": cfg.universe,
            "score_name": cfg.score_name, "top_k": cfg.top_k,
            "rebalance_days": cfg.rebalance_days, "start_year": min(years), "end_year": max(years),
            "valid_years": len(years), "positive_years": positive_years, **metrics,
            "avg_turnover": float(np.nanmean([row["avg_turnover"] for row in strategy_year_rows])),
        })

    pd.DataFrame(rows).to_csv(ROLLING_RESULTS, index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(ROLLING_CURVES, index=False, encoding="utf-8-sig")
    print(f"[done] results={ROLLING_RESULTS}")
    print(f"[done] curves={ROLLING_CURVES}")
    print(f"[done] elapsed={(time.time() - start_time) / 60:.1f}m")


if __name__ == "__main__":
    main()
