"""低内存复现旧版逐年LightGBM Top5原型。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ...common.paths import ensure_output_dir


FEATURE_COLUMNS = [
    "ret_1d_rank",
    "ret_3d_rank",
    "ret_5d_rank",
    "ret_10d_rank",
    "ret_20d_rank",
    "ret_60d_rank",
    "volatility_10d_rank",
    "volatility_20d_rank",
    "risk_adj_5_20_rank",
    "risk_adj_10_20_rank",
    "risk_adj_20_60_rank",
    "close_pos_20d_rank",
    "drawdown_60d_rank",
    "ma_gap_5_20_rank",
    "ma_gap_10_60_rank",
    "volume_z_20d_rank",
    "turnover_z_20d_rank",
    "range_1d_rank",
]
TARGET_COLUMN = "future_ret_5d_rank"
LEGACY_STRATEGY_ID = "rolling_lgbm5_top5"
LEGACY_TEST_END = "20260529"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annualized_metrics(returns: pd.Series) -> dict[str, float | int]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "days": 0,
            "total_ret": np.nan,
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    curve = (1.0 + returns).cumprod()
    volatility = float(returns.std(ddof=0))
    return {
        "days": int(len(returns)),
        "total_ret": float(curve.iloc[-1] - 1.0),
        "ann_ret": float(curve.iloc[-1] ** (252 / len(returns)) - 1.0),
        "ann_vol": volatility * math.sqrt(252),
        "sharpe": float(returns.mean() / volatility * math.sqrt(252)),
        "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
        "win_rate": float((returns > 0).mean()),
    }


def _load_legacy_panel(path: Path) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_date",
        "type",
        "ret_1d",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    # 旧缓存很大，只投影模型必需列，并尽早过滤到行业和概念板块。
    panel = pd.read_parquet(path, columns=columns, filters=[("type", "in", ["I", "N"])])
    panel["trade_date"] = panel["trade_date"].astype(str)
    # LightGBM包含行采样，必须保留旧缓存原始行序才能复现随机采样。
    return panel.reset_index(drop=True)


def _legacy_year_returns(frame: pd.DataFrame, scores: np.ndarray) -> pd.Series:
    work = frame[["ts_code", "trade_date", "ret_1d"]].copy()
    work["score"] = scores
    score_pivot = work.pivot(index="trade_date", columns="ts_code", values="score").sort_index()
    return_pivot = work.pivot(index="trade_date", columns="ts_code", values="ret_1d").sort_index()
    dates = list(score_pivot.index)
    daily_returns: dict[str, float] = {}
    for index in range(len(dates) - 1):
        score = score_pivot.loc[dates[index]].dropna().sort_values(ascending=False)
        if len(score) < 5:
            continue
        holdings = list(score.head(5).index)
        next_date = dates[index + 1]
        realized = return_pivot.loc[next_date, holdings].dropna()
        # 忠实保留旧实现：收益全部缺失时记作零，用于验证历史结果而非正式策略。
        daily_returns[next_date] = float(realized.mean()) if len(realized) else 0.0
    return pd.Series(daily_returns, dtype=float).sort_index()


def reproduce_legacy(panel_path: Path) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb

    panel = _load_legacy_panel(panel_path)
    rows: list[dict] = []
    yearly_returns: list[pd.Series] = []
    returns_by_year: dict[int, pd.Series] = {}
    for year in range(2018, 2027):
        train_end = f"{year - 1}1231"
        predict_start = f"{year}0101"
        predict_end = LEGACY_TEST_END if year == 2026 else f"{year}1231"
        train_mask = panel["trade_date"].le(train_end) & panel[TARGET_COLUMN].notna()
        predict_mask = panel["trade_date"].between(predict_start, predict_end)
        train = (
            panel.loc[train_mask, [*FEATURE_COLUMNS, TARGET_COLUMN]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        prediction_frame = panel.loc[predict_mask, FEATURE_COLUMNS]
        prediction_frame = prediction_frame.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        if train.empty or prediction_frame.empty:
            continue
        print(
            f"[legacy] year={year} train_rows={len(train):,} "
            f"prediction_rows={len(prediction_frame):,}"
        )
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=160,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42 + year + 5,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        scores = model.predict(prediction_frame).astype("float32")
        year_frame = panel.loc[predict_mask, ["ts_code", "trade_date", "ret_1d"]]
        returns = _legacy_year_returns(year_frame, scores)
        metrics = _annualized_metrics(returns)
        rows.append(
            {
                "row_type": "year",
                "strategy_id": LEGACY_STRATEGY_ID,
                "year": year,
                "start": predict_start,
                "end": predict_end,
                "train_end": train_end,
                **metrics,
            }
        )
        yearly_returns.append(returns)
        returns_by_year[year] = returns.copy()
        del model, train, prediction_frame, scores, year_frame, returns
        gc.collect()

    period_years = {
        "development": range(2018, 2024),
        "selection": range(2024, 2026),
        "legacy_observation": range(2026, 2027),
    }
    for period, years in period_years.items():
        period_returns = pd.concat([returns_by_year[year] for year in years]).sort_index()
        rows.append(
            {
                "row_type": "period",
                "strategy_id": LEGACY_STRATEGY_ID,
                "year": period,
                "start": str(period_returns.index.min()),
                "end": str(period_returns.index.max()),
                "train_end": "annual_expanding",
                **_annualized_metrics(period_returns),
            }
        )

    combined = pd.concat(yearly_returns).sort_index()
    rows.append(
        {
            "row_type": "summary",
            "strategy_id": LEGACY_STRATEGY_ID,
            "year": "",
            "start": "20180101",
            "end": LEGACY_TEST_END,
            "train_end": "annual_expanding",
            **_annualized_metrics(combined),
        }
    )
    metadata = {
        "source_file": panel_path.name,
        "source_size": panel_path.stat().st_size,
        "source_sha256": _sha256(panel_path),
        "legacy_test_end": LEGACY_TEST_END,
        "memory_mode": "projected columns; I/N filter; one yearly model at a time; n_jobs=1",
        "protocol_role": "historical reproduction only; not an executable strategy",
    }
    return pd.DataFrame(rows), metadata


def compare_corrected_protocol(
    legacy_results: pd.DataFrame,
    corrected_score_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """用同为年度扩展训练的修正协议评分隔离协议差异。"""
    from .product_backtest import product_feature_columns, summarize_backtest_period
    from .return_bridge import _direct_topk_backtest
    from .run_experiments import load_feature_subset

    panel = load_feature_subset(product_feature_columns("score_breakout", external_score=True))
    scores = pd.read_parquet(corrected_score_path)
    score_columns = [column for column in scores if column not in {"ts_code", "trade_date"}]
    if len(score_columns) != 1:
        raise ValueError("修正协议年度评分应只有一个评分字段")
    score_name = score_columns[0]
    scored = panel.merge(scores, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    corrected_daily = _direct_topk_backtest(
        scored,
        score_name,
        "20180101",
        LEGACY_TEST_END,
        smoothing_sessions=1,
        cost_bps=0.0,
    )
    periods = {
        "development": ("20180101", "20231231"),
        "selection": ("20240101", "20251231"),
        "legacy_observation": ("20260101", LEGACY_TEST_END),
    }
    legacy_periods = legacy_results[legacy_results["row_type"].eq("period")].set_index("year")
    rows: list[dict] = []
    for period, (start, end) in periods.items():
        _, _, corrected = summarize_backtest_period(
            corrected_daily, pd.DataFrame(), start, end
        )
        legacy = legacy_periods.loc[period]
        rows.append(
            {
                "period": period,
                "start": start,
                "end": end,
                "legacy_close_to_close_total_ret": float(legacy["total_ret"]),
                "corrected_open_to_open_total_ret": float(corrected["total_ret"]),
                "delta_total_ret": float(corrected["total_ret"] - legacy["total_ret"]),
                "legacy_sharpe": float(legacy["sharpe"]),
                "corrected_sharpe": float(corrected["sharpe"]),
                "legacy_max_drawdown": float(legacy["max_drawdown"]),
                "corrected_max_drawdown": float(corrected["max_drawdown"]),
            }
        )
    audit = {
        "comparison_role": "same annual expanding LightGBM schedule; no smoothing; daily Top5; no costs",
        "corrected_score_file": corrected_score_path.name,
        "corrected_score_sha256": _sha256(corrected_score_path),
        "changes_are_joint_not_shapley": True,
        "legacy_protocol_findings": [
            {
                "name": "execution_return_mismatch",
                "legacy": "signal after close but next row ret_1d is prior-close to current-close and includes an untradable overnight move",
                "corrected": "signal at t close; enter t+1 open; exit subsequent open",
            },
            {
                "name": "label_execution_mismatch",
                "legacy": "five-day close-to-close label",
                "corrected": "five-day next-open to later-open label aligned with execution",
            },
            {
                "name": "training_boundary",
                "legacy": "feature date <= cutoff and target non-null; labels may mature after cutoff",
                "corrected": "both feature date and label realization date <= cutoff",
            },
            {
                "name": "rank_universe",
                "legacy": "ranks formed before selecting the I+N universe",
                "corrected": "ranks formed inside the I+N research universe",
            },
            {
                "name": "missing_returns",
                "legacy": "drops missing holdings and can record zero when all are missing",
                "corrected": "requires executable next-open returns and never fabricates zero",
            },
        ],
    }
    return pd.DataFrame(rows), audit


def compare_expected(reproduced: pd.DataFrame, expected_path: Path) -> dict:
    expected = pd.read_csv(expected_path)
    expected = expected[expected["strategy_id"].eq(LEGACY_STRATEGY_ID)].copy()
    expected_year = expected[expected["row_type"].eq("year")].copy()
    expected_year["year"] = pd.to_numeric(expected_year["year"]).astype(int)
    actual_year = reproduced[reproduced["row_type"].eq("year")].copy()
    actual_year["year"] = pd.to_numeric(actual_year["year"]).astype(int)
    joined = actual_year.merge(
        expected_year[["year", "total_ret", "ann_ret", "sharpe", "max_drawdown"]],
        on="year",
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    fields = ("total_ret", "ann_ret", "sharpe", "max_drawdown")
    errors = {
        field: float((joined[f"{field}_actual"] - joined[f"{field}_expected"]).abs().max())
        for field in fields
    }
    tolerance = 1e-6
    return {
        "passed": all(error <= tolerance for error in errors.values()),
        "tolerance": tolerance,
        "compared_years": joined["year"].tolist(),
        "max_absolute_error": errors,
        "expected_file": expected_path.name,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="低内存复现旧版板块高收益原型")
    parser.add_argument("--legacy-panel", type=Path, required=True)
    parser.add_argument("--expected-results", type=Path)
    parser.add_argument("--corrected-annual-scores", type=Path)
    args = parser.parse_args()
    if not args.legacy_panel.is_file():
        raise FileNotFoundError(args.legacy_panel)

    results, metadata = reproduce_legacy(args.legacy_panel)
    validation = None
    if args.expected_results:
        if not args.expected_results.is_file():
            raise FileNotFoundError(args.expected_results)
        validation = compare_expected(results, args.expected_results)
        metadata["expected_validation"] = validation

    output_dir = ensure_output_dir("sector", "prototype_recovery")
    results.to_csv(output_dir / "LEGACY_REPRODUCTION.csv", index=False, encoding="utf-8-sig")
    (output_dir / "METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation is not None:
        (output_dir / "VALIDATION.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not validation["passed"]:
            raise RuntimeError("旧原型复现未通过历史结果核对")
    if args.corrected_annual_scores:
        if not args.corrected_annual_scores.is_file():
            raise FileNotFoundError(args.corrected_annual_scores)
        comparison, audit = compare_corrected_protocol(results, args.corrected_annual_scores)
        comparison.to_csv(
            output_dir / "PROTOCOL_COMPARISON.csv", index=False, encoding="utf-8-sig"
        )
        (output_dir / "PROTOCOL_AUDIT.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[done] {output_dir / 'LEGACY_REPRODUCTION.csv'}")


if __name__ == "__main__":
    main()
