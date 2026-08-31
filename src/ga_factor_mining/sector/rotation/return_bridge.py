#!/usr/bin/env python3
"""把冻结模型的理论Top5逐层桥接到状态化产品收益。"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...common.paths import ensure_output_dir
from .low_risk import DEFAULT_LOW_RISK_CODE, build_low_risk_return_frame, low_risk_data_signature
from .product_backtest import (
    PRODUCT_HISTORY_START,
    _drift_weights,
    _turnover,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
)
from .run_experiments import (
    FEATURE_PROTOCOL_VERSION,
    OBSERVATION_END,
    OBSERVATION_START,
    TRAIN_END,
    UNIVERSES,
    VAL_END,
    VAL_START,
    current_feature_cache_signature,
    load_feature_subset,
)
from .strategy import get_strategy_policy


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_topk_backtest(
    panel: pd.DataFrame,
    score_name: str,
    start: str,
    end: str,
    *,
    smoothing_sessions: int,
    cost_bps: float,
    top_k: int = 5,
) -> pd.DataFrame:
    """每日直接持有TopK；使用与产品相同的开盘收益、权重漂移和成本日历。"""
    sub = panel[panel["type"].isin(UNIVERSES["industry_concept"])].copy()
    sub = sub.sort_values(["trade_date", "ts_code"])
    score_col = "_bridge_score"
    sub[score_col] = sub.groupby("ts_code", sort=False)[score_name].transform(
        lambda values: values.rolling(smoothing_sessions, min_periods=1).mean()
    )
    date_map = (
        sub[["trade_date", "next_open_date", "return_end_date"]]
        .drop_duplicates("trade_date")
        .set_index("trade_date")
    )
    score_pivot = sub.pivot(index="trade_date", columns="ts_code", values=score_col)
    return_pivot = sub.pivot(index="trade_date", columns="ts_code", values="forward_open_ret_1d")
    open_pivot = sub.pivot(index="trade_date", columns="ts_code", values="open")
    signal_dates = [
        date
        for date in score_pivot.index
        if start <= date <= end
        and date in date_map.index
        and pd.notna(date_map.loc[date, "next_open_date"])
        and pd.notna(date_map.loc[date, "return_end_date"])
        and str(date_map.loc[date, "return_end_date"]) <= end
    ]
    if not signal_dates:
        return pd.DataFrame()

    def target_for(signal_date: str) -> dict[str, float]:
        execution_date = str(date_map.loc[signal_date, "next_open_date"])
        valuation_date = str(date_map.loc[signal_date, "return_end_date"])
        score = score_pivot.loc[signal_date].dropna()
        eligible = (
            open_pivot.loc[execution_date].notna()
            & open_pivot.loc[valuation_date].notna()
            & return_pivot.loc[signal_date].notna()
        )
        score = score[eligible.reindex(score.index).fillna(False)]
        if len(score) < top_k:
            raise RuntimeError(f"{signal_date} 可成交候选不足{top_k}个")
        selected = score.sort_values(ascending=False).head(top_k).index
        return {str(code): 1.0 / top_k for code in selected}

    cost_rate = cost_bps / 10_000.0
    first_signal = signal_dates[0]
    live_weights = target_for(first_signal)
    initial_turnover = _turnover({}, live_weights)
    equity = 1.0 - initial_turnover * cost_rate
    peak = max(1.0, equity)
    rows = [
        {
            "date": str(date_map.loc[first_signal, "next_open_date"]),
            "signal_date": first_signal,
            "gross_return": 0.0,
            "sector_contribution": 0.0,
            "low_risk_contribution": 0.0,
            "turnover": initial_turnover,
            "cost": initial_turnover * cost_rate,
            "net_return": equity - 1.0,
            "equity": equity,
            "drawdown": equity / peak - 1.0,
            "regime": "FULL_EXPOSURE",
            "exposure": 1.0,
            "low_risk_weight": 0.0,
            "position_count": top_k,
        }
    ]
    for index in range(1, len(signal_dates)):
        prior_signal = signal_dates[index - 1]
        signal_date = signal_dates[index]
        asset_returns = return_pivot.loc[prior_signal].reindex(live_weights)
        if asset_returns.isna().any():
            raise RuntimeError(f"{prior_signal} 的TopK持仓缺少开盘收益")
        gross_return = float(
            sum(live_weights[code] * asset_returns.loc[code] for code in live_weights)
        )
        pretrade = _drift_weights(live_weights, asset_returns, gross_return)
        target = target_for(signal_date)
        turnover = _turnover(pretrade, target)
        net_return = (1.0 + gross_return) * (1.0 - turnover * cost_rate) - 1.0
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        live_weights = target
        rows.append(
            {
                "date": str(date_map.loc[prior_signal, "return_end_date"]),
                "signal_date": signal_date,
                "gross_return": gross_return,
                "sector_contribution": gross_return,
                "low_risk_contribution": 0.0,
                "turnover": turnover,
                "cost": turnover * cost_rate,
                "net_return": net_return,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "regime": "FULL_EXPOSURE",
                "exposure": 1.0,
                "low_risk_weight": 0.0,
                "position_count": top_k,
            }
        )
    last_signal = signal_dates[-1]
    last_returns = return_pivot.loc[last_signal].reindex(live_weights)
    if last_returns.isna().any():
        raise RuntimeError(f"{last_signal} 的末期TopK持仓缺少开盘收益")
    final_return = float(
        sum(live_weights[code] * last_returns.loc[code] for code in live_weights)
    )
    equity *= 1.0 + final_return
    peak = max(peak, equity)
    rows.append(
        {
            "date": str(date_map.loc[last_signal, "return_end_date"]),
            "signal_date": last_signal,
            "gross_return": final_return,
            "sector_contribution": final_return,
            "low_risk_contribution": 0.0,
            "turnover": 0.0,
            "cost": 0.0,
            "net_return": final_return,
            "equity": equity,
            "drawdown": equity / peak - 1.0,
            "regime": "FULL_EXPOSURE",
            "exposure": 1.0,
            "low_risk_weight": 0.0,
            "position_count": top_k,
        }
    )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    # 收益桥只读取执行账本所需列，避免为了诊断加载整张特征面板。
    panel = load_feature_subset(
        product_feature_columns("score_breakout", external_score=True)
    )
    adaptation_dir = ensure_output_dir("sector", "adaptation")
    score_path = adaptation_dir / "SELECTED_SCORES.parquet"
    score_meta = json.loads((adaptation_dir / "SELECTED.json").read_text(encoding="utf-8"))
    if (
        score_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION
        or score_meta.get("feature_cache_signature") != current_feature_cache_signature()
    ):
        raise RuntimeError("冻结评分与当前特征协议不一致")
    predictions = pd.read_parquet(score_path)
    score_columns = [column for column in predictions if column not in {"ts_code", "trade_date"}]
    if len(score_columns) != 1:
        raise ValueError("冻结评分应只包含一个评分字段")
    score_name = score_columns[0]
    scored = panel.merge(predictions, on=["ts_code", "trade_date"], how="left")
    low_risk_frame = build_low_risk_return_frame(panel)

    simple = get_strategy_policy("simple_v2")
    smoothing_sessions = simple.score_smoothing_sessions
    layers: dict[str, pd.DataFrame] = {
        "raw_top5_gross": _direct_topk_backtest(
            scored,
            score_name,
            PRODUCT_HISTORY_START,
            OBSERVATION_END,
            smoothing_sessions=1,
            cost_bps=0.0,
        ),
        "smoothed_top5_gross": _direct_topk_backtest(
            scored,
            score_name,
            PRODUCT_HISTORY_START,
            OBSERVATION_END,
            smoothing_sessions=smoothing_sessions,
            cost_bps=0.0,
        ),
    }

    product_specs = {
        "stateful_full_exposure_gross": dict(
            strategy_policy=simple,
            cost_bps=0.0,
            low_risk_frame=None,
            use_market_regime=False,
            use_drawdown_cap=False,
        ),
        "market_only_cash_gross": dict(
            strategy_policy=simple,
            cost_bps=0.0,
            low_risk_frame=None,
            use_market_regime=True,
            use_drawdown_cap=False,
        ),
        "risk_control_cash_gross": dict(
            strategy_policy=simple,
            cost_bps=0.0,
            low_risk_frame=None,
            use_market_regime=True,
            use_drawdown_cap=True,
        ),
        "risk_control_low_risk_gross": dict(
            strategy_policy=simple,
            cost_bps=0.0,
            low_risk_frame=low_risk_frame,
            use_market_regime=True,
            use_drawdown_cap=True,
        ),
        "simple_v2_product_20bp": dict(
            strategy_policy=simple,
            cost_bps=20.0,
            low_risk_frame=low_risk_frame,
            use_market_regime=True,
            use_drawdown_cap=True,
        ),
    }
    for layer, kwargs in product_specs.items():
        layers[layer], _, _ = run_product_backtest(
            scored,
            score_name,
            PRODUCT_HISTORY_START,
            OBSERVATION_END,
            **kwargs,
        )

    period_specs = (
        ("development", PRODUCT_HISTORY_START, TRAIN_END),
        ("selection", VAL_START, VAL_END),
        ("same_window_2026", OBSERVATION_START, "20260529"),
        ("observation", OBSERVATION_START, OBSERVATION_END),
    )
    summary_rows: list[dict] = []
    daily_rows: list[pd.DataFrame] = []
    for layer, daily in layers.items():
        tagged = daily.copy()
        tagged.insert(0, "layer", layer)
        daily_rows.append(tagged)
        for period, start, end in period_specs:
            period_daily, _, metrics = summarize_backtest_period(
                daily, pd.DataFrame(), start, end
            )
            gross_total = float((1.0 + period_daily["gross_return"]).prod() - 1.0)
            log_return = float(np.log1p(period_daily["net_return"]).sum())
            summary_rows.append(
                {
                    "period": period,
                    "layer": layer,
                    "boundary_mode": "continuous_carry",
                    "first_return_date": period_daily["date"].min(),
                    "last_return_date": period_daily["date"].max(),
                    "gross_total_ret": gross_total,
                    "log_return": log_return,
                    **metrics,
                }
            )
    summary = pd.DataFrame(summary_rows)
    daily_output = pd.concat(daily_rows, ignore_index=True)
    bridge_pairs = (
        ("raw_top5_gross", "smoothed_top5_gross", "score_smoothing"),
        ("smoothed_top5_gross", "stateful_full_exposure_gross", "holding_rules"),
        ("stateful_full_exposure_gross", "market_only_cash_gross", "market_regime_control"),
        ("market_only_cash_gross", "risk_control_cash_gross", "portfolio_drawdown_cap"),
        ("risk_control_cash_gross", "risk_control_low_risk_gross", "low_risk_residual_asset"),
        ("risk_control_low_risk_gross", "simple_v2_product_20bp", "trading_cost_20bp"),
    )
    bridge_rows: list[dict] = []
    for period, _, _ in period_specs:
        indexed = summary[summary["period"].eq(period)].set_index("layer")
        for order, (source, target, component) in enumerate(bridge_pairs, start=1):
            bridge_rows.append(
                {
                    "period": period,
                    "comparison_order": order,
                    "from_layer": source,
                    "to_layer": target,
                    "component": component,
                    "delta_log_return": float(indexed.loc[target, "log_return"] - indexed.loc[source, "log_return"]),
                    "delta_total_ret": float(indexed.loc[target, "total_ret"] - indexed.loc[source, "total_ret"]),
                    "delta_sharpe": float(indexed.loc[target, "sharpe"] - indexed.loc[source, "sharpe"]),
                    "delta_max_drawdown": float(indexed.loc[target, "max_drawdown"] - indexed.loc[source, "max_drawdown"]),
                    "delta_avg_turnover": float(indexed.loc[target, "avg_turnover"] - indexed.loc[source, "avg_turnover"]),
                }
            )

    output_dir = ensure_output_dir("sector", "return_bridge")
    summary.to_csv(output_dir / "RETURN_BRIDGE_SUMMARY.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bridge_rows).to_csv(
        output_dir / "RETURN_BRIDGE.csv", index=False, encoding="utf-8-sig"
    )
    daily_output.to_parquet(output_dir / "RETURN_BRIDGE_DAILY.parquet", index=False)
    layer_dates = {
        layer: tuple(frame["date"].astype(str)) for layer, frame in layers.items()
    }
    reference_dates = next(iter(layer_dates.values()))
    same_dates = all(dates == reference_dates for dates in layer_dates.values())
    gross_error = float(
        (
            daily_output["gross_return"]
            - daily_output["sector_contribution"]
            - daily_output["low_risk_contribution"]
        ).abs().max()
    )
    net_error = float(
        (
            daily_output["net_return"]
            - (
                (1.0 + daily_output["gross_return"])
                * (1.0 - daily_output["cost"])
                - 1.0
            )
        ).abs().max()
    )
    duplicate_dates = int(
        sum(frame["date"].duplicated().sum() for frame in layers.values())
    )
    validation_checks = {
        "same_return_dates_across_layers": same_dates,
        "no_duplicate_layer_dates": duplicate_dates == 0,
        "gross_contribution_identity": gross_error < 1e-12,
        "net_return_identity": net_error < 1e-12,
    }
    (output_dir / "VALIDATION.json").write_text(
        json.dumps(
            {
                "passed": all(validation_checks.values()),
                "checks": validation_checks,
                "layer_count": len(layers),
                "return_days_per_layer": len(reference_dates),
                "first_return_date": reference_dates[0],
                "last_return_date": reference_dates[-1],
                "max_gross_identity_error": gross_error,
                "max_net_identity_error": net_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "METADATA.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "score_name": score_name,
                "score_file_sha256": _file_sha256(score_path),
                "period": [reference_dates[0], reference_dates[-1]],
                "requested_end": OBSERVATION_END,
                "legacy_comparison_end": "20260529",
                "boundary_mode": "continuous_carry",
                "observation_used_for_selection": False,
                "cost_bps": 20.0,
                "policy_name": "simple_v2",
                "score_smoothing_sessions": smoothing_sessions,
                "low_risk_code": DEFAULT_LOW_RISK_CODE,
                "low_risk_data_signature": low_risk_data_signature(),
                "interpretation": "fixed-order scenario bridge; deltas are not independent Shapley contributions",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {output_dir / 'RETURN_BRIDGE_SUMMARY.csv'}")


if __name__ == "__main__":
    main()
