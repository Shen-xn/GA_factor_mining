"""每 5 个交易日重建一次 30 日 IC 加权投票模型，并在 2026 年回测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import daily_rank_ic, load_config, prepare_data
from factor_engine import cap_normalized_weights, evaluate


def build_factor_frame(data: pd.DataFrame, library: dict, config: dict) -> pd.DataFrame:
    out = data[["ts_code", "trade_date", config["target"]["name"]]].copy()
    for item in library["factors"]:
        out[item["name"]] = evaluate(item["expression"], data, float(config["factor_mining"]["max_abs_value"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--library", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, _ = prepare_data(config)
    output_dir = Path(config["data"]["output_dir"])
    library_path = Path(args.library).resolve() if args.library else output_dir / "factor_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    factors = [x["name"] for x in library["factors"]]
    if not factors:
        raise RuntimeError("因子库为空，请先运行 build_factor_library.py")
    all_dates = np.array(sorted(data["trade_date"].unique()))
    test_dates = all_dates[(all_dates >= config["split"]["test_start"]) & (all_dates <= config["split"]["test_end"])]
    horizon = int(config["target"]["horizon"])
    rolling = config["rolling_model"]
    lookback = int(rolling["lookback_days"])
    first_test_position = int(np.searchsorted(all_dates, test_dates[0]))
    required_start = all_dates[max(0, first_test_position - horizon - lookback)]
    data = data[
        (data["trade_date"] >= required_start)
        & (data["trade_date"] <= config["split"]["test_end"])
    ].copy()
    frame = build_factor_frame(data, library, config)
    rebalance = int(rolling["rebalance_days"])
    min_days = int(rolling["min_ic_days"])
    min_stocks = int(config["target"]["min_daily_stocks"])
    target = config["target"]["name"]
    scores_all = []
    weights_all = []

    for start in range(0, len(test_dates), rebalance):
        block_dates = test_dates[start:start + rebalance]
        rebalance_date = block_dates[0]
        position = int(np.searchsorted(all_dates, rebalance_date))
        # 在真实时点 t，最近 horizon 日的未来收益尚未兑现，必须从 IC 窗口剔除。
        history_end = position - horizon
        history_start = history_end - lookback
        if history_start < 0:
            continue
        history_dates = all_dates[history_start:history_end]
        history = frame[frame["trade_date"].isin(history_dates)]
        ic_history = pd.DataFrame(index=history_dates, columns=factors, dtype=float)
        for factor in factors:
            ic = daily_rank_ic(history[factor], history[target], history["trade_date"], min_stocks)
            if len(ic) >= min_days:
                ic_history.loc[ic.index, factor] = ic
        half_life = float(rolling.get("ic_half_life", 0.0))
        if half_life > 0:
            age = np.arange(len(ic_history) - 1, -1, -1)
            decay = np.power(0.5, age / half_life)
            weighted_sum = ic_history.mul(decay, axis=0).sum()
            valid_weight = ic_history.notna().mul(decay, axis=0).sum()
            mean_ics = weighted_sum.div(valid_weight.replace(0.0, np.nan)).fillna(0.0)
        else:
            mean_ics = ic_history.mean().fillna(0.0)
        dead = float(rolling["ic_dead_zone"])
        raw = np.sign(mean_ics) * mean_ics.abs().pow(float(rolling["ic_power"]))
        raw[mean_ics.abs() < dead] = 0.0
        top_k = int(rolling.get("top_k_factors", len(raw)))
        if 0 < top_k < len(raw):
            keep = raw.abs().nlargest(top_k).index
            raw.loc[~raw.index.isin(keep)] = 0.0
        weights = cap_normalized_weights(raw, float(rolling["max_factor_weight"]))
        for name in factors:
            weights_all.append({"rebalance_date": rebalance_date, "factor": name, "mean_ic": mean_ics[name], "weight": weights[name]})

        block = frame[frame["trade_date"].isin(block_dates)].copy()
        block["score"] = 0.0
        for factor in factors:
            signal = block.groupby("trade_date", sort=False)[factor].rank(pct=True).mul(2.0).sub(1.0)
            if rolling.get("signal_mode", "linear") == "binary":
                signal = np.sign(signal)
            elif rolling.get("signal_mode") == "tanh":
                signal = np.tanh(2.0 * signal) / np.tanh(2.0)
            block["score"] += signal.fillna(0.0) * weights[factor]
        block["rebalance_date"] = rebalance_date
        scores_all.append(block[["ts_code", "trade_date", "rebalance_date", target, "score"]])
        print(f"[rolling] date={rebalance_date} active={(weights != 0).sum():02d} max_weight={weights.abs().max():.3f}")

    result = pd.concat(scores_all, ignore_index=True).dropna(subset=[target, "score"])
    result["score_rank"] = result.groupby("trade_date", sort=False)["score"].rank(pct=True)
    q_top = 1.0 - float(config["backtest"]["top_quantile"])
    q_bottom = float(config["backtest"]["bottom_quantile"])
    daily_ic = daily_rank_ic(result["score"], result[target], result["trade_date"], min_stocks)
    grouped = result.groupby("trade_date", sort=False)
    summary_daily = grouped.apply(lambda x: pd.Series({
        "rank_ic": x["score"].rank().corr(x[target].rank()),
        "top_return_5d": x.loc[x["score_rank"] >= q_top, target].mean(),
        "bottom_return_5d": x.loc[x["score_rank"] <= q_bottom, target].mean(),
    }), include_groups=False)
    summary_daily["long_short_5d"] = summary_daily["top_return_5d"] - summary_daily["bottom_return_5d"]
    result.to_parquet(output_dir / "test_predictions_2026.parquet", index=False)
    pd.DataFrame(weights_all).to_csv(output_dir / "rolling_factor_weights_2026.csv", index=False, encoding="utf-8-sig")
    summary_daily.to_csv(output_dir / "daily_backtest_2026.csv", encoding="utf-8-sig")
    summary = {
        "rolling_model": rolling,
        "test_days": int(len(summary_daily)),
        "mean_rank_ic": float(daily_ic.mean()),
        "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()) if daily_ic.std() > 0 else None,
        "mean_top_return_5d": float(summary_daily["top_return_5d"].mean()),
        "mean_bottom_return_5d": float(summary_daily["bottom_return_5d"].mean()),
        "mean_long_short_5d": float(summary_daily["long_short_5d"].mean()),
        "long_short_win_rate": float((summary_daily["long_short_5d"] > 0).mean()),
    }
    (output_dir / "backtest_summary_2026.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[result]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
