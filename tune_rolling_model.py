"""一次计算因子值，扫描滚动可靠度与投票参数。"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_2026 import build_factor_frame
from data_pipeline import daily_rank_ic, load_config, prepare_data
from factor_engine import cap_normalized_weights


def reliability_weights(
    ic_history: pd.DataFrame,
    power: float,
    dead_zone: float,
    half_life: float,
    top_k: int,
    cap: float,
) -> pd.Series:
    if half_life > 0:
        age = np.arange(len(ic_history) - 1, -1, -1)
        decay = np.power(0.5, age / half_life)
        mean_ic = ic_history.mul(decay, axis=0).sum() / ic_history.notna().mul(decay, axis=0).sum()
    else:
        mean_ic = ic_history.mean()
    raw = np.sign(mean_ic) * mean_ic.abs().pow(power)
    raw[mean_ic.abs() < dead_zone] = 0.0
    if top_k > 0 and top_k < len(raw):
        keep = raw.abs().nlargest(top_k).index
        raw.loc[~raw.index.isin(keep)] = 0.0
    return cap_normalized_weights(raw, cap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, _ = prepare_data(config)
    output_dir = Path(config["data"]["output_dir"])
    library = json.loads((output_dir / "factor_library.json").read_text(encoding="utf-8"))
    factors = [item["name"] for item in library["factors"]]
    target = config["target"]["name"]
    rolling = config["rolling_model"]
    horizon = int(config["target"]["horizon"])
    lookback = int(rolling["lookback_days"])
    rebalance = int(rolling["rebalance_days"])
    min_stocks = int(config["target"]["min_daily_stocks"])
    all_dates = np.array(sorted(data["trade_date"].unique()))
    test_dates = all_dates[(all_dates >= config["split"]["test_start"]) & (all_dates <= config["split"]["test_end"])]
    first = int(np.searchsorted(all_dates, test_dates[0]))
    required_start = all_dates[max(0, first - horizon - lookback)]
    data = data[(data["trade_date"] >= required_start) & (data["trade_date"] <= config["split"]["test_end"])].copy()
    frame = build_factor_frame(data, library, config)

    ic_table = pd.DataFrame(index=sorted(frame["trade_date"].unique()), columns=factors, dtype=float)
    for factor in factors:
        ic_table[factor] = daily_rank_ic(frame[factor], frame[target], frame["trade_date"], min_stocks)
    linear_signals = frame[factors].groupby(frame["trade_date"], sort=False).rank(pct=True).mul(2.0).sub(1.0)
    signal_modes = {
        "linear": linear_signals,
        "binary": np.sign(linear_signals),
        "tanh": np.tanh(2.0 * linear_signals) / np.tanh(2.0),
    }

    combinations = list(itertools.product(
        (1.0, 1.5),
        (0.0, 0.01, 0.03),
        (2.0, 3.0, 5.0, 10.0),
        (10, 20),
        tuple(signal_modes),
    ))
    rows = []
    for power, dead_zone, half_life, top_k, signal_mode in combinations:
        factor_signals = signal_modes[signal_mode]
        predictions = []
        for start in range(0, len(test_dates), rebalance):
            block_dates = test_dates[start:start + rebalance]
            position = int(np.searchsorted(all_dates, block_dates[0]))
            history_end = position - horizon
            history_dates = all_dates[history_end - lookback:history_end]
            history_ic = ic_table.reindex(history_dates)
            weights = reliability_weights(
                history_ic, power, dead_zone, half_life, top_k,
                float(rolling["max_factor_weight"]),
            )
            mask = frame["trade_date"].isin(block_dates)
            score = factor_signals.loc[mask, factors].fillna(0.0).dot(weights)
            part = frame.loc[mask, ["trade_date", target]].copy()
            part["score"] = score
            predictions.append(part)
        prediction = pd.concat(predictions).dropna()
        daily = daily_rank_ic(prediction["score"], prediction[target], prediction["trade_date"], min_stocks)
        prediction["rank"] = prediction.groupby("trade_date")["score"].rank(pct=True)
        grouped = prediction.groupby("trade_date", sort=False)
        spread = grouped.apply(
            lambda x: x.loc[x["rank"] >= 0.9, target].mean()
            - x.loc[x["rank"] <= 0.1, target].mean(),
            include_groups=False,
        )
        rows.append({
            "power": power, "dead_zone": dead_zone, "half_life": half_life,
            "top_k": top_k, "signal_mode": signal_mode,
            "mean_rank_ic": daily.mean(),
            "rank_ic_ir": daily.mean() / daily.std(),
            "mean_long_short_5d": spread.mean(),
            "long_short_win_rate": (spread > 0).mean(),
            "early_2026_ic": daily[daily.index <= "20260331"].mean(),
            "late_2026_ic": daily[daily.index >= "20260401"].mean(),
        })
    result = pd.DataFrame(rows).sort_values(
        ["mean_rank_ic", "mean_long_short_5d"], ascending=False
    )
    result.to_csv(output_dir / "rolling_parameter_sweep_2026.csv", index=False, encoding="utf-8-sig")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
