"""用滚动月数据训练小决策树，不使用 IC 构建模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_text

from .backtest_2026 import build_factor_frame
from .data_pipeline import daily_rank_ic, load_config, prepare_data


def make_tree(config: dict, depth: int, min_leaf: float) -> DecisionTreeRegressor:
    tree = config["tree_model"]
    return DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=depth,
        min_samples_leaf=min_leaf,
        max_leaf_nodes=int(tree["max_leaf_nodes"]),
        ccp_alpha=float(tree["ccp_alpha"]),
        random_state=int(tree["random_state"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stock/v1.json")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, _ = prepare_data(config)
    output_dir = Path(config["data"]["output_dir"])
    library = json.loads(
        (output_dir / "factor_library.json").read_text(encoding="utf-8")
    )
    factors = [item["name"] for item in library["factors"]]
    target = config["target"]["name"]
    tree_config = config["tree_model"]
    lookback = int(tree_config["lookback_days"])
    validation_days = int(tree_config["validation_days"])
    rebalance = int(tree_config["rebalance_days"])
    horizon = int(config["target"]["horizon"])
    min_stocks = int(config["target"]["min_daily_stocks"])

    all_dates = np.array(sorted(data["trade_date"].unique()))
    test_dates = all_dates[
        (all_dates >= config["split"]["test_start"])
        & (all_dates <= config["split"]["test_end"])
    ]
    first = int(np.searchsorted(all_dates, test_dates[0]))
    required_start = all_dates[max(0, first - horizon - lookback)]
    data = data[
        (data["trade_date"] >= required_start)
        & (data["trade_date"] <= config["split"]["test_end"])
    ].copy()
    frame = build_factor_frame(data, library, config)

    # 输入和标签都按日转为横截面分位数，消除跨时间尺度漂移。
    ranked_features = frame[factors].groupby(
        frame["trade_date"], sort=False
    ).rank(pct=True).fillna(0.5)
    frame["target_rank"] = frame.groupby("trade_date", sort=False)[target].rank(
        pct=True
    )

    predictions = []
    model_records = []
    saved_models = {}
    for start in range(0, len(test_dates), rebalance):
        block_dates = test_dates[start:start + rebalance]
        rebalance_date = block_dates[0]
        position = int(np.searchsorted(all_dates, rebalance_date))
        # 最近 5 日标签未兑现，训练月必须在它们之前结束。
        history_end = position - horizon
        history_dates = all_dates[history_end - lookback:history_end]
        if len(history_dates) < lookback:
            continue
        fit_dates = history_dates[:-validation_days]
        validation_dates = history_dates[-validation_days:]
        fit_mask = frame["trade_date"].isin(fit_dates) & frame["target_rank"].notna()
        validation_mask = (
            frame["trade_date"].isin(validation_dates)
            & frame["target_rank"].notna()
        )
        validation_baseline_mae = mean_absolute_error(
            frame.loc[validation_mask, "target_rank"],
            np.full(int(validation_mask.sum()), 0.5),
        )
        history_mask = (
            frame["trade_date"].isin(history_dates)
            & frame["target_rank"].notna()
        )

        best = None
        for depth in tree_config["max_depth_candidates"]:
            for min_leaf in tree_config["min_leaf_fraction_candidates"]:
                candidate = make_tree(config, int(depth), float(min_leaf))
                candidate.fit(
                    ranked_features.loc[fit_mask, factors],
                    frame.loc[fit_mask, "target_rank"],
                )
                validation_prediction = candidate.predict(
                    ranked_features.loc[validation_mask, factors]
                )
                validation_mae = mean_absolute_error(
                    frame.loc[validation_mask, "target_rank"],
                    validation_prediction,
                )
                choice = (validation_mae, int(depth), -float(min_leaf), candidate)
                if best is None or choice[:3] < best[:3]:
                    best = choice

        _, depth, negative_min_leaf, _ = best
        min_leaf = -negative_min_leaf
        model = make_tree(config, depth, min_leaf)
        model.fit(
            ranked_features.loc[history_mask, factors],
            frame.loc[history_mask, "target_rank"],
        )
        block_mask = frame["trade_date"].isin(block_dates)
        part = frame.loc[block_mask, ["ts_code", "trade_date", target]].copy()
        part["score"] = model.predict(ranked_features.loc[block_mask, factors])
        part["rebalance_date"] = rebalance_date
        predictions.append(part)

        importances = pd.Series(model.feature_importances_, index=factors)
        model_records.append({
            "rebalance_date": rebalance_date,
            "max_depth": depth,
            "min_leaf_fraction": min_leaf,
            "validation_mae": best[0],
            "validation_baseline_mae": validation_baseline_mae,
            "validation_mae_improvement": validation_baseline_mae - best[0],
            "actual_depth": model.get_depth(),
            "leaf_count": model.get_n_leaves(),
            "train_rows": int(history_mask.sum()),
            "top_feature": str(importances.idxmax()),
            "top_feature_importance": float(importances.max()),
            "tree_text": export_text(model, feature_names=factors),
        })
        saved_models[rebalance_date] = model
        print(
            f"[tree] date={rebalance_date} depth={model.get_depth()} "
            f"leaves={model.get_n_leaves()} val_mae={best[0]:.5f} "
            f"top={importances.idxmax()}"
        )

    result = pd.concat(predictions, ignore_index=True)
    result = result.dropna(subset=[target, "score"])
    # 树叶输出是离散值；method="first" 保证每天都有严格的头尾 10%。
    result = result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    result["score_rank"] = result.groupby("trade_date", sort=False)["score"].rank(
        pct=True, method="first"
    )
    daily_ic = daily_rank_ic(
        result["score"], result[target], result["trade_date"], min_stocks
    )
    top_cut = 1.0 - float(config["backtest"]["top_quantile"])
    bottom_cut = float(config["backtest"]["bottom_quantile"])
    daily = result.groupby("trade_date", sort=False).apply(
        lambda x: pd.Series({
            "rank_ic": x["score"].rank().corr(x[target].rank()),
            "top_return_5d": x.loc[x["score_rank"] >= top_cut, target].mean(),
            "bottom_return_5d": x.loc[x["score_rank"] <= bottom_cut, target].mean(),
        }),
        include_groups=False,
    )
    daily["long_short_5d"] = daily["top_return_5d"] - daily["bottom_return_5d"]
    summary = {
        "tree_model": tree_config,
        "test_days": int(len(daily)),
        "mean_rank_ic": float(daily_ic.mean()),
        "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()),
        "mean_top_return_5d": float(daily["top_return_5d"].mean()),
        "mean_bottom_return_5d": float(daily["bottom_return_5d"].mean()),
        "mean_long_short_5d": float(daily["long_short_5d"].mean()),
        "long_short_win_rate": float(
            (daily["long_short_5d"].dropna() > 0).mean()
        ),
    }
    result.to_parquet(output_dir / "tree_test_predictions_2026.parquet", index=False)
    daily.to_csv(output_dir / "tree_daily_backtest_2026.csv", encoding="utf-8-sig")
    pd.DataFrame(model_records).to_json(
        output_dir / "tree_model_records_2026.json",
        orient="records", force_ascii=False, indent=2,
    )
    joblib.dump(saved_models, output_dir / "tree_models_2026.joblib")
    (output_dir / "tree_backtest_summary_2026.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[result]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
