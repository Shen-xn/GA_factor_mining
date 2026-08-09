"""使用受强正则约束的 LightGBM 做滚动月度横截面预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from backtest_2026 import build_factor_frame
from data_pipeline import daily_rank_ic, load_config, prepare_data


def make_model(
    config: dict, candidate: dict, n_estimators: int, mode: str
) -> lgb.LGBMModel:
    model = config["lightgbm_model"]
    parameters = dict(
        n_estimators=n_estimators,
        learning_rate=float(model["learning_rate"]),
        max_depth=int(candidate["max_depth"]),
        num_leaves=int(candidate["num_leaves"]),
        min_child_samples=int(candidate["min_child_samples"]),
        max_bin=int(model["max_bin"]),
        reg_alpha=float(model["reg_alpha"]),
        reg_lambda=float(model["reg_lambda"]),
        min_split_gain=float(model["min_split_gain"]),
        feature_fraction=float(model["feature_fraction"]),
        bagging_fraction=float(model["bagging_fraction"]),
        bagging_freq=int(model["bagging_freq"]),
        random_state=int(model["random_state"]),
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    if mode == "ranker":
        return lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
            lambdarank_truncation_level=int(model["rank_eval_at"]),
            **parameters,
        )
    return lgb.LGBMRegressor(objective="regression_l2", **parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--mode", choices=("regression", "ranker"), default="regression"
    )
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, _ = prepare_data(config)
    output_dir = Path(config["data"]["output_dir"])
    library = json.loads(
        (output_dir / "factor_library.json").read_text(encoding="utf-8")
    )
    factors = [item["name"] for item in library["factors"]]
    target = config["target"]["name"]
    model_config = config["lightgbm_model"]
    lookback = int(model_config["lookback_days"])
    validation_days = int(model_config["validation_days"])
    rebalance = int(model_config["rebalance_days"])
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
    ranked_features = frame[factors].groupby(
        frame["trade_date"], sort=False
    ).rank(pct=True).fillna(0.5).astype("float32")
    frame["target_rank"] = frame.groupby("trade_date", sort=False)[target].rank(
        pct=True
    ).astype("float32")
    frame["target_relevance"] = np.floor(frame["target_rank"] * 10.0).clip(
        upper=9
    )

    predictions = []
    model_records = []
    saved_models = {}
    for start in range(0, len(test_dates), rebalance):
        block_dates = test_dates[start:start + rebalance]
        rebalance_date = block_dates[0]
        position = int(np.searchsorted(all_dates, rebalance_date))
        history_end = position - horizon
        history_dates = all_dates[history_end - lookback:history_end]
        fit_dates = history_dates[:-validation_days]
        validation_dates = history_dates[-validation_days:]
        fit_mask = frame["trade_date"].isin(fit_dates) & frame["target_rank"].notna()
        validation_mask = (
            frame["trade_date"].isin(validation_dates)
            & frame["target_rank"].notna()
        )
        history_mask = (
            frame["trade_date"].isin(history_dates)
            & frame["target_rank"].notna()
        )
        validation_y = frame.loc[validation_mask, "target_rank"]
        baseline_mae = mean_absolute_error(
            validation_y, np.full(len(validation_y), 0.5)
        )

        best = None
        for candidate in model_config["candidates"]:
            trial = make_model(
                config, candidate, int(model_config["n_estimators"]), args.mode
            )
            callbacks = [
                lgb.early_stopping(
                    int(model_config["early_stopping_rounds"]), verbose=False
                ),
                lgb.log_evaluation(0),
            ]
            if args.mode == "ranker":
                fit_group = frame.loc[fit_mask].groupby(
                    "trade_date", sort=False
                ).size().tolist()
                validation_group = frame.loc[validation_mask].groupby(
                    "trade_date", sort=False
                ).size().tolist()
                trial.fit(
                    ranked_features.loc[fit_mask, factors],
                    frame.loc[fit_mask, "target_relevance"].astype("int8"),
                    group=fit_group,
                    eval_set=[(
                        ranked_features.loc[validation_mask, factors],
                        frame.loc[validation_mask, "target_relevance"].astype("int8"),
                    )],
                    eval_group=[validation_group],
                    eval_at=[int(model_config["rank_eval_at"])],
                    eval_metric="ndcg",
                    callbacks=callbacks,
                )
            else:
                trial.fit(
                    ranked_features.loc[fit_mask, factors],
                    frame.loc[fit_mask, "target_rank"],
                    eval_set=[(
                        ranked_features.loc[validation_mask, factors],
                        validation_y,
                    )],
                    eval_metric="l1",
                    callbacks=callbacks,
                )
            validation_prediction = trial.predict(
                ranked_features.loc[validation_mask, factors],
                num_iteration=trial.best_iteration_,
            )
            if args.mode == "ranker":
                validation_metric = float(
                    trial.best_score_["valid_0"][
                        f"ndcg@{int(model_config['rank_eval_at'])}"
                    ]
                )
                selection_metric = -validation_metric
            else:
                validation_metric = mean_absolute_error(
                    validation_y, validation_prediction
                )
                selection_metric = validation_metric
            choice = (
                selection_metric,
                int(candidate["max_depth"]),
                int(trial.best_iteration_),
                candidate,
            )
            if best is None or choice[:3] < best[:3]:
                best = choice

        selection_metric, _, best_iteration, candidate = best
        validation_metric = (
            -selection_metric if args.mode == "ranker" else selection_metric
        )
        model = make_model(
            config, candidate, max(1, best_iteration), args.mode
        )
        if args.mode == "ranker":
            history_group = frame.loc[history_mask].groupby(
                "trade_date", sort=False
            ).size().tolist()
            model.fit(
                ranked_features.loc[history_mask, factors],
                frame.loc[history_mask, "target_relevance"].astype("int8"),
                group=history_group,
                callbacks=[lgb.log_evaluation(0)],
            )
        else:
            model.fit(
                ranked_features.loc[history_mask, factors],
                frame.loc[history_mask, "target_rank"],
                callbacks=[lgb.log_evaluation(0)],
            )
        block_mask = frame["trade_date"].isin(block_dates)
        part = frame.loc[block_mask, ["ts_code", "trade_date", target]].copy()
        part["score"] = model.predict(ranked_features.loc[block_mask, factors])
        part["rebalance_date"] = rebalance_date
        predictions.append(part)

        gain = pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=factors,
        ).sort_values(ascending=False)
        model_records.append({
            "rebalance_date": rebalance_date,
            "max_depth": int(candidate["max_depth"]),
            "num_leaves": int(candidate["num_leaves"]),
            "min_child_samples": int(candidate["min_child_samples"]),
            "best_iteration": int(best_iteration),
            "mode": args.mode,
            "validation_metric": float(validation_metric),
            "validation_metric_name": (
                f"ndcg@{int(model_config['rank_eval_at'])}"
                if args.mode == "ranker" else "mae"
            ),
            "validation_baseline_mae": float(baseline_mae),
            "validation_mae_improvement": (
                None if args.mode == "ranker"
                else float(baseline_mae - validation_metric)
            ),
            "train_rows": int(history_mask.sum()),
            "top_features": gain.head(5).to_dict(),
        })
        saved_models[rebalance_date] = model
        print(
            f"[lightgbm] date={rebalance_date} depth={candidate['max_depth']} "
            f"leaves={candidate['num_leaves']} iter={best_iteration:03d} "
            f"val_{'ndcg' if args.mode == 'ranker' else 'mae'}="
            f"{validation_metric:.5f} top={gain.index[0]}"
        )

    result = pd.concat(predictions, ignore_index=True).dropna(subset=[target, "score"])
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
        "mode": args.mode,
        "lightgbm_model": model_config,
        "test_days": int(len(daily)),
        "mean_rank_ic": float(daily_ic.mean()),
        "rank_ic_ir": float(daily_ic.mean() / daily_ic.std()),
        "mean_top_return_5d": float(daily["top_return_5d"].mean()),
        "mean_bottom_return_5d": float(daily["bottom_return_5d"].mean()),
        "mean_long_short_5d": float(daily["long_short_5d"].mean()),
        "long_short_win_rate": float((daily["long_short_5d"] > 0).mean()),
    }
    prefix = "lightgbm_ranker" if args.mode == "ranker" else "lightgbm"
    result.to_parquet(
        output_dir / f"{prefix}_test_predictions_2026.parquet", index=False
    )
    daily.to_csv(
        output_dir / f"{prefix}_daily_backtest_2026.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(model_records).to_json(
        output_dir / f"{prefix}_model_records_2026.json",
        orient="records", force_ascii=False, indent=2,
    )
    joblib.dump(saved_models, output_dir / f"{prefix}_models_2026.joblib")
    (output_dir / f"{prefix}_backtest_summary_2026.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[result]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
