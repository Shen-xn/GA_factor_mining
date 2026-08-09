"""使用遗传算法建立 20 个基础因子库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import load_config, prepare_data
from .factor_engine import (
    canonical,
    crossover,
    evaluate,
    expression_depth,
    expression_text,
    factor_fitness,
    mutate,
    random_expr,
)


def sample_training_data(data: pd.DataFrame, config: dict, quick: bool) -> pd.DataFrame:
    end = config["split"]["factor_library_end"]
    train = data[(data["trade_date"] <= end) & data[config["target"]["name"]].notna()].copy()
    months = sorted(train["trade_date"].str[:6].unique())
    count = 8 if quick else min(int(config["factor_mining"]["sample_months"]), len(months))
    rng = np.random.default_rng(int(config["factor_mining"]["seed"]))
    chosen = set(rng.choice(months, size=count, replace=False).tolist())
    train = train[train["trade_date"].str[:6].isin(chosen)]
    # 搜索阶段每个交易日固定抽样，降低数千个表达式的排序成本。
    per_day = 350 if quick else 700
    train["_sample_key"] = train["ts_code"].map(
        lambda x: int.from_bytes(
            hashlib.blake2b(str(x).encode(), digest_size=8).digest(), "little"
        )
    )
    train = train.sort_values(["trade_date", "_sample_key"]).groupby("trade_date", sort=False).head(per_day)
    train = train.drop(columns="_sample_key").sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    train["_date_code"] = pd.factorize(train["trade_date"], sort=False)[0]
    target = config["target"]["name"]
    train["_target_rank"] = train.groupby("_date_code", sort=False)[target].rank(pct=True)
    return train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stock/v1.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, terminals = prepare_data(config, args.force_data)
    search = sample_training_data(data, config, args.quick)
    mining = config["factor_mining"]
    rng = random.Random(int(mining["seed"]))
    pop_size = 18 if args.quick else int(mining["population_size"])
    generations = 3 if args.quick else int(mining["generations"])
    elite_size = min(6 if args.quick else int(mining["elite_size"]), pop_size)
    max_depth = int(mining["max_depth"])
    population = [random_expr(terminals, max_depth, rng) for _ in range(pop_size)]
    hall: dict[str, tuple[float, object, object]] = {}
    cache: dict[str, tuple] = {}

    print(f"[ga] rows={len(search):,} months={search.trade_date.str[:6].nunique()} population={pop_size} generations={generations}")
    for generation in range(1, generations + 1):
        scored = []
        for expr in population:
            key = canonical(expr)
            if key not in cache:
                cache[key] = factor_fitness(expr, search, config)
            fitness, _ = cache[key]
            scored.append((fitness.score, expr, fitness))
            old = hall.get(key)
            if old is None or fitness.score > old[0]:
                hall[key] = (fitness.score, expr, fitness)
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        print(f"[ga] generation={generation:03d} best={best[0]:.5f} month_ic={best[2].best_month_abs_ic:.5f} unique={len(cache)}")
        elites = [x[1] for x in scored[:elite_size]]
        next_population = list(elites)
        immigrant_count = max(1, int(pop_size * float(mining["random_immigrant_rate"])))
        offspring_limit = pop_size - immigrant_count
        while len(next_population) < offspring_limit:
            parent = rng.choice(elites)
            child = parent
            if rng.random() < float(mining["crossover_rate"]):
                child = crossover(parent, rng.choice(elites), rng)
            if rng.random() < float(mining["mutation_rate"]):
                child = mutate(child, terminals, max_depth, rng)
            if expression_depth(child) > max_depth:
                child = random_expr(terminals, max_depth, rng)
            next_population.append(child)
        while len(next_population) < pop_size:
            next_population.append(random_expr(terminals, max_depth, rng))
        population = next_population

    candidates = sorted(hall.values(), key=lambda x: x[0], reverse=True)[:int(mining["candidate_keep"])]
    selected = []
    selected_values = []
    peak_month_counts: dict[str, int] = {}
    threshold = float(mining["corr_prune_threshold"])
    # 对候选表达式做相关性去重；完整回测仍使用全量股票。
    for score, expr, fitness in candidates:
        if peak_month_counts.get(fitness.peak_month, 0) >= int(mining["max_factors_per_peak_month"]):
            continue
        raw = evaluate(expr, search, float(mining["max_abs_value"]))
        ranks = raw.groupby(search["trade_date"], sort=False).rank(pct=True)
        if any(abs(ranks.corr(previous)) >= threshold for previous in selected_values):
            continue
        selected_values.append(ranks)
        peak_month_counts[fitness.peak_month] = peak_month_counts.get(fitness.peak_month, 0) + 1
        selected.append({
            "name": f"factor_{len(selected)+1:02d}",
            "expression": expr,
            "expression_text": expression_text(expr),
            "depth": expression_depth(expr),
            "search_score": fitness.score,
            "best_month_abs_ic": fitness.best_month_abs_ic,
            "mean_top_abs_ic": fitness.mean_top_abs_ic,
            "valid_months": fitness.valid_months,
            "peak_month": fitness.peak_month,
        })
        if len(selected) >= int(mining["library_size"]):
            break

    output = Path(config["data"]["output_dir"]) / "factor_library.json"
    output.write_text(json.dumps({"factors": selected, "quick": args.quick}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(selected).drop(columns="expression").to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"[done] selected={len(selected)} path={output}")


if __name__ == "__main__":
    main()
