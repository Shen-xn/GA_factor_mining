"""遗传表达式、适应度计算和因子去重。"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline import daily_rank_ic

UNARY = ("neg", "abs", "signed_log", "signed_sqrt")
BINARY = ("add", "sub", "mul", "div", "min", "max")


def canonical(expr: Any) -> str:
    return json.dumps(expr, ensure_ascii=False, separators=(",", ":"))


def expression_text(expr: Any) -> str:
    if isinstance(expr, str):
        return expr
    op = expr[0]
    if op in UNARY:
        return f"{op}({expression_text(expr[1])})"
    return f"{op}({expression_text(expr[1])},{expression_text(expr[2])})"


def expression_depth(expr: Any) -> int:
    if isinstance(expr, str):
        return 0
    return 1 + max(expression_depth(child) for child in expr[1:])


def expression_nodes(expr: Any) -> int:
    if isinstance(expr, str):
        return 1
    return 1 + sum(expression_nodes(child) for child in expr[1:])


def evaluate(expr: Any, frame: pd.DataFrame, max_abs: float = 1e6) -> pd.Series:
    if isinstance(expr, str):
        return frame[expr].astype("float64")
    op = expr[0]
    a = evaluate(expr[1], frame, max_abs)
    if op == "neg":
        out = -a
    elif op == "abs":
        out = a.abs()
    elif op == "signed_log":
        out = np.sign(a) * np.log1p(a.abs())
    elif op == "signed_sqrt":
        out = np.sign(a) * np.sqrt(a.abs())
    else:
        b = evaluate(expr[2], frame, max_abs)
        if op == "add": out = a + b
        elif op == "sub": out = a - b
        elif op == "mul": out = a * b
        elif op == "div": out = a / b.where(b.abs() > 1e-8)
        elif op == "min": out = np.minimum(a, b)
        elif op == "max": out = np.maximum(a, b)
        else: raise ValueError(f"未知算子: {op}")
    return pd.Series(np.clip(out, -max_abs, max_abs), index=frame.index).replace([np.inf, -np.inf], np.nan)


def random_expr(terminals: list[str], max_depth: int, rng: random.Random, depth: int = 0) -> Any:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.30):
        return rng.choice(terminals)
    if rng.random() < 0.35:
        return [rng.choice(UNARY), random_expr(terminals, max_depth, rng, depth + 1)]
    return [rng.choice(BINARY), random_expr(terminals, max_depth, rng, depth + 1), random_expr(terminals, max_depth, rng, depth + 1)]


def mutate(expr: Any, terminals: list[str], max_depth: int, rng: random.Random) -> Any:
    if isinstance(expr, str) or rng.random() < 0.25:
        return random_expr(terminals, max_depth, rng)
    child = copy.deepcopy(expr)
    slot = rng.randrange(1, len(child))
    child[slot] = mutate(child[slot], terminals, max_depth - 1 if max_depth > 1 else 1, rng)
    return child


def crossover(left: Any, right: Any, rng: random.Random) -> Any:
    if isinstance(left, str) or rng.random() < 0.30:
        return copy.deepcopy(right)
    child = copy.deepcopy(left)
    slot = rng.randrange(1, len(child))
    child[slot] = crossover(child[slot], right, rng)
    return child


@dataclass
class Fitness:
    score: float
    best_month_abs_ic: float
    mean_top_abs_ic: float
    daily_ic_std: float
    valid_months: int
    peak_month: str


def factor_fitness(expr: Any, data: pd.DataFrame, config: dict) -> tuple[Fitness, pd.Series]:
    mining = config["factor_mining"]
    values = evaluate(expr, data, float(mining["max_abs_value"]))
    if values.notna().sum() == 0 or float(values.std(skipna=True)) < float(mining["min_factor_std"]):
        return Fitness(-1e9, 0.0, 0.0, 0.0, 0, ""), pd.Series(dtype=float)
    if "_target_rank" in data and "_date_code" in data:
        daily = fast_daily_rank_ic(
            values, data, int(config["target"]["min_daily_stocks"])
        ).dropna()
    else:
        daily = daily_rank_ic(
            values, data[config["target"]["name"]], data["trade_date"],
            int(config["target"]["min_daily_stocks"]),
        ).dropna()
    if daily.empty:
        return Fitness(-1e9, 0.0, 0.0, 0.0, 0, ""), daily
    monthly = daily.groupby(daily.index.str[:6]).agg(["mean", "count"])
    monthly = monthly[monthly["count"] >= int(mining["min_month_days"])]
    if monthly.empty:
        return Fitness(-1e9, 0.0, 0.0, float(daily.std()), 0, ""), daily
    top = monthly["mean"].abs().nlargest(int(mining["top_months_for_score"]))
    # 强调阶段性有效，同时轻度惩罚完全由日噪声造成的尖峰。
    score = float(top.mean() - 0.05 * daily.std() - 0.0005 * expression_nodes(expr))
    return Fitness(
        score, float(top.max()), float(top.mean()), float(daily.std()),
        len(monthly), str(top.index[0]),
    ), daily


def fast_daily_rank_ic(
    values: pd.Series, data: pd.DataFrame, min_stocks: int
) -> pd.Series:
    """复用预计算标签排名，以 NumPy 聚合逐日 IC。"""
    factor_rank = values.groupby(data["_date_code"], sort=False).rank(pct=True)
    target_rank = data["_target_rank"]
    valid = factor_rank.notna() & target_rank.notna()
    if not valid.any():
        return pd.Series(dtype=float)

    codes = data.loc[valid, "_date_code"].to_numpy(dtype=np.int64)
    x = factor_rank.loc[valid].to_numpy(dtype=np.float64)
    y = target_rank.loc[valid].to_numpy(dtype=np.float64)
    size = int(data["_date_code"].max()) + 1
    count = np.bincount(codes, minlength=size).astype(np.float64)
    sum_x = np.bincount(codes, weights=x, minlength=size)
    sum_y = np.bincount(codes, weights=y, minlength=size)
    sum_x2 = np.bincount(codes, weights=x * x, minlength=size)
    sum_y2 = np.bincount(codes, weights=y * y, minlength=size)
    sum_xy = np.bincount(codes, weights=x * y, minlength=size)
    safe_count = np.maximum(count, 1.0)
    covariance = sum_xy - sum_x * sum_y / safe_count
    variance_x = sum_x2 - sum_x * sum_x / safe_count
    variance_y = sum_y2 - sum_y * sum_y / safe_count
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full(size, np.nan),
        where=(denominator > 0) & (count >= min_stocks),
    )
    dates = data[["_date_code", "trade_date"]].drop_duplicates("_date_code")
    dates = dates.sort_values("_date_code")["trade_date"].to_numpy()
    return pd.Series(correlation, index=dates)


def cap_normalized_weights(raw: pd.Series, cap: float) -> pd.Series:
    """反复截断后归一，确保单因子权重不会突破上限。"""
    weights = raw.copy().fillna(0.0)
    total = weights.abs().sum()
    if total <= 0:
        return weights
    weights /= total
    for _ in range(10):
        over = weights.abs() > cap
        if not over.any():
            break
        weights.loc[over] = np.sign(weights.loc[over]) * cap
        free = ~over
        remaining = 1.0 - weights.loc[over].abs().sum()
        free_total = weights.loc[free].abs().sum()
        if free_total <= 0 or remaining <= 0:
            break
        weights.loc[free] *= remaining / free_total
    return weights
