from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ...common.expression_tree import (
    canonical,
    depth,
    expression_text,
    get_subtree,
    nodes,
    paths,
    replace_subtree,
    valid_expression,
)

try:
    import markdown
except ImportError:  # pragma: no cover
    markdown = None


ROOT = Path(__file__).resolve().parent


UNARY = ("neg", "abs", "signed_log", "signed_sqrt")
BINARY = ("add", "sub", "mul", "div", "min", "max")
TEMPORAL = ("delta_5", "slope_5", "mean_5", "std_20", "zscore_20", "ts_rank_20")


def expr_text(expr) -> str:
    return expression_text(expr, separator=", ")


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def num(x: float, digits: int = 4) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def load_config(path: str | Path) -> dict:
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for key in ("feature_panel", "artifacts", "reports"):
        cfg["paths"][key] = str((p.parent / cfg["paths"][key]).resolve())
    Path(cfg["paths"]["artifacts"]).mkdir(parents=True, exist_ok=True)
    return cfg


def terminal_category(name: str, cfg: dict) -> str:
    for cat, names in cfg["terminal_groups"].items():
        if name in names:
            return cat
    return "other"


def expression_category(expr, cfg: dict) -> str:
    cats: list[str] = []

    def visit(x) -> None:
        if isinstance(x, str):
            cats.append(terminal_category(x, cfg))
        else:
            for child in x[1:]:
                visit(child)

    visit(expr)
    if not cats:
        return "other"
    return max(set(cats), key=cats.count)


class Evaluator:
    """表达式逐行计算器。时间算子只允许作用于基础字段。"""

    def __init__(self, frame: pd.DataFrame, max_abs: float = 1e6, temporal_cache_size: int = 32):
        self.df = frame
        self.max_abs = max_abs
        self.temporal_cache_size = temporal_cache_size
        self.temporal_cache: OrderedDict[tuple[str, str], pd.Series] = OrderedDict()

    def temporal(self, op: str, col: str) -> pd.Series:
        key = (op, col)
        if key in self.temporal_cache:
            self.temporal_cache.move_to_end(key)
            return self.temporal_cache[key]
        x = self.df[col].astype(float)
        g = self.df.groupby("ts_code", sort=False)[col]
        if op == "delta_5":
            out = x - g.shift(5)
        elif op == "slope_5":
            out = (-2 * g.shift(4) - g.shift(3) + g.shift(1) + 2 * x) / 10.0
        elif op == "mean_5":
            out = g.transform(lambda s: s.rolling(5, min_periods=3).mean())
        elif op == "std_20":
            out = g.transform(lambda s: s.rolling(20, min_periods=5).std())
        elif op == "zscore_20":
            mean = g.transform(lambda s: s.rolling(20, min_periods=5).mean())
            std = g.transform(lambda s: s.rolling(20, min_periods=5).std())
            out = (x - mean) / std.replace(0, np.nan)
        elif op == "ts_rank_20":
            out = g.transform(lambda s: s.rolling(20, min_periods=5).rank(pct=True))
        else:  # pragma: no cover
            raise ValueError(op)
        out = pd.Series(out, index=self.df.index).replace([np.inf, -np.inf], np.nan).astype("float32")
        self.temporal_cache[key] = out
        self.temporal_cache.move_to_end(key)
        while len(self.temporal_cache) > self.temporal_cache_size:
            self.temporal_cache.popitem(last=False)
        return out

    def eval(self, expr) -> pd.Series:
        if isinstance(expr, str):
            return self.df[expr].astype(float)
        op = expr[0]
        if op in TEMPORAL:
            return self.temporal(op, expr[1])
        a = self.eval(expr[1])
        if op == "neg":
            out = -a
        elif op == "abs":
            out = a.abs()
        elif op == "signed_log":
            out = np.sign(a) * np.log1p(a.abs())
        elif op == "signed_sqrt":
            out = np.sign(a) * np.sqrt(a.abs())
        else:
            b = self.eval(expr[2])
            if op == "add":
                out = a + b
            elif op == "sub":
                out = a - b
            elif op == "mul":
                out = a * b
            elif op == "div":
                out = a / b.where(b.abs() > 1e-8)
            elif op == "min":
                out = np.minimum(a, b)
            elif op == "max":
                out = np.maximum(a, b)
            else:  # pragma: no cover
                raise ValueError(op)
        return pd.Series(np.clip(out, -self.max_abs, self.max_abs), index=self.df.index).replace(
            [np.inf, -np.inf], np.nan
        )


class AlphaScorer:
    """用未来10日TopK超额收益评价单个因子。"""

    def __init__(self, frame: pd.DataFrame, cfg: dict, mask: pd.Series):
        self.frame = frame.loc[mask].sort_values(["trade_date", "ts_code"])
        self.k = int(cfg["target"]["primary_top_k"])
        self.min_daily = int(cfg["ga"]["min_daily_sectors"])
        self.min_month_days = int(cfg["ga"]["min_month_days"])
        self.target_return = cfg["target"]["return_column"]
        self.target_rank = cfg["target"]["rank_column"]
        self.groups = [g.index.to_numpy() for _, g in self.frame.groupby("trade_date", sort=False)]
        self.dates = [d for d, _ in self.frame.groupby("trade_date", sort=False)]
        self.months = np.array([d[:6] for d in self.dates])
        self.future = self.frame[self.target_return].to_numpy(dtype=float)
        self.rank = self.frame[self.target_rank].to_numpy(dtype=float)
        self.relevance = self._relevance(self.rank, cfg["target"]["relevance_edges"])

    @staticmethod
    def _relevance(rank: np.ndarray, edges: list[float]) -> np.ndarray:
        out = np.zeros(len(rank), dtype=np.int8)
        for edge in edges:
            out += np.nan_to_num(rank >= edge).astype(np.int8)
        return out

    def _daily_for_direction(self, values: np.ndarray, sign: int, top_k: int) -> pd.DataFrame:
        rows = []
        offset = 0
        signed = sign * values
        for date, idx in zip(self.dates, self.groups):
            n = len(idx)
            factor = signed[offset : offset + n]
            future = self.future[offset : offset + n]
            rank = self.rank[offset : offset + n]
            rel = self.relevance[offset : offset + n]
            offset += n
            valid = np.isfinite(factor) & np.isfinite(future) & np.isfinite(rank)
            if valid.sum() < max(self.min_daily, top_k):
                rows.append((date, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            fv = factor[valid]
            fr = future[valid]
            rk = rank[valid]
            rv = rel[valid]
            if np.nanstd(fv) < 1e-12:
                rows.append((date, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            kk = min(top_k, len(fv))
            selected = np.argpartition(fv, -kk)[-kk:]
            top_alpha = float(np.nanmean(fr[selected]) - np.nanmean(fr))
            top_ret = float(np.nanmean(fr[selected]))
            ic = np.nan
            if np.nanstd(rk) > 1e-12:
                ic = float(np.corrcoef(rankdata(fv), rankdata(rk))[0, 1])
            ideal = np.sort(rv)[::-1][:kk]
            order = selected[np.argsort(fv[selected])[::-1]]
            weights = 1 / np.log2(np.arange(2, kk + 2))
            denom = np.sum((2**ideal - 1) * weights)
            ndcg = float(np.sum((2 ** rv[order] - 1) * weights) / denom) if denom > 0 else np.nan
            rows.append((date, top_alpha, top_ret, float(np.nanmean(rk[selected])), ic, ndcg))
        return pd.DataFrame(rows, columns=["trade_date", "alpha", "top_return", "mean_rank", "rank_ic", "ndcg"])

    def score(self, value: pd.Series) -> tuple[float, dict, pd.DataFrame]:
        vals = value.loc[self.frame.index].to_numpy(dtype=float)
        best = None
        best_daily = None
        for sign in (1, -1):
            daily = self._fast_alpha_daily(vals, sign, self.k)
            daily["month"] = daily["trade_date"].str[:6]
            monthly = daily.groupby("month").agg(alpha=("alpha", "mean"), days=("alpha", "count"))
            monthly = monthly[monthly["days"] >= self.min_month_days]
            if monthly.empty:
                candidate = (-1e9, {"direction": sign, "peak_month": "", "raw_alpha": np.nan, "daily_alpha_std": np.nan})
            else:
                top3 = monthly["alpha"].nlargest(min(3, len(monthly)))
                contiguous = monthly["alpha"].rolling(2).mean().max() if len(monthly) >= 2 else top3.iloc[0]
                daily_std = float(daily["alpha"].std(skipna=True))
                long_mean = float(monthly["alpha"].mean())
                positive_ratio = float((monthly["alpha"] > 0).mean())
                # 板块主升浪因子不能只奖励极少数历史月份，否则很容易挖到特殊行情。
                # 对峰值项做上限截断，并提高长期均值和正月份比例权重。
                top3_capped = min(float(top3.mean()), 0.03)
                contiguous_capped = min(float(contiguous), 0.03)
                raw = float(
                    1.50 * long_mean
                    + 0.50 * contiguous_capped
                    + 0.50 * top3_capped
                    + 0.02 * (positive_ratio - 0.50)
                )
                candidate = (
                    raw,
                    {
                        "direction": sign,
                        "peak_month": str(top3.index[0]),
                        "raw_alpha": raw,
                        "top3_month_alpha": float(top3.mean()),
                        "best_contiguous_2m_alpha": float(contiguous),
                        "long_mean_alpha": long_mean,
                        "positive_month_ratio": positive_ratio,
                        "daily_alpha_std": daily_std,
                    },
                )
            if best is None or candidate[0] > best[0]:
                best = candidate
                best_daily = daily
        assert best is not None and best_daily is not None
        return best[0], best[1], best_daily

    def _fast_alpha_daily(self, values: np.ndarray, sign: int, top_k: int) -> pd.DataFrame:
        """GA 搜索只需要 TopK alpha，用向量化分组排名避免逐日 Python 循环。"""
        signed = pd.Series(sign * values, index=self.frame.index)
        future = pd.Series(self.future, index=self.frame.index)
        date = self.frame["trade_date"]
        valid = signed.notna() & future.notna() & np.isfinite(signed.to_numpy()) & np.isfinite(future.to_numpy())
        if not valid.any():
            return pd.DataFrame(columns=["trade_date", "alpha", "top_return"])
        score_rank = signed[valid].groupby(date[valid]).rank(method="first", ascending=False)
        top_mask = score_rank <= top_k
        daily_top = future[valid][top_mask].groupby(date[valid][top_mask]).mean()
        daily_universe = future[valid].groupby(date[valid]).mean()
        daily_count = future[valid].groupby(date[valid]).count()
        daily = pd.DataFrame(
            {
                "trade_date": daily_universe.index.astype(str),
                "alpha": (daily_top - daily_universe).reindex(daily_universe.index).to_numpy(),
                "top_return": daily_top.reindex(daily_universe.index).to_numpy(),
                "count": daily_count.to_numpy(),
            }
        )
        daily.loc[daily["count"] < self.min_daily, ["alpha", "top_return"]] = np.nan
        return daily.drop(columns=["count"])

    def diagnostics(self, value: pd.Series, direction: int, top_k: int) -> dict:
        daily = self._daily_for_direction(value.loc[self.frame.index].to_numpy(dtype=float), direction, top_k)
        daily["month"] = daily["trade_date"].str[:6]
        monthly = daily.groupby("month").agg(alpha=("alpha", "mean"), top_return=("top_return", "mean"))
        return {
            f"top{top_k}_alpha": float(daily["alpha"].mean(skipna=True)),
            f"top{top_k}_return": float(daily["top_return"].mean(skipna=True)),
            f"top{top_k}_mean_rank": float(daily["mean_rank"].mean(skipna=True)),
            f"top{top_k}_ndcg": float(daily["ndcg"].mean(skipna=True)),
            f"top{top_k}_positive_month_ratio": float((monthly["alpha"] > 0).mean()) if len(monthly) else np.nan,
        }


def random_expr(terminals: list[str], cfg: dict, rng: random.Random, d: int = 0, group: str | None = None):
    max_depth = int(cfg["ga"]["max_depth"])
    temporal = tuple(cfg["operators"]["temporal"])
    unary = tuple(cfg["operators"]["unary"])
    binary = tuple(cfg["operators"]["binary"])
    if d >= max_depth or (d > 0 and rng.random() < 0.30):
        return rng.choice(terminals)
    if temporal and rng.random() < 0.18:
        return [rng.choice(temporal), rng.choice(terminals)]
    if rng.random() < 0.35:
        return [rng.choice(unary), random_expr(terminals, cfg, rng, d + 1, group)]
    return [
        rng.choice(binary),
        random_expr(terminals, cfg, rng, d + 1, group),
        random_expr(terminals, cfg, rng, d + 1, group),
    ]


def get(expr, path):
    return get_subtree(expr, path)


def put(expr, path, value):
    return replace_subtree(expr, path, value)


def valid_expr(expr) -> bool:
    return valid_expression(expr, TEMPORAL)


def crossover(a, b, rng: random.Random):
    return put(a, rng.choice(paths(a)), get(b, rng.choice(paths(b))))


def mutate(expr, terminals: list[str], cfg: dict, rng: random.Random):
    kind = rng.choice(("terminal", "operator", "subtree"))
    p = rng.choice(paths(expr))
    node = get(expr, p)
    if kind == "terminal":
        return put(expr, p, rng.choice(terminals))
    if kind == "operator" and not isinstance(node, str):
        new_node = copy.deepcopy(node)
        choices = tuple(cfg["operators"]["unary"]) if len(node) == 2 else tuple(cfg["operators"]["binary"])
        if node[0] in TEMPORAL:
            choices = tuple(cfg["operators"]["temporal"])
        new_node[0] = rng.choice(choices)
        return put(expr, p, new_node)
    return put(expr, p, random_expr(terminals, cfg, rng))


@dataclass
class Candidate:
    fitness: float
    expr: object
    direction: int
    peak_month: str
    raw_alpha: float
    top3_month_alpha: float
    contiguous_alpha: float
    daily_alpha_std: float


def build_population(terminals: list[str], cfg: dict, rng: random.Random) -> list:
    pop_size = int(cfg["ga"]["population_size"])
    groups = ["trend", "breakout", "volatility", "activity"]
    population = []
    for group in groups:
        group_terms = [x for x in terminals if terminal_category(x, cfg) == group]
        if not group_terms:
            group_terms = terminals
        for _ in range(pop_size // len(groups)):
            population.append(random_expr(group_terms, cfg, rng, group=group))
    while len(population) < pop_size:
        population.append(random_expr(terminals, cfg, rng))
    return population[:pop_size]


def load_discovery_frame(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(cfg["paths"]["feature_panel"])
    panel["trade_date"] = panel["trade_date"].astype(str)
    universe_types = set(cfg["universe"]["main"])
    panel = panel[panel["type"].isin(universe_types)].copy()
    terminals = []
    missing = []
    for group_terms in cfg["terminal_groups"].values():
        for term in group_terms:
            if term in panel.columns:
                terminals.append(term)
            else:
                missing.append(term)
    terminals = list(dict.fromkeys(terminals))
    if missing:
        print(f"[warn] missing terminals ignored: {missing}")
    needed = ["ts_code", "trade_date", "name", "type", cfg["target"]["return_column"], cfg["target"]["rank_column"]]
    keep = list(dict.fromkeys(needed + terminals))
    panel = panel[keep].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel, terminals


def run_ga(frame: pd.DataFrame, terminals: list[str], cfg: dict, out_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    rng = random.Random(int(cfg["seed"]))
    discovery_mask = (
        (frame["trade_date"] >= cfg["split"]["discovery_start"])
        & (frame["trade_date"] <= cfg["split"]["discovery_end"])
        & frame[cfg["target"]["return_column"]].notna()
    )
    ev = Evaluator(frame)
    scorer = AlphaScorer(frame, cfg, discovery_mask)
    population = build_population(terminals, cfg, rng)
    cache: dict[str, Candidate] = {}
    hall: dict[str, Candidate] = {}
    g = cfg["ga"]
    pop_size = int(g["population_size"])
    elite_size = int(g["elite_size"])
    immigrant_count = max(1, int(pop_size * float(g["random_immigrant_rate"])))
    main_count = pop_size - immigrant_count
    start = time.time()
    for gen in range(int(g["generations"])):
        scored = []
        for expr in population:
            key = canonical(expr)
            if key not in cache:
                value = ev.eval(expr)
                raw, detail, _daily = scorer.score(value)
                fit = (
                    raw
                    - float(g["daily_alpha_volatility_penalty"]) * (detail.get("daily_alpha_std") or 0.0)
                    - float(g["complexity_penalty"]) * nodes(expr)
                )
                cache[key] = Candidate(
                    fitness=float(fit),
                    expr=expr,
                    direction=int(detail["direction"]),
                    peak_month=str(detail["peak_month"]),
                    raw_alpha=float(detail.get("raw_alpha", np.nan)),
                    top3_month_alpha=float(detail.get("top3_month_alpha", np.nan)),
                    contiguous_alpha=float(detail.get("best_contiguous_2m_alpha", np.nan)),
                    daily_alpha_std=float(detail.get("daily_alpha_std", np.nan)),
                )
            cand = cache[key]
            scored.append(cand)
            hall[key] = cand
        scored.sort(key=lambda x: x.fitness, reverse=True)
        print(
            f"[ga-sector] gen={gen + 1:02d}/{g['generations']} "
            f"best={scored[0].fitness:.5f} raw={scored[0].raw_alpha:.5f} "
            f"unique={len(cache)} elapsed={(time.time() - start) / 60:.1f}m"
        )
        elites = scored[:elite_size]
        next_pop = [copy.deepcopy(x.expr) for x in elites]

        def parent() -> object:
            sample = rng.sample(scored, min(int(g["tournament_size"]), len(scored)))
            return max(sample, key=lambda x: x.fitness).expr

        while len(next_pop) < main_count:
            child = copy.deepcopy(parent())
            if rng.random() < float(g["crossover_rate"]):
                child = crossover(child, parent(), rng)
            if rng.random() < float(g["mutation_rate"]):
                child = mutate(child, terminals, cfg, rng)
            if depth(child) > int(g["max_depth"]) or not valid_expr(child):
                child = random_expr(terminals, cfg, rng)
            next_pop.append(child)
        while len(next_pop) < pop_size:
            next_pop.append(random_expr(terminals, cfg, rng))
        population = next_pop
    candidates = sorted(hall.values(), key=lambda x: x.fitness, reverse=True)[: int(g["candidate_keep"])]
    selected, candidate_rows = select_library(candidates, frame, ev, scorer, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(out_dir / "search_candidates.csv", index=False, encoding="utf-8-sig")
    payload = {
        "seed": cfg["seed"],
        "target": cfg["target"],
        "ga": cfg["ga"],
        "universe": cfg["universe"]["main"],
        "factors": selected,
        "unique_expressions": len(cache),
        "seconds": time.time() - start,
    }
    (out_dir / "sector_factor_library.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(selected).to_csv(out_dir / "sector_factor_library.csv", index=False, encoding="utf-8-sig")
    return selected, pd.DataFrame(candidate_rows)


def rank_for_corr(value: pd.Series, scorer: AlphaScorer, direction: int) -> pd.Series:
    s = pd.Series(direction * value.loc[scorer.frame.index], index=scorer.frame.index)
    return s.groupby(scorer.frame["trade_date"]).rank(pct=True).astype("float32")


def select_library(candidates: list[Candidate], frame: pd.DataFrame, ev: Evaluator, scorer: AlphaScorer, cfg: dict):
    selected = []
    ranked_series: list[pd.Series] = []
    peak_counts: dict[str, int] = {}
    rows = []
    g = cfg["ga"]
    for cand in candidates:
        value = ev.eval(cand.expr)
        rank_series = rank_for_corr(value, scorer, cand.direction)
        duplicate_corr = 0.0
        duplicate = False
        for existing in ranked_series:
            corr = float(rank_series.corr(existing))
            duplicate_corr = max(duplicate_corr, abs(corr) if np.isfinite(corr) else 0.0)
            if np.isfinite(corr) and abs(corr) >= float(g["corr_prune_threshold"]):
                duplicate = True
                break
        row = {
            "expression_text": expr_text(cand.expr),
            "fitness": cand.fitness,
            "direction": cand.direction,
            "peak_month": cand.peak_month,
            "nodes": nodes(cand.expr),
            "depth": depth(cand.expr),
            "category": expression_category(cand.expr, cfg),
            "duplicate_corr": duplicate_corr,
            "selected": False,
        }
        if peak_counts.get(cand.peak_month, 0) >= int(g["max_factors_per_peak_month"]):
            row["reject_reason"] = "peak_month_limit"
            rows.append(row)
            continue
        if duplicate:
            row["reject_reason"] = "correlation_duplicate"
            rows.append(row)
            continue
        if rank_series.notna().sum() < 100 or float(rank_series.std(skipna=True)) < 1e-8:
            row["reject_reason"] = "invalid_rank"
            rows.append(row)
            continue
        name = f"sector_factor_{len(selected) + 1:02d}"
        item = {
            "name": name,
            "expression": cand.expr,
            "expression_text": expr_text(cand.expr),
            "direction": cand.direction,
            "category": expression_category(cand.expr, cfg),
            "fitness": cand.fitness,
            "raw_alpha": cand.raw_alpha,
            "top3_month_alpha": cand.top3_month_alpha,
            "best_contiguous_2m_alpha": cand.contiguous_alpha,
            "daily_alpha_std": cand.daily_alpha_std,
            "peak_month": cand.peak_month,
            "nodes": nodes(cand.expr),
            "depth": depth(cand.expr),
        }
        selected.append(item)
        ranked_series.append(rank_series)
        peak_counts[cand.peak_month] = peak_counts.get(cand.peak_month, 0) + 1
        row["selected"] = True
        row["factor_name"] = name
        row["reject_reason"] = ""
        rows.append(row)
        if len(selected) >= int(g["library_size"]):
            break
    return selected, rows


def evaluate_library(frame: pd.DataFrame, library: list[dict], cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = Evaluator(frame)
    periods = {
        "discovery": (cfg["split"]["discovery_start"], cfg["split"]["discovery_end"]),
        "validation": (cfg["split"]["validation_start"], cfg["split"]["validation_end"]),
        "test_observation": (cfg["split"]["test_start"], cfg["split"]["test_end"]),
    }
    all_rows = []
    monthly_rows = []
    for item in library:
        value = ev.eval(item["expression"])
        for period, (start, end) in periods.items():
            mask = (
                (frame["trade_date"] >= start)
                & (frame["trade_date"] <= end)
                & frame[cfg["target"]["return_column"]].notna()
            )
            scorer = AlphaScorer(frame, cfg, mask)
            row = {"factor_name": item["name"], "period": period}
            for k in cfg["target"]["diagnostic_top_k"]:
                row.update(scorer.diagnostics(value, int(item["direction"]), int(k)))
            daily = scorer._daily_for_direction(value.loc[scorer.frame.index].to_numpy(dtype=float), int(item["direction"]), int(cfg["target"]["primary_top_k"]))
            daily["month"] = daily["trade_date"].str[:6]
            monthly = daily.groupby("month").agg(alpha=("alpha", "mean"), top_return=("top_return", "mean"), ndcg=("ndcg", "mean"), rank_ic=("rank_ic", "mean"))
            row["rank_ic"] = float(daily["rank_ic"].mean(skipna=True))
            row["ndcg"] = float(daily["ndcg"].mean(skipna=True))
            row["positive_month_ratio"] = float((monthly["alpha"] > 0).mean()) if len(monthly) else np.nan
            row["best_month"] = str(monthly["alpha"].idxmax()) if len(monthly) else ""
            row["worst_month"] = str(monthly["alpha"].idxmin()) if len(monthly) else ""
            row["best_month_alpha"] = float(monthly["alpha"].max()) if len(monthly) else np.nan
            row["worst_month_alpha"] = float(monthly["alpha"].min()) if len(monthly) else np.nan
            all_rows.append(row)
            for month, m in monthly.iterrows():
                monthly_rows.append({"factor_name": item["name"], "period": period, "month": month, **m.to_dict()})
    metrics = pd.DataFrame(all_rows)
    monthly = pd.DataFrame(monthly_rows)
    wide = metrics.pivot(index="factor_name", columns="period")
    flat_rows = []
    for item in library:
        name = item["name"]
        out = {k: v for k, v in item.items() if k not in {"expression"}}
        out["factor_name"] = name
        out["expression_json"] = json.dumps(item["expression"], ensure_ascii=False)
        for period in periods:
            sub = metrics[(metrics["factor_name"] == name) & (metrics["period"] == period)]
            if sub.empty:
                continue
            for col, val in sub.iloc[0].items():
                if col not in {"factor_name", "period"}:
                    out[f"{period}_{col}"] = val
        valid_alpha = out.get("validation_top10_alpha", np.nan)
        valid_pos = out.get("validation_positive_month_ratio", np.nan)
        valid_ndcg = out.get("validation_top10_ndcg", np.nan)
        out["status"] = (
            "core"
            if pd.notna(valid_alpha)
            and valid_alpha > float(cfg["selection"]["core_validation_top10_alpha_min"])
            and valid_pos >= float(cfg["selection"]["core_validation_positive_month_ratio_min"])
            and pd.notna(valid_ndcg)
            else "diagnostic"
        )
        flat_rows.append(out)
    flat = pd.DataFrame(flat_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_dir / "factor_full_validation.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(out_dir / "factor_monthly_alpha.csv", index=False, encoding="utf-8-sig")
    return flat, monthly


def compute_correlation_matrix(frame: pd.DataFrame, library: list[dict], cfg: dict, out_dir: Path) -> pd.DataFrame:
    mask = (
        (frame["trade_date"] >= cfg["split"]["discovery_start"])
        & (frame["trade_date"] <= cfg["split"]["discovery_end"])
        & frame[cfg["target"]["return_column"]].notna()
    )
    scorer = AlphaScorer(frame, cfg, mask)
    ev = Evaluator(frame)
    ranks = {}
    for item in library:
        value = ev.eval(item["expression"])
        ranks[item["name"]] = rank_for_corr(value, scorer, int(item["direction"]))
    corr = pd.DataFrame(ranks).corr()
    corr.to_csv(out_dir / "factor_correlation_matrix.csv", encoding="utf-8-sig")
    return corr


def plot_report_figures(validation: pd.DataFrame, monthly: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    if "factor_name" not in validation.columns and "name" in validation.columns:
        validation = validation.copy()
        validation["factor_name"] = validation["name"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    val = validation.sort_values("validation_top10_alpha", ascending=False).head(15)
    fig1 = out_dir / "validation_top10_alpha.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(val["factor_name"], val["validation_top10_alpha"] * 100, color="#1f5fbf")
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_title("验证期 Top10 Alpha 前15因子")
    ax.set_ylabel("平均未来10日超额收益 (%)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(fig1, bbox_inches="tight")
    plt.close(fig)

    core_names = validation.sort_values("validation_top10_alpha", ascending=False).head(8)["factor_name"]
    heat = monthly[(monthly["period"] == "validation") & (monthly["factor_name"].isin(core_names))]
    pivot = heat.pivot(index="factor_name", columns="month", values="alpha").reindex(core_names)
    fig2 = out_dir / "validation_monthly_alpha_heatmap.png"
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(pivot.fillna(0).to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=60, ha="right")
    ax.set_title("验证期前8因子月度Top10 Alpha")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("未来10日超额收益 (%)")
    fig.tight_layout()
    fig.savefig(fig2, bbox_inches="tight")
    plt.close(fig)
    return fig1, fig2


def build_report(validation: pd.DataFrame, monthly: pd.DataFrame, cfg: dict, out_dir: Path) -> tuple[Path, Path]:
    if "factor_name" not in validation.columns and "name" in validation.columns:
        validation = validation.copy()
        validation["factor_name"] = validation["name"]
    fig1, fig2 = plot_report_figures(validation, monthly, out_dir)
    top = validation.sort_values(["status", "validation_top10_alpha"], ascending=[True, False]).head(30)
    summary_rows = []
    for row in top.itertuples(index=False):
        summary_rows.append(
            [
                row.factor_name,
                row.status,
                row.category,
                pct(row.discovery_top10_alpha),
                pct(row.validation_top10_alpha),
                pct(row.validation_positive_month_ratio),
                num(row.validation_rank_ic, 4),
                row.peak_month,
                row.expression_text,
            ]
        )
    factor_sections = []
    for row in top.head(12).itertuples(index=False):
        factor_sections.append(
            f"""### {row.factor_name}

```text
{row.expression_text}
```

- 方向：`{int(row.direction)}`
- 类别：`{row.category}`
- 验证期 Top10 alpha：{pct(row.validation_top10_alpha)}
- 验证期 Top5 alpha：{pct(row.validation_top5_alpha)}
- 验证期 Top20 alpha：{pct(row.validation_top20_alpha)}
- 验证期正 alpha 月份比例：{pct(row.validation_positive_month_ratio)}
- 验证期 Rank IC：{num(row.validation_rank_ic, 4)}
- 发现期峰值月份：`{row.peak_month}`
- 状态：`{row.status}`
"""
        )
    core_count = int((validation["status"] == "core").sum())
    md = f"""# 板块高 Alpha 因子挖掘报告

生成日期：2026-07-01

## 目的

本轮只做板块因子挖掘，不训练滚动模型。目标是找到一批能够提示板块未来 10 个交易日进入主升浪的高 alpha 因子。评价标准优先看 Top10 板块未来 10 日相对全板块均值的超额收益，而不是单纯 IC。

## 数据与目标

- 宇宙：同花顺行业 + 概念板块，类型为 `{', '.join(cfg['universe']['main'])}`。
- 发现期：{cfg['split']['discovery_start']} 至 {cfg['split']['discovery_end']}。
- 验证期：{cfg['split']['validation_start']} 至 {cfg['split']['validation_end']}。
- 观察期：{cfg['split']['test_start']} 至 {cfg['split']['test_end']}，不参与筛选。
- 主标签：`future_ret_10d`。
- 主目标：Top10 未来 10 日 alpha。

## 遗传搜索设置

```text
population_size = {cfg['ga']['population_size']}
generations = {cfg['ga']['generations']}
elite_size = {cfg['ga']['elite_size']}
tournament_size = {cfg['ga']['tournament_size']}
crossover_rate = {cfg['ga']['crossover_rate']}
mutation_rate = {cfg['ga']['mutation_rate']}
max_depth = {cfg['ga']['max_depth']}
library_size = {cfg['ga']['library_size']}
```

fitness 使用 robust 版：

```text
1.50 * mean(monthly_top10_alpha)
+ 0.50 * min(best_contiguous_2_month_alpha, 3%)
+ 0.50 * min(mean(top3 monthly_top10_alpha), 3%)
+ 0.02 * (positive_month_ratio - 50%)
- 0.10 * std(daily_top10_alpha)
- 0.0005 * expression_nodes
```

## 因子库概览

- 最终因子数：{len(validation)}
- core 因子数：{core_count}
- diagnostic 因子数：{len(validation) - core_count}

{md_table(['因子', '状态', '类别', '发现期Top10 alpha', '验证期Top10 alpha', '验证期正月份', '验证期Rank IC', '峰值月', '公式'], summary_rows)}

![验证期Top10 Alpha]({fig1.name})

![验证期月度Alpha热力图]({fig2.name})

## 代表性因子说明

{chr(10).join(factor_sections)}

## 结论

本轮因子挖掘把未来 10 日 Top10 alpha 作为主目标，更贴近“板块主升浪入场信号”。`core` 因子表示发现期和验证期都有正向 Top10 alpha，`diagnostic` 因子表示发现期强但验证期不完全达标，后续可以作为模型输入或人工观察项。

下一步建议先不要急着扩大模型复杂度，而是用这些因子做两件事：

1. 看 Top10 alpha 在 2024、2025、2026 各月份是否集中在少数行情阶段。
2. 用 core 因子构造简单投票或 LightGBM 滚动模型，比较是否优于之前直接使用原始板块特征。
"""
    md_path = out_dir / "FACTOR_MINING_REPORT.md"
    html_path = out_dir / "FACTOR_MINING_REPORT.html"
    pdf_path = out_dir / "FACTOR_MINING_REPORT.pdf"
    md_path.write_text(md, encoding="utf-8")
    html = markdown_to_html(md)
    html_path.write_text(html, encoding="utf-8")
    export_pdf(html_path, pdf_path)
    return md_path, pdf_path


def markdown_to_html(md: str) -> str:
    if markdown is None:
        body = "<pre>" + md + "</pre>"
    else:
        body = markdown.markdown(md, extensions=["tables", "sane_lists", "fenced_code"])
    css = """
    @page { size: A4; margin: 18mm 16mm; }
    body { font-family: "Microsoft YaHei", "SimSun", Arial, sans-serif; color: #20242a; line-height: 1.6; font-size: 12.5px; }
    h1 { color: #17365d; border-bottom: 2px solid #17365d; padding-bottom: 8px; }
    h2 { color: #17365d; border-left: 5px solid #5b8cc0; padding-left: 10px; margin-top: 24px; }
    h3 { color: #2f4f6f; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 9.5px; }
    th { background: #eaf1f8; color: #17365d; }
    th, td { border: 1px solid #cfd8e3; padding: 4px 5px; vertical-align: middle; }
    tr:nth-child(even) td { background: #fbfcfe; }
    code, pre { background: #f4f6f8; border: 1px solid #e5e8ec; border-radius: 3px; }
    code { padding: 1px 4px; }
    pre { padding: 8px 10px; white-space: pre-wrap; }
    img { display: block; max-width: 94%; margin: 12px auto 18px; }
    """
    return f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = find_browser()
    if not browser:
        print("[warn] no browser found, skip pdf export")
        return
    if pdf_path.exists():
        pdf_path.unlink()
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sector/factor_mining.json")
    parser.add_argument("--report-only", action="store_true", help="只基于已有验证CSV重新生成报告")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["artifacts"])
    report_dir = Path(cfg["paths"]["reports"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report_only:
        validation = pd.read_csv(out_dir / "factor_full_validation.csv")
        if "factor_name" not in validation.columns and "name" in validation.columns:
            validation.insert(1, "factor_name", validation["name"])
            validation.to_csv(out_dir / "factor_full_validation.csv", index=False, encoding="utf-8-sig")
        monthly = pd.read_csv(out_dir / "factor_monthly_alpha.csv")
        md_path, pdf_path = build_report(validation, monthly, cfg, report_dir)
        print(f"[done] {md_path}")
        print(f"[done] {pdf_path}")
        return
    print("[load] feature panel")
    frame, terminals = load_discovery_frame(cfg)
    print(f"[load] rows={len(frame):,} sectors={frame.ts_code.nunique():,} dates={frame.trade_date.nunique():,}")
    print(f"[load] terminals={len(terminals)}")
    library, _candidates = run_ga(frame, terminals, cfg, out_dir)
    print(f"[validate] selected factors={len(library)}")
    validation, monthly = evaluate_library(frame, library, cfg, out_dir)
    compute_correlation_matrix(frame, library, cfg, out_dir)
    md_path, pdf_path = build_report(validation, monthly, cfg, report_dir)
    print(f"[done] {out_dir / 'sector_factor_library.csv'}")
    print(f"[done] {out_dir / 'factor_full_validation.csv'}")
    print(f"[done] {md_path}")
    print(f"[done] {pdf_path}")


if __name__ == "__main__":
    main()
