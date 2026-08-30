from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

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



UNARY = ("neg", "abs", "signed_log", "signed_sqrt")
BINARY = ("add", "sub", "mul", "div", "min", "max")
TEMPORAL = ("delta_5", "slope_5", "mean_5", "std_20", "zscore_20", "ts_rank_20")


def expr_text(expr) -> str:
    return expression_text(expr, separator=", ")


def simple_derived_duplicate(expr) -> bool:
    """识别人工特征间最简单的线性重写，避免作为独立 GA 因子晋级。"""
    return (
        isinstance(expr, list)
        and len(expr) == 3
        and expr[0] in {"add", "sub"}
        and isinstance(expr[1], str)
        and isinstance(expr[2], str)
    )



def load_config(path: str | Path) -> dict:
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for key in ("feature_panel", "artifacts"):
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

    def __init__(
        self,
        frame: pd.DataFrame,
        max_abs: float = 10.0,
        div_epsilon: float = 0.1,
        temporal_cache_size: int = 32,
    ):
        self.df = frame
        self.max_abs = max_abs
        self.div_epsilon = div_epsilon
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
                sign = np.sign(b).replace(0, 1.0)
                denominator = b.where(b.abs() >= self.div_epsilon, sign * self.div_epsilon)
                out = a / denominator
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
    if cfg["ga"].get("terminal_transform") == "centered_rank":
        non_rank = [terminal for terminal in terminals if not terminal.endswith("_rank")]
        if non_rank:
            raise ValueError(f"centered_rank 模式不允许非排名终端: {non_rank}")
        for terminal in terminals:
            panel[terminal] = (2.0 * panel[terminal] - 1.0).astype("float32")
    return panel, terminals


def make_evaluator(frame: pd.DataFrame, cfg: dict) -> Evaluator:
    return Evaluator(
        frame,
        max_abs=float(cfg["ga"].get("value_clip", 10.0)),
        div_epsilon=float(cfg["ga"].get("protected_div_epsilon", 0.1)),
    )


def run_ga(frame: pd.DataFrame, terminals: list[str], cfg: dict, out_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    rng = random.Random(int(cfg["seed"]))
    discovery_mask = (
        (frame["trade_date"] >= cfg["split"]["discovery_start"])
        & (frame["trade_date"] <= cfg["split"]["discovery_end"])
        & frame[cfg["target"]["return_column"]].notna()
    )
    ev = make_evaluator(frame, cfg)
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
    terminal_names = list(
        dict.fromkeys(
            terminal
            for group in cfg["terminal_groups"].values()
            for terminal in group
            if terminal in frame.columns
        )
    )
    sample_size = int(g.get("corr_sample_size", 20_000))
    sample_step = max(1, len(scorer.frame) // sample_size)
    sample_index = scorer.frame.index[::sample_step]
    manual_ranks = {
        terminal: rank_for_corr(frame[terminal], scorer, 1).loc[sample_index]
        for terminal in terminal_names
    }
    for cand in candidates:
        value = ev.eval(cand.expr)
        rank_series = rank_for_corr(value, scorer, cand.direction)
        sampled_rank = rank_series.loc[sample_index]
        manual_duplicate_factor = ""
        manual_duplicate_corr = 0.0
        for terminal, manual_rank in manual_ranks.items():
            corr = float(sampled_rank.corr(manual_rank))
            if np.isfinite(corr) and abs(corr) > manual_duplicate_corr:
                manual_duplicate_corr = abs(corr)
                manual_duplicate_factor = terminal
        duplicate_corr = 0.0
        duplicate = False
        for existing in ranked_series:
            corr = float(sampled_rank.corr(existing.loc[sample_index]))
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
            "manual_duplicate_corr": manual_duplicate_corr,
            "manual_duplicate_factor": manual_duplicate_factor,
            "selected": False,
        }
        if manual_duplicate_corr >= float(g["corr_prune_threshold"]):
            row["reject_reason"] = "manual_factor_duplicate"
            rows.append(row)
            continue
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
    ev = make_evaluator(frame, cfg)
    periods = {
        "discovery": (cfg["split"]["discovery_start"], cfg["split"]["discovery_end"]),
        "validation": (cfg["split"]["validation_start"], cfg["split"]["validation_end"]),
        "observation": (cfg["split"]["observation_start"], cfg["split"]["observation_end"]),
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
        out["structural_duplicate"] = simple_derived_duplicate(item["expression"])
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
            "shadow"
            if pd.notna(valid_alpha)
            and valid_alpha > float(cfg["selection"]["shadow_validation_top10_alpha_min"])
            and valid_pos >= float(cfg["selection"]["shadow_validation_positive_month_ratio_min"])
            and pd.notna(valid_ndcg)
            and not out["structural_duplicate"]
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
    ev = make_evaluator(frame, cfg)
    ranks = {}
    for item in library:
        value = ev.eval(item["expression"])
        ranks[item["name"]] = rank_for_corr(value, scorer, int(item["direction"]))
    corr = pd.DataFrame(ranks).corr()
    corr.rename_axis("factor_name").to_csv(
        out_dir / "factor_correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    return corr



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sector/factor_mining.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[load] feature panel")
    frame, terminals = load_discovery_frame(cfg)
    print(f"[load] rows={len(frame):,} sectors={frame.ts_code.nunique():,} dates={frame.trade_date.nunique():,}")
    print(f"[load] terminals={len(terminals)}")
    library, _candidates = run_ga(frame, terminals, cfg, out_dir)
    print(f"[validate] selected factors={len(library)}")
    evaluate_library(frame, library, cfg, out_dir)
    compute_correlation_matrix(frame, library, cfg, out_dir)
    print(f"[done] {out_dir / 'sector_factor_library.csv'}")
    print(f"[done] {out_dir / 'factor_full_validation.csv'}")
    print(f"[done] {out_dir / 'factor_monthly_alpha.csv'}")


if __name__ == "__main__":
    main()
