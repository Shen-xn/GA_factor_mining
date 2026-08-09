from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_experiments import (
    OUT_DIR,
    REPORT_DIR,
    TEST_END,
    UNIVERSES,
    StrategyConfig,
    add_formula_scores,
    annualized_metrics,
    backtest_one_cached,
    load_or_build_features,
    prepare_universe_cache,
)


ROLLING_RESULTS = OUT_DIR / "ROLLING_VALIDATION_RESULTS.csv"
ROLLING_CURVES = OUT_DIR / "ROLLING_VALIDATION_EQUITY_CURVES.csv"
ROLLING_REPORT = REPORT_DIR / "ROLLING_VALIDATION_REPORT.md"
ROLLING_HTML = REPORT_DIR / "ROLLING_VALIDATION_REPORT.html"
ROLLING_PDF = REPORT_DIR / "ROLLING_VALIDATION_REPORT.pdf"
ROLLING_ANNUAL_FIG = REPORT_DIR / "rolling_validation_annual_returns.png"
ROLLING_EQUITY_FIG = REPORT_DIR / "rolling_validation_equity_curves.png"


FEATURE_COLS = [
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


ML_STRATEGIES = [
    {
        "strategy_id": "rolling_lgbm5_top5",
        "display_name": "滚动 LGBM5 Top5",
        "horizon": 5,
        "top_k": 5,
        "rebalance_days": 1,
        "hold_days": 1,
    },
    {
        "strategy_id": "rolling_lgbm10_top10",
        "display_name": "滚动 LGBM10 Top10",
        "horizon": 10,
        "top_k": 10,
        "rebalance_days": 1,
        "hold_days": 1,
    },
]


FORMULA_STRATEGIES = [
    {
        "strategy_id": "formula_industry_breakout_top3",
        "display_name": "公式 行业突破 Top3",
        "config": StrategyConfig(
            "formula_industry_breakout_top3",
            "industry",
            "score_breakout",
            3,
            1,
            1,
        ),
    },
    {
        "strategy_id": "formula_low_vol_top20",
        "display_name": "公式 低波动动量 Top20",
        "config": StrategyConfig(
            "formula_low_vol_top20",
            "industry_concept",
            "score_low_vol_mom",
            20,
            1,
            1,
        ),
    },
]


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def year_bounds(df: pd.DataFrame, start_year: int, end_year: int) -> dict[int, tuple[str, str]]:
    dates = df[["trade_date"]].drop_duplicates().sort_values("trade_date")
    dates["year"] = dates["trade_date"].astype(str).str[:4].astype(int)
    bounds: dict[int, tuple[str, str]] = {}
    for year in range(start_year, end_year + 1):
        sub = dates[dates["year"].eq(year)]
        if not sub.empty:
            bounds[year] = (str(sub["trade_date"].min()), str(sub["trade_date"].max()))
    if 2026 in bounds:
        start, _ = bounds[2026]
        bounds[2026] = (start, TEST_END)
    return bounds


def fit_predict_lgbm(
    df: pd.DataFrame,
    horizon: int,
    fold_years: list[int],
) -> pd.DataFrame:
    import lightgbm as lgb

    target = f"future_ret_{horizon}d_rank"
    score_col = f"score_rolling_lgbm_{horizon}d"
    sub = df[df["type"].isin(UNIVERSES["industry_concept"])].copy()
    sub[score_col] = np.nan
    for year in fold_years:
        train_end = f"{year - 1}1231"
        pred_start = f"{year}0101"
        pred_end = TEST_END if year == 2026 else f"{year}1231"
        train_mask = (sub["trade_date"] <= train_end) & sub[target].notna()
        pred_mask = (sub["trade_date"] >= pred_start) & (sub["trade_date"] <= pred_end)
        train = (
            sub.loc[train_mask, FEATURE_COLS + [target]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        pred = sub.loc[pred_mask, FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        if train.empty or pred.empty:
            continue
        print(
            f"[rolling-lgbm] horizon={horizon} validate={year} "
            f"train_end={train_end} train_rows={len(train):,} pred_rows={len(pred):,}"
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
            random_state=42 + year + horizon,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(train[FEATURE_COLS], train[target])
        sub.loc[pred_mask, score_col] = model.predict(pred).astype("float32")
    return sub[["ts_code", "trade_date", score_col]]


def evaluate_year(
    cache: dict,
    cfg: StrategyConfig,
    year: int,
    start: str,
    end: str,
) -> tuple[dict, pd.Series]:
    ret, _pos, aux = backtest_one_cached(cache, cfg, start, end)
    bench = cache["benchmark"].loc[(cache["benchmark"].index >= start) & (cache["benchmark"].index <= end)]
    bench = bench.reindex(ret.index).fillna(0.0)
    excess = ret - bench
    metrics = annualized_metrics(ret)
    bench_metrics = annualized_metrics(bench)
    excess_metrics = annualized_metrics(excess)
    row = {
        "year": year,
        "start": start,
        "end": end,
        **metrics,
        "bench_ann_ret": bench_metrics["ann_ret"],
        "excess_ann_ret": excess_metrics["ann_ret"],
        "excess_sharpe": excess_metrics["sharpe"],
        "avg_turnover": aux["avg_turnover"],
        "rebalance_count": aux["rebalance_count"],
    }
    return row, ret


def aggregate_metrics(ret: pd.Series) -> dict[str, float]:
    return annualized_metrics(ret.sort_index())


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def make_figures(results: pd.DataFrame, curves: pd.DataFrame) -> None:
    setup_matplotlib()
    annual = results[results["row_type"].eq("year")].copy()
    selected = [
        "rolling_lgbm5_top5",
        "rolling_lgbm10_top10",
        "formula_industry_breakout_top3",
        "formula_low_vol_top20",
    ]
    annual = annual[annual["strategy_id"].isin(selected)]
    pivot = annual.pivot(index="year", columns="strategy_id", values="ann_ret").sort_index()
    labels = {
        "rolling_lgbm5_top5": "滚动 LGBM5 Top5",
        "rolling_lgbm10_top10": "滚动 LGBM10 Top10",
        "formula_industry_breakout_top3": "公式突破 Top3",
        "formula_low_vol_top20": "公式低波动 Top20",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for col in selected:
        if col in pivot:
            ax.plot(pivot.index, pivot[col] * 100, marker="o", lw=1.9, label=labels[col])
    ax.axhline(0, color="#555", lw=0.9)
    ax.set_title("滚动验证：年度年化收益")
    ax.set_ylabel("年化收益率 (%)")
    ax.set_xlabel("验证年份")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ROLLING_ANNUAL_FIG, bbox_inches="tight")
    plt.close(fig)

    curve_df = curves.copy()
    curve_df["date"] = pd.to_datetime(curve_df["date"].astype(str), format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for sid in selected:
        sub = curve_df[curve_df["strategy_id"].eq(sid)].sort_values("date")
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["equity"], lw=1.9, label=labels[sid])
    ax.axhline(1.0, color="#555", lw=0.9, ls="--")
    ax.set_title("滚动验证：拼接样本外净值曲线")
    ax.set_ylabel("净值，起点=1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ROLLING_EQUITY_FIG, bbox_inches="tight")
    plt.close(fig)


def build_report(results: pd.DataFrame) -> str:
    summary = results[results["row_type"].eq("summary")].copy()
    yearly = results[results["row_type"].eq("year")].copy()
    summary_rows = []
    name_map = {
        "rolling_lgbm5_top5": "滚动 LGBM5 Top5",
        "rolling_lgbm10_top10": "滚动 LGBM10 Top10",
        "formula_industry_breakout_top3": "公式 行业突破 Top3",
        "formula_low_vol_top20": "公式 低波动动量 Top20",
    }
    for row in summary.itertuples(index=False):
        summary_rows.append(
            [
                name_map.get(row.strategy_id, row.strategy_id),
                str(int(row.start_year)),
                str(int(row.end_year)),
                pct(row.ann_ret),
                num(row.sharpe, 2),
                pct(row.max_drawdown),
                pct(row.win_rate),
                pct(row.avg_turnover),
                f"{int(row.positive_years)}/{int(row.valid_years)}",
            ]
        )

    lgbm5_rows = []
    for row in yearly[yearly["strategy_id"].eq("rolling_lgbm5_top5")].itertuples(index=False):
        lgbm5_rows.append(
            [
                str(int(row.year)),
                str(int(row.days)),
                pct(row.total_ret),
                pct(row.ann_ret),
                num(row.sharpe, 2),
                pct(row.max_drawdown),
                pct(row.bench_ann_ret),
                pct(row.excess_ann_ret),
            ]
        )

    formula_rows = []
    for row in yearly[yearly["strategy_id"].eq("formula_industry_breakout_top3")].itertuples(index=False):
        formula_rows.append(
            [
                str(int(row.year)),
                pct(row.ann_ret),
                num(row.sharpe, 2),
                pct(row.max_drawdown),
                pct(row.excess_ann_ret),
            ]
        )

    return f"""# 板块策略滚动验证补充报告

生成日期：2026-07-01

## 目的

本补充验证用于解决固定训练/验证/测试切分可能带来的偶然性问题。对机器学习策略，采用逐年 walk-forward：每一年只使用上一年年底以前已经兑现标签的数据训练模型，然后验证下一年；对公式评分策略，由于没有训练过程，直接展示 2016 年以来每一年的样本外表现。

## 滚动验证口径

- 机器学习模型：`industry_concept` 宇宙，LightGBM 回归，输入仍为板块历史状态的横截面 rank 特征。
- 训练方式：扩展窗口训练。例如验证 2024 年时，只使用 2023-12-31 及以前数据；验证 2026 年时，只使用 2025-12-31 及以前数据。
- 验证年份：机器学习从 2018 年滚动至 2026 年；2026 年只统计至 2026-05-29。
- 公式策略：不训练，直接从 2016 年统计至 2026 年。
- 收益口径：信号日收盘后选板块，下一交易日开始持有，等权组合，暂不扣手续费和滑点。

## 总体结果

{md_table(['策略', '起始年', '结束年', '拼接年化', '拼接夏普', '最大回撤', '日胜率', '平均换手', '正收益年份'], summary_rows)}

![滚动年度收益]({ROLLING_ANNUAL_FIG.name})

![滚动拼接净值]({ROLLING_EQUITY_FIG.name})

## 推荐机器学习模型逐年表现

{md_table(['年份', '交易日', '总收益', '年化', '夏普', '最大回撤', '基准年化', '超额年化'], lgbm5_rows)}

## 代表性公式策略逐年表现

{md_table(['年份', '年化', '夏普', '最大回撤', '超额年化'], formula_rows)}

## 结论

滚动验证比固定切分更接近真实使用方式。若滚动 LGBM5 Top5 在多数年份保持正收益，同时拼接样本外净值曲线没有只依赖单一年份贡献，则说明板块强度模型不是完全来自一次性参数选择。公式策略的逐年展示则用于判断简单可解释信号是否具有跨年份稳定性。

需要注意，滚动验证仍然没有扣交易成本，且板块基金仍是假设资产。下一步应把该滚动框架作为默认评估入口，并加入成本压力测试和真实 ETF 映射。
"""


def markdown_to_html(md: str) -> str:
    import markdown

    css = """
    @page { size: A4; margin: 18mm 16mm; }
    body { font-family: "Microsoft YaHei", "SimSun", Arial, sans-serif; color: #20242a; line-height: 1.62; font-size: 13px; }
    h1 { color: #17365d; font-size: 27px; border-bottom: 2px solid #17365d; padding-bottom: 9px; }
    h2 { color: #17365d; border-left: 5px solid #5b8cc0; padding-left: 10px; margin-top: 26px; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 11px; }
    th { background: #eaf1f8; color: #17365d; }
    th, td { border: 1px solid #cfd8e3; padding: 5px 6px; vertical-align: middle; }
    tr:nth-child(even) td { background: #fbfcfe; }
    code { background: #f4f6f8; border: 1px solid #e5e8ec; padding: 1px 4px; border-radius: 3px; }
    img { display: block; max-width: 94%; margin: 12px auto 18px; }
    """
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
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


def export_pdf() -> None:
    browser = find_browser()
    if browser is None:
        return
    if ROLLING_PDF.exists():
        ROLLING_PDF.unlink()
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            f"--print-to-pdf={ROLLING_PDF}",
            ROLLING_HTML.as_uri(),
        ],
        check=True,
        cwd=str(REPORT_DIR),
    )


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_features(force=False)
    df = add_formula_scores(df)
    bounds = year_bounds(df, 2016, 2026)

    fold_years = [year for year in range(2018, 2027) if year in bounds]
    score_frames = []
    for spec in ML_STRATEGIES:
        score_frames.append(fit_predict_lgbm(df, spec["horizon"], fold_years))
    for frame in score_frames:
        df = df.merge(frame, on=["ts_code", "trade_date"], how="left")

    rows: list[dict] = []
    curve_rows: list[dict] = []
    score_names = [f"score_rolling_lgbm_{spec['horizon']}d" for spec in ML_STRATEGIES]
    score_names += [spec["config"].score_name for spec in FORMULA_STRATEGIES]
    caches = {
        "industry_concept": prepare_universe_cache(df, "industry_concept", score_names),
        "industry": prepare_universe_cache(df, "industry", score_names),
    }

    eval_specs = []
    for spec in ML_STRATEGIES:
        cfg = StrategyConfig(
            spec["strategy_id"],
            "industry_concept",
            f"score_rolling_lgbm_{spec['horizon']}d",
            spec["top_k"],
            spec["rebalance_days"],
            spec["hold_days"],
        )
        eval_specs.append((spec["strategy_id"], spec["display_name"], "ml_walk_forward", cfg, fold_years))
    for spec in FORMULA_STRATEGIES:
        formula_years = [year for year in range(2016, 2027) if year in bounds]
        eval_specs.append((spec["strategy_id"], spec["display_name"], "formula_all_years", spec["config"], formula_years))

    for strategy_id, display_name, validation_kind, cfg, years in eval_specs:
        cache = caches[cfg.universe]
        combined = []
        positive_years = 0
        for year in years:
            start, end = bounds[year]
            row, ret = evaluate_year(cache, cfg, year, start, end)
            row.update(
                {
                    "row_type": "year",
                    "strategy_id": strategy_id,
                    "display_name": display_name,
                    "validation_kind": validation_kind,
                    "universe": cfg.universe,
                    "score_name": cfg.score_name,
                    "top_k": cfg.top_k,
                    "rebalance_days": cfg.rebalance_days,
                    "train_end": "" if validation_kind.startswith("formula") else f"{year - 1}1231",
                }
            )
            rows.append(row)
            if row["ann_ret"] > 0:
                positive_years += 1
            if not ret.empty:
                ret = ret.copy()
                combined.append(ret)
        if combined:
            combined_ret = pd.concat(combined).sort_index()
            metrics = aggregate_metrics(combined_ret)
            curve = (1.0 + combined_ret.fillna(0.0)).cumprod()
            for date, equity in curve.items():
                curve_rows.append({"strategy_id": strategy_id, "date": date, "equity": float(equity)})
        else:
            metrics = annualized_metrics(pd.Series(dtype=float))
        rows.append(
            {
                "row_type": "summary",
                "strategy_id": strategy_id,
                "display_name": display_name,
                "validation_kind": validation_kind,
                "universe": cfg.universe,
                "score_name": cfg.score_name,
                "top_k": cfg.top_k,
                "rebalance_days": cfg.rebalance_days,
                "start_year": min(years),
                "end_year": max(years),
                "valid_years": len(years),
                "positive_years": positive_years,
                **metrics,
                "avg_turnover": float(np.nanmean([r["avg_turnover"] for r in rows if r.get("strategy_id") == strategy_id and r.get("row_type") == "year"])),
            }
        )

    results = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    results.to_csv(ROLLING_RESULTS, index=False, encoding="utf-8-sig")
    curves.to_csv(ROLLING_CURVES, index=False, encoding="utf-8-sig")
    make_figures(results, curves)
    report = build_report(results)
    ROLLING_REPORT.write_text(report, encoding="utf-8")
    ROLLING_HTML.write_text(markdown_to_html(report), encoding="utf-8")
    export_pdf()
    print(f"[done] results={ROLLING_RESULTS}")
    print(f"[done] report={ROLLING_REPORT}")
    print(f"[done] elapsed={(time.time() - start_time) / 60:.1f}m")


if __name__ == "__main__":
    main()
