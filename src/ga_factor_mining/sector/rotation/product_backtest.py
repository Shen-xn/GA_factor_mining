#!/usr/bin/env python3
"""把每日板块评分转换为带现金、成本和风险约束的产品策略。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.paths import ensure_output_dir
from .low_risk import DEFAULT_LOW_RISK_CODE, build_low_risk_return_frame, low_risk_data_signature
from .risk import (
    DrawdownState,
    RegimePolicy,
    RegimeState,
    advance_drawdown_state,
    advance_regime,
    build_market_state,
    classify_market,
    leading_sector_strength,
    regime_exposure,
    technical_regime_exposure,
)
from .run_experiments import (
    OBSERVATION_END,
    OBSERVATION_START,
    FEATURE_PROTOCOL_VERSION,
    DATA_DIR,
    OUT_DIR,
    TRAIN_END,
    UNIVERSES,
    VAL_END,
    VAL_START,
    add_formula_scores,
    annualized_metrics,
    current_feature_cache_signature,
    load_feature_subset,
    load_or_build_features,
)
from .strategy import (
    STRATEGY_POLICY_VERSION,
    PositionState,
    StrategyPolicy,
    get_strategy_policy,
    step_portfolio,
    strategy_policy_presets,
)


# 冻结评分从2018年开始；正式产品状态只沿这一条时间路径连续推进。
PRODUCT_HISTORY_START = "20180101"


FORMULA_RANK_COLS = {
    "ret_3d_rank",
    "ret_5d_rank",
    "ret_10d_rank",
    "ret_20d_rank",
    "risk_adj_5_20_rank",
    "risk_adj_10_20_rank",
    "risk_adj_20_60_rank",
    "volatility_20d_rank",
    "close_pos_20d_rank",
    "ma_gap_5_20_rank",
    "range_1d_rank",
    "volume_z_20d_rank",
    "turnover_z_20d_rank",
}
FORMULA_SCORE_NAMES = {
    "score_mom_5_10",
    "score_mom_10_20",
    "score_risk_adj",
    "score_breakout",
    "score_pullback_trend",
    "score_volume_confirm",
    "score_low_vol_mom",
}
PRODUCT_BASE_COLS = {
    "ts_code",
    "trade_date",
    "type",
    "open",
    "close",
    "forward_open_ret_1d",
    "next_open_date",
    "return_end_date",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "volatility_20d",
}


def product_feature_columns(score_name: str, external_score: bool = False) -> set[str]:
    """返回产品账本必需列；外部分数会在读取后单独合并。"""
    columns = PRODUCT_BASE_COLS | {"ret_5d_rank", "volatility_20d_rank"}
    if external_score:
        return columns
    columns |= FORMULA_RANK_COLS
    columns |= {column.removesuffix("_rank") for column in FORMULA_RANK_COLS}
    if score_name not in FORMULA_SCORE_NAMES:
        columns.add(score_name)
    return columns


def prepare_product_panel(
    panel: pd.DataFrame,
    score_name: str,
    universe: str,
) -> pd.DataFrame:
    """只复制产品账本实际使用的列，避免整张特征面板重复扩张。"""
    types = UNIVERSES[universe]
    formula_score = score_name not in panel.columns
    rank_columns = {"ret_5d_rank", "volatility_20d_rank"}
    if formula_score:
        rank_columns |= FORMULA_RANK_COLS
    raw_rank_columns = {column.removesuffix("_rank") for column in rank_columns}
    required = PRODUCT_BASE_COLS | raw_rank_columns | rank_columns
    if not formula_score:
        required.add(score_name)
    missing_base = PRODUCT_BASE_COLS - set(panel.columns)
    if missing_base:
        raise ValueError(f"产品面板缺少字段: {sorted(missing_base)}")
    available = [column for column in panel.columns if column in required]
    sub = panel.loc[panel["type"].isin(types), available].copy()

    # 主宇宙直接复用带协议签名的缓存排名；其他宇宙只重算真正需要的排名。
    reuse_cached_ranks = universe == "industry_concept"
    for rank_column in rank_columns:
        raw_column = rank_column.removesuffix("_rank")
        if reuse_cached_ranks and rank_column in sub.columns:
            continue
        if raw_column not in sub.columns:
            raise ValueError(f"重算 {rank_column} 缺少原始字段 {raw_column}")
        sub[rank_column] = sub.groupby("trade_date", sort=False)[raw_column].transform(
            lambda values: values.rank(pct=True, method="average")
        ).astype("float32")

    if formula_score:
        scored = add_formula_scores(sub)
        if score_name not in scored.columns:
            raise ValueError(f"评分字段不存在: {score_name}")
        sub[score_name] = scored[score_name].astype("float32")
    keep = PRODUCT_BASE_COLS | {score_name, "ret_5d_rank", "volatility_20d_rank"}
    return sub[[column for column in panel.columns if column in keep] + [
        column for column in (score_name, "ret_5d_rank", "volatility_20d_rank")
        if column not in panel.columns and column in sub.columns
    ]].copy()


REGIME_POSITION_LIMITS = {"CASH": 0, "DEFENSIVE": 3, "NEUTRAL": 5, "RISK_ON": 5}

ACTION_LABELS = {"buy": "买入", "sell": "卖出"}
TRADE_TYPE_LABELS = {"entry": "新开", "exit": "清仓", "rebalance": "调仓"}
REASON_LABELS = {
    "data_unavailable_next_valuation": "下一估值日数据不可用",
    "hard_position_loss": "持仓触发硬止损",
    "left_retain_zone": "排名离开保留区",
    "low_risk_residual_allocation": "未配置资金进入低风险资产",
    "market_risk_off": "市场风险状态降仓",
    "position_limit_reduction": "市场状态降低持仓数量",
    "trailing_stop": "触发跟踪止损",
    "vacancy_and_strong_signal": "空缺仓位出现强买入信号",
    "volatility_shock": "高波动与弱动量共振",
    "weight_rebalance": "目标权重调整",
}


def _turnover(pretrade: dict[str, float], target: dict[str, float]) -> float:
    assets = set(pretrade) | set(target)
    asset_change = sum(abs(target.get(code, 0.0) - pretrade.get(code, 0.0)) for code in assets)
    pretrade_cash = 1.0 - sum(pretrade.values())
    target_cash = 1.0 - sum(target.values())
    return 0.5 * (asset_change + abs(target_cash - pretrade_cash))


def _drift_weights(weights: dict[str, float], returns: pd.Series, portfolio_return: float) -> dict[str, float]:
    gross = 1.0 + portfolio_return
    if gross <= 0:
        return {}
    return {
        code: weight * (1.0 + float(returns.get(code, 0.0))) / gross
        for code, weight in weights.items()
    }


def _target_dict(targets: pd.DataFrame) -> dict[str, float]:
    if targets.empty:
        return {}
    return dict(zip(targets["ts_code"], targets["target_weight"], strict=True))


def execution_stability_metrics(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, float]:
    """统一计算换手和持有期，供策略摘要与验收门共同使用。"""
    sell_holding = (
        pd.to_numeric(
            actions.loc[actions["action"].eq("sell"), "held_sessions"],
            errors="coerce",
        ).dropna()
        if not actions.empty and {"action", "held_sessions"}.issubset(actions.columns)
        else pd.Series(dtype=float)
    )
    return {
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "trade_day_ratio": float(daily["turnover"].gt(1e-12).mean()),
        "completed_position_exit_count": int(len(sell_holding)),
        "completed_position_holding_sessions_p25": (
            float(sell_holding.quantile(0.25)) if not sell_holding.empty else np.nan
        ),
        "completed_position_holding_sessions_p50": (
            float(sell_holding.quantile(0.50)) if not sell_holding.empty else np.nan
        ),
        "completed_position_holding_sessions_p75": (
            float(sell_holding.quantile(0.75)) if not sell_holding.empty else np.nan
        ),
        "completed_position_holding_sessions_le_5_ratio": (
            float(sell_holding.le(5).mean()) if not sell_holding.empty else np.nan
        ),
    }


def _apply_rebalance_band(
    pretrade: dict[str, float],
    desired: dict[str, float],
    tolerance: float,
) -> dict[str, float]:
    """成员和总仓位不变时允许权重自然漂移，避免每日机械恢复等权。"""
    if tolerance < 0:
        raise ValueError("tolerance 不能为负数")
    if set(pretrade) != set(desired):
        return desired
    exposure_gap = abs(sum(pretrade.values()) - sum(desired.values()))
    largest_asset_gap = max(
        (abs(pretrade[code] - desired[code]) for code in desired),
        default=0.0,
    )
    if exposure_gap <= tolerance and largest_asset_gap <= tolerance:
        return pretrade.copy()
    return desired


def _execution_actions(
    signal_date: str,
    execution_date: str,
    decisions: pd.DataFrame,
    pretrade: dict[str, float],
    target: dict[str, float],
    turnover: float,
    cost_rate: float,
    regime: str,
    low_risk_code: str | None = None,
) -> pd.DataFrame:
    """把策略判断展开为可核对的实际权重变化。"""
    decision_lookup: dict[tuple[str, str], dict] = {}
    if not decisions.empty:
        for row in decisions.to_dict("records"):
            decision_lookup[(str(row["ts_code"]), str(row["action"]))] = row

    changed = []
    assets = sorted(set(pretrade) | set(target))
    weight_changes = [
        (code, float(pretrade.get(code, 0.0)), float(target.get(code, 0.0)), None)
        for code in assets
    ]
    if low_risk_code is not None:
        weight_changes.append(
            (
                "LOW_RISK",
                1.0 - float(sum(pretrade.values())),
                1.0 - float(sum(target.values())),
                low_risk_code,
            )
        )
    total_abs_change = sum(abs(target_weight - current_weight) for _, current_weight, target_weight, _ in weight_changes)
    for code, current_weight, target_weight, etf_code in weight_changes:
        change = target_weight - current_weight
        if abs(change) <= 1e-12:
            continue
        action = "buy" if change > 0 else "sell"
        decision = decision_lookup.get((code, action), {})
        trade_type = (
            "entry" if current_weight == 0.0 and change > 0
            else "exit" if target_weight == 0.0 and change < 0
            else "rebalance"
        )
        allocation = abs(change) / total_abs_change if total_abs_change > 0 else 0.0
        changed.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "ts_code": code,
                "action": action,
                "trade_type": trade_type,
                "reason": decision.get(
                    "reason",
                    "low_risk_residual_allocation" if code == "LOW_RISK" else "weight_rebalance",
                ),
                "held_sessions": decision.get("held_sessions", np.nan),
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_change": change,
                "portfolio_turnover": turnover,
                "portfolio_expected_cost": turnover * cost_rate,
                "allocated_expected_cost": turnover * cost_rate * allocation,
                "regime": regime,
                "etf_code": etf_code,
            }
        )
    return pd.DataFrame(changed)


def write_latest_advice(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    output_dir,
    sector_name_lookup: dict[str, str] | None = None,
    asof_date=None,
    market_data_end_date: str | None = None,
    latest_plan: dict | None = None,
    etf_execution_ready: bool = False,
) -> dict:
    """输出无需额外脚本即可阅读的最新交易建议和目标组合。"""
    advice_columns = [
        "信号日期",
        "计划执行日",
        "板块代码",
        "板块名称",
        "指令",
        "交易类型",
        "当前权重",
        "目标权重",
        "权重变化",
        "原因",
        "市场状态",
        "ETF代码",
    ]
    portfolio_columns = ["策略日期", "目标形成日期", "板块代码", "板块名称", "目标权重", "ETF代码"]
    status_columns = [
        "数据截止日",
        "数据年龄(天)",
        "数据状态",
        "策略日期",
        "信号同步状态",
        "指令状态",
        "最近调仓信号日",
        "策略动作",
        "执行提示",
        "市场状态",
        "当前板块仓位",
        "当前低风险仓位",
        "当前持仓数量",
    ]
    if daily.empty:
        for filename in ("LATEST_ACTIONS.csv", "LAST_REBALANCE_ACTIONS.csv"):
            pd.DataFrame(columns=advice_columns).to_csv(
                output_dir / filename, index=False, encoding="utf-8-sig"
            )
        pd.DataFrame(columns=portfolio_columns).to_csv(
            output_dir / "LATEST_TARGET_PORTFOLIO.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=status_columns).to_csv(
            output_dir / "LATEST_STATUS.csv", index=False, encoding="utf-8-sig"
        )
        return {
            "data_end_date": None,
            "data_age_days": None,
            "data_stale": True,
            "instruction_current": False,
            "action_plan_valid": False,
            "execution_allowed": False,
            "strategy_action": "无可用数据",
            "execution_message": "无可用数据，禁止执行",
        }

    if sector_name_lookup is None:
        names = pd.read_parquet(DATA_DIR / "ths_index.parquet", columns=["ts_code", "name"])
        name_lookup = names.drop_duplicates("ts_code").set_index("ts_code")["name"].to_dict()
    else:
        name_lookup = dict(sector_name_lookup)
    latest_strategy_date = str(
        latest_plan["signal_date"]
        if latest_plan and latest_plan.get("signal_date")
        else daily["signal_date"].dropna().max()
    )
    latest_daily = daily.sort_values(["signal_date", "date"]).iloc[-1]
    # 行情截止日与历史绩效结算日不是同一概念；优先使用原始行情日期。
    data_end_date = str(market_data_end_date or daily["date"].dropna().max())
    reference_date = pd.Timestamp(asof_date or pd.Timestamp.today()).normalize()
    data_date = pd.to_datetime(data_end_date, format="%Y%m%d").normalize()
    data_age_days = int((reference_date - data_date).days)
    data_stale = data_age_days > 7
    if actions.empty:
        last_rebalance_actions = pd.DataFrame()
        last_order_date = ""
    else:
        ordered = actions.sort_values(["signal_date", "execution_date", "ts_code"])
        last_order_date = str(ordered["signal_date"].max())
        last_rebalance_actions = ordered.loc[ordered["signal_date"].eq(last_order_date)].copy()
        last_rebalance_actions = last_rebalance_actions.loc[
            last_rebalance_actions["weight_change"].abs().ge(0.0005)
        ]

    if latest_plan is not None:
        current_actions = latest_plan.get("actions", pd.DataFrame()).copy()
        target_state = {
            str(code): (
                float(weight),
                DEFAULT_LOW_RISK_CODE if str(code) == "LOW_RISK" else None,
            )
            for code, weight in latest_plan.get("target_weights", {}).items()
            if float(weight) > 1e-12
        }
        residual = 1.0 - float(sum(weight for weight, _ in target_state.values()))
        if residual > 1e-12:
            target_state["LOW_RISK"] = (residual, DEFAULT_LOW_RISK_CODE)
    else:
        current_actions = (
            actions.loc[actions["signal_date"].astype(str).eq(latest_strategy_date)].copy()
            if not actions.empty
            else pd.DataFrame()
        )
        target_state: dict[str, tuple[float, str | None]] = {}
        if not actions.empty:
            for row in ordered.to_dict("records"):
                code = str(row["ts_code"])
                weight = float(row["target_weight"])
                if weight <= 1e-12:
                    target_state.pop(code, None)
                else:
                    etf_code = row.get("etf_code")
                    target_state[code] = (
                        weight,
                        str(etf_code) if pd.notna(etf_code) else None,
                    )

    # 人工建议不展示低于5bp的机械尾差，完整账本仍保留原始记录。
    if not current_actions.empty:
        current_actions = current_actions.loc[current_actions["weight_change"].abs().ge(0.0005)]

    def readable_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=advice_columns)
        return pd.DataFrame(
            {
                "信号日期": frame["signal_date"],
                "计划执行日": frame["execution_date"],
                "板块代码": frame["ts_code"],
                "板块名称": frame["ts_code"].map(name_lookup).fillna("货币ETF"),
                "指令": frame["action"].map(ACTION_LABELS).fillna(frame["action"]),
                "交易类型": frame["trade_type"]
                .map(TRADE_TYPE_LABELS)
                .fillna(frame["trade_type"]),
                "当前权重": frame["current_weight"].map(lambda value: f"{value:.2%}"),
                "目标权重": frame["target_weight"].map(lambda value: f"{value:.2%}"),
                "权重变化": frame["weight_change"].map(lambda value: f"{value:+.2%}"),
                "原因": frame["reason"].map(REASON_LABELS).fillna(frame["reason"]),
                "市场状态": frame["regime"],
                "ETF代码": frame["etf_code"],
            }
        )

    instruction_current = latest_strategy_date == data_end_date
    if latest_plan is not None:
        action_plan_valid = bool(
            latest_plan.get("stage") == "planned"
            and str(latest_plan.get("planned_execution_date") or "") > data_end_date
        )
    else:
        action_plan_valid = bool(
            current_actions.empty
            or (
                current_actions["signal_date"].astype(str).eq(data_end_date).all()
                and current_actions["execution_date"].astype(str).gt(data_end_date).all()
            )
        )
    execution_allowed = bool(
        not data_stale and instruction_current and action_plan_valid and etf_execution_ready
    )
    strategy_action = (
        "信号尚未推进到最新行情"
        if not instruction_current
        else "有新调仓建议"
        if not current_actions.empty
        else "持有不动"
    )
    actionable_actions = current_actions if execution_allowed else current_actions.iloc[0:0]
    if data_stale:
        execution_message = "数据已过期，禁止执行"
    elif not instruction_current:
        execution_message = "策略信号滞后于行情，禁止执行"
    elif not action_plan_valid:
        execution_message = "动作不是未来交易计划，禁止执行"
    elif not etf_execution_ready:
        execution_message = "ETF执行层未就绪，禁止执行"
    elif not actionable_actions.empty:
        execution_message = "按LATEST_ACTIONS人工复核"
    else:
        execution_message = "无需交易"
    readable_action_frame(actionable_actions).to_csv(
        output_dir / "LATEST_ACTIONS.csv", index=False, encoding="utf-8-sig"
    )
    readable_action_frame(last_rebalance_actions).to_csv(
        output_dir / "LAST_REBALANCE_ACTIONS.csv", index=False, encoding="utf-8-sig"
    )
    plan_preview = readable_action_frame(current_actions)
    plan_preview.insert(
        0,
        "指令状态",
        "待人工复核" if execution_allowed else "禁止执行",
    )
    plan_preview.to_csv(
        output_dir / "LATEST_PLAN_ACTIONS.csv", index=False, encoding="utf-8-sig"
    )
    plan_payload = {
        "stage": latest_plan.get("stage") if latest_plan else "historical_engine_only",
        "market_data_asof": data_end_date,
        "signal_date": latest_strategy_date,
        "planned_execution_date": (
            latest_plan.get("planned_execution_date") if latest_plan else None
        ),
        "simulated_portfolio_asof_date": (
            latest_plan.get("simulated_portfolio_asof_date") if latest_plan else latest_strategy_date
        ),
        "instruction_current": instruction_current,
        "action_plan_valid": action_plan_valid,
        "data_stale": data_stale,
        "execution_allowed": execution_allowed,
        "target_weights": {
            code: weight for code, (weight, _) in target_state.items()
        },
        "blockers": [
            reason
            for reason, blocked in (
                ("data_stale", data_stale),
                ("signal_lag", not instruction_current),
                ("calendar_or_plan_invalid", not action_plan_valid),
                ("etf_execution_not_ready", not etf_execution_ready),
            )
            if blocked
        ],
    }
    (output_dir / "LATEST_PLAN.json").write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        [
            {
                "数据截止日": data_end_date,
                "数据年龄(天)": data_age_days,
                "数据状态": "已过期" if data_stale else "可用" if instruction_current else "信号滞后",
                "策略日期": latest_strategy_date,
                "信号同步状态": "同日" if instruction_current else "滞后",
                "指令状态": "可人工复核" if execution_allowed else "禁止执行",
                "最近调仓信号日": last_order_date,
                "策略动作": strategy_action,
                "执行提示": execution_message,
                "市场状态": latest_plan.get("regime", latest_daily["regime"]) if latest_plan else latest_daily["regime"],
                "当前板块仓位": f"{float(latest_plan.get('exposure', latest_daily['exposure']) if latest_plan else latest_daily['exposure']):.2%}",
                "当前低风险仓位": f"{1.0 - float(latest_plan.get('exposure', latest_daily['exposure']) if latest_plan else latest_daily['exposure']):.2%}",
                "当前持仓数量": len([code for code in target_state if code != "LOW_RISK"]),
            }
        ],
        columns=status_columns,
    ).to_csv(output_dir / "LATEST_STATUS.csv", index=False, encoding="utf-8-sig")

    portfolio_rows = []
    for code, (weight, etf_code) in sorted(target_state.items(), key=lambda item: item[1][0], reverse=True):
        portfolio_rows.append(
            {
                "策略日期": latest_strategy_date,
                "目标形成日期": latest_strategy_date if latest_plan is not None else last_order_date,
                "板块代码": code,
                "板块名称": name_lookup.get(code, "货币ETF" if code == "LOW_RISK" else code),
                "目标权重": f"{weight:.2%}",
                "ETF代码": etf_code,
            }
        )
    pd.DataFrame(portfolio_rows, columns=portfolio_columns).to_csv(
        output_dir / "LATEST_TARGET_PORTFOLIO.csv", index=False, encoding="utf-8-sig"
    )
    return {
        "data_end_date": data_end_date,
        "data_age_days": data_age_days,
        "data_stale": data_stale,
        "instruction_current": instruction_current,
        "action_plan_valid": action_plan_valid,
        "execution_allowed": execution_allowed,
        "strategy_action": strategy_action,
        "execution_message": execution_message,
    }


def latest_market_risk_snapshot(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    policy: RegimePolicy | None = None,
    latest_plan: dict | None = None,
) -> dict:
    """独立推进到最新行情日，不受回测未来收益可用性裁切。"""
    policy = policy or RegimePolicy()
    market = build_market_state(panel, tuple(UNIVERSES["industry_concept"]))
    market = market.loc[market["trade_date"].ge(PRODUCT_HISTORY_START)].sort_values("trade_date")
    if market.empty:
        return {"status": "no_data", "scope": "sector_breadth_not_broad_market_index"}
    state = RegimeState()
    raw_regime = "NEUTRAL"
    for row in market.itertuples(index=False):
        raw_regime = classify_market(pd.Series(row._asdict()), policy)
        state = advance_regime(state, raw_regime, policy)
    latest = market.iloc[-1]
    planned_risk = latest_plan.get("risk", {}) if latest_plan else {}
    drawdown_cap = float(
        planned_risk.get(
            "drawdown_cap",
            daily.iloc[-1]["drawdown_cap"] if not daily.empty else 1.0,
        )
    )
    simulated_exposure = float(
        sum(latest_plan.get("current_weights", {}).values())
        if latest_plan
        else daily.iloc[-1]["exposure"]
        if not daily.empty
        else 0.0
    )
    portfolio_asof = (
        str(latest_plan.get("simulated_portfolio_asof_date"))
        if latest_plan
        else str(daily.iloc[-1]["signal_date"])
        if not daily.empty
        else None
    )
    regime_base = regime_exposure(state.current, policy)
    risk_target = float(planned_risk.get("risk_target_exposure", min(regime_base, drawdown_cap)))
    return {
        "status": "ready",
        "scope": "sector_breadth_not_broad_market_index",
        "score_direction": "high_is_risk_on",
        "risk_asof_date": str(latest["trade_date"]),
        "risk_score": float(latest["risk_score"]),
        "risk_data_quality": str(latest["risk_data_quality"]),
        "benchmark_trend_60d": float(latest["benchmark_trend_60d"]),
        "breadth_positive_20d": float(latest["risk_breadth_positive_20d"]),
        "breadth_positive_60d": float(latest["risk_breadth_positive_60d"]),
        "breadth_20d_valid_count": int(latest["breadth_20d_valid_count"]),
        "breadth_60d_valid_count": int(latest["breadth_60d_valid_count"]),
        "breadth_20d_coverage": float(latest["breadth_20d_coverage"]),
        "breadth_60d_coverage": float(latest["breadth_60d_coverage"]),
        "market_vol_percentile": float(latest["market_vol_percentile"]),
        "trend_health": float(latest["trend_health"]),
        "breadth_20d_health": float(latest["breadth_20d_health"]),
        "breadth_60d_health": float(latest["breadth_60d_health"]),
        "volatility_health": float(latest["volatility_health"]),
        "raw_regime": raw_regime,
        "confirmed_regime": latest_plan.get("regime", state.current) if latest_plan else state.current,
        "pending_regime": state.pending,
        "pending_sessions": state.pending_sessions,
        "regime_base_exposure": regime_base,
        "drawdown_cap": drawdown_cap,
        "drawdown_cap_asof_date": portfolio_asof,
        "risk_target_exposure": risk_target,
        "simulated_portfolio_exposure": simulated_exposure,
        "simulated_portfolio_asof_date": portfolio_asof,
        # 兼容旧消费者；这里从来不是券商账户的真实持仓。
        "actual_portfolio_exposure": simulated_exposure,
        "actual_portfolio_asof_date": portfolio_asof,
    }


def write_latest_market_risk(
    snapshot: dict,
    output_dir,
    *,
    data_freshness: dict | None = None,
) -> dict:
    """输出可直接读取的最新板块广度风险CSV和JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if snapshot.get("status") == "no_data":
        payload = snapshot
    else:
        payload = dict(snapshot)
        payload["data_age_days"] = data_freshness.get("data_age_days") if data_freshness else None
        stale = bool(data_freshness and data_freshness.get("data_stale"))
        quality_ok = payload["risk_data_quality"] == "complete"
        payload["status"] = "stale" if stale else "ready" if quality_ok else "insufficient"
        instruction_current = bool(data_freshness and data_freshness.get("instruction_current"))
        plan_valid = bool(data_freshness and data_freshness.get("action_plan_valid"))
        payload["diagnostic_available"] = bool(not stale and quality_ok)
        product_execution_allowed = bool(
            data_freshness and data_freshness.get("execution_allowed")
        )
        payload["execution_allowed"] = bool(quality_ok and product_execution_allowed)
        pending = payload.get("pending_regime")
        reasons = [
            f"分数越高越适合承担权益风险，当前{payload['risk_score']:.1f}分",
            f"60日趋势健康度{payload['trend_health']:.1%}",
            f"20/60日上涨板块占比{payload['breadth_positive_20d']:.1%}/"
            f"{payload['breadth_positive_60d']:.1%}",
            f"20/60日数据覆盖率{payload['breadth_20d_coverage']:.1%}/"
            f"{payload['breadth_60d_coverage']:.1%}",
            f"波动健康度{payload['volatility_health']:.1%}",
        ]
        if pending:
            reasons.append(
                f"原始状态{payload['raw_regime']}正在确认（{payload['pending_sessions']}日）"
            )
        if payload["drawdown_cap"] < payload["regime_base_exposure"]:
            reasons.append("组合回撤保护进一步压低仓位")
        if abs(payload["actual_portfolio_exposure"] - payload["risk_target_exposure"]) > 1e-6:
            reasons.append("实际仓位与风险目标存在执行时点或再平衡带差异")
        if not quality_ok:
            reasons.append("风险输入覆盖不足，禁止执行")
        if quality_ok and not instruction_current:
            reasons.append("产品信号尚未推进到最新行情日，风险诊断不可直接当作交易指令")
        elif quality_ok and not plan_valid:
            reasons.append("当前动作不是下一交易日计划，禁止执行")
        elif quality_ok and not product_execution_allowed:
            reasons.append("ETF执行层未就绪，风险诊断不可直接当作交易指令")
        payload["reason"] = "；".join(reasons)
    pd.DataFrame([payload]).to_csv(
        output_dir / "LATEST_MARKET_RISK.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "LATEST_MARKET_RISK.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def append_latest_signal_strength(
    panel: pd.DataFrame,
    score_name: str,
    policy: StrategyPolicy,
    daily: pd.DataFrame,
    risk_snapshot: dict,
    output_dir,
) -> None:
    """给最新目标组合补充相对排名与风险调整后的绝对买入强度。"""
    portfolio_path = output_dir / "LATEST_TARGET_PORTFOLIO.csv"
    if daily.empty or not portfolio_path.exists():
        return
    portfolio = pd.read_csv(portfolio_path, dtype={"板块代码": str})
    for column in (
        "模型相对排名",
        "模型相对强度",
        "板块广度健康分",
        "风险目标仓位",
        "连续风险调整强度",
    ):
        portfolio[column] = ""
    signal_date = str(risk_snapshot["risk_asof_date"])
    usable = panel.loc[
        panel["type"].isin(UNIVERSES["industry_concept"])
        & panel["trade_date"].le(signal_date)
    ].sort_values(["ts_code", "trade_date"])
    # 每个板块独立取自身最近三个观测，与正式回测的rolling口径一致。
    recent = usable.groupby("ts_code", sort=False).tail(policy.score_smoothing_sessions).copy()
    if score_name not in recent.columns:
        recent = add_formula_scores(recent)
    if score_name not in recent.columns:
        return
    latest_dates = recent.groupby("ts_code", sort=False)["trade_date"].max()
    scores = recent.groupby("ts_code", sort=False)[score_name].mean().dropna()
    scores = scores.loc[latest_dates.reindex(scores.index).eq(signal_date)]
    if scores.empty:
        return
    ordered = (
        scores.rename("score")
        .reset_index()
        .sort_values(["score", "ts_code"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered["ts_code"])
    strength = scores.rank(method="average", pct=True)
    risk_target = float(risk_snapshot["risk_target_exposure"])
    risk_score = float(risk_snapshot["risk_score"])
    for index, row in portfolio.iterrows():
        code = str(row["板块代码"])
        if code == "LOW_RISK" or code not in scores.index:
            continue
        relative_strength = float(strength.loc[code])
        portfolio.at[index, "模型相对排名"] = str(int(ranks.loc[code]))
        portfolio.at[index, "模型相对强度"] = f"{relative_strength:.2%}"
        portfolio.at[index, "板块广度健康分"] = f"{risk_score:.1f}"
        portfolio.at[index, "风险目标仓位"] = f"{risk_target:.2%}"
        portfolio.at[index, "连续风险调整强度"] = f"{relative_strength * risk_score / 100.0:.2%}"
    portfolio.to_csv(portfolio_path, index=False, encoding="utf-8-sig")


def run_product_backtest(
    panel: pd.DataFrame,
    score_name: str,
    start: str,
    end: str,
    universe: str = "industry_concept",
    cost_bps: float = 20.0,
    strategy_policy: StrategyPolicy | None = None,
    regime_policy: RegimePolicy | None = None,
    low_risk_frame: pd.DataFrame | None = None,
    use_market_regime: bool = True,
    use_drawdown_cap: bool = True,
    use_continuous_defensive_exposure: bool = False,
    use_sector_strength_override: bool = False,
    latest_plan_sink: dict | None = None,
    planned_execution_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """运行状态化产品回测，所有决策在收盘后产生并于次日开盘执行。"""
    if universe not in UNIVERSES:
        raise ValueError(f"未知投资宇宙: {universe}")
    if cost_bps < 0:
        raise ValueError("cost_bps 不能为负数")

    base_policy = strategy_policy or StrategyPolicy()
    regime_policy = regime_policy or RegimePolicy()
    sub = prepare_product_panel(panel, score_name, universe)
    if score_name not in sub.columns:
        raise ValueError(f"评分字段不存在: {score_name}")
    sub = sub.sort_values(["trade_date", "ts_code"])
    product_score = "_product_score"
    sub[product_score] = sub.groupby("ts_code", sort=False)[score_name].transform(
        lambda values: values.rolling(base_policy.score_smoothing_sessions, min_periods=1).mean()
    )
    # 只要求订单执行日已经出现；再下一开盘仅属于收益结算，不能反向参与信号。
    score_dates = sub.loc[
        (sub["trade_date"] >= start)
        & (sub["trade_date"] <= end)
        & sub["next_open_date"].notna()
        & sub["next_open_date"].le(end),
        "trade_date",
    ].drop_duplicates().tolist()
    if not score_dates:
        return pd.DataFrame(), pd.DataFrame(), annualized_metrics(pd.Series(dtype=float))

    # groupby 只保存分组索引；避免 set_index 再复制一份完整产品面板。
    daily_groups = sub.groupby("trade_date", sort=False, observed=True)
    return_pivot = sub.pivot(index="trade_date", columns="ts_code", values="forward_open_ret_1d")
    open_pivot = sub.pivot(index="trade_date", columns="ts_code", values="open")
    close_pivot = sub.pivot(index="trade_date", columns="ts_code", values="close")
    date_map = (
        sub[["trade_date", "next_open_date", "return_end_date"]]
        .drop_duplicates("trade_date")
        .set_index("trade_date")
    )
    low_risk_code: str | None = None
    low_risk_lookup: pd.DataFrame | None = None
    if low_risk_frame is not None:
        required = {"trade_date", "low_risk_code", "intraday_return", "forward_open_ret_1d"}
        missing_columns = required - set(low_risk_frame.columns)
        if missing_columns:
            raise ValueError(f"低风险收益缺少字段: {sorted(missing_columns)}")
        codes = low_risk_frame["low_risk_code"].dropna().astype(str).unique()
        if len(codes) != 1:
            raise ValueError("低风险收益必须对应唯一冻结ETF")
        low_risk_code = str(codes[0])
        low_risk_lookup = low_risk_frame.drop_duplicates("trade_date").set_index("trade_date")

    def low_risk_return(signal_date: str, column: str) -> float:
        if low_risk_lookup is None:
            return 0.0
        if signal_date not in low_risk_lookup.index:
            raise RuntimeError(f"{signal_date} 缺少低风险ETF收益")
        value = float(low_risk_lookup.loc[signal_date, column])
        if not np.isfinite(value):
            raise RuntimeError(f"{signal_date} 低风险ETF字段 {column} 无效")
        return value

    market = build_market_state(sub, tuple(UNIVERSES[universe])).set_index("trade_date")

    regime_state = RegimeState()
    drawdown_state = DrawdownState()
    positions: dict[str, PositionState] = {}
    live_weights: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    risk_peak = 1.0
    cost_rate = cost_bps / 10_000.0
    rows: list[dict] = []
    actions: list[pd.DataFrame] = []
    position_entry_open: dict[str, float] = {}
    position_close_peak: dict[str, float] = {}

    def decide(signal_date: str) -> tuple[dict[str, float], str, float, bool, pd.DataFrame, dict]:
        nonlocal positions, regime_state, drawdown_state, risk_peak
        market_row = market.loc[signal_date]
        if use_market_regime:
            raw_regime = classify_market(market_row, regime_policy)
            regime_state = advance_regime(regime_state, raw_regime, regime_policy)
        else:
            regime_state = RegimeState(current="RISK_ON")

        # 收盘先盯市，再生成下一交易日开盘订单；不把次日开盘收益用于当日判断。
        marked_equity = equity
        position_returns: dict[str, float] = {}
        position_drawdowns: dict[str, float] = {}
        low_risk_weight = 1.0 - float(sum(live_weights.values()))
        low_risk_intraday = low_risk_return(signal_date, "intraday_return")
        if live_weights:
            signal_open = open_pivot.loc[signal_date].reindex(live_weights)
            signal_close = close_pivot.loc[signal_date].reindex(live_weights)
            missing_mark = signal_open.isna() | signal_close.isna()
            # 停牌/缺报价时沿用上一估值，不把未来是否恢复交易反馈给当日信号。
            intraday_returns = (signal_close / signal_open - 1.0).where(~missing_mark, 0.0)
            marked_equity *= 1.0 + float(
                sum(live_weights[code] * intraday_returns.loc[code] for code in live_weights)
                + low_risk_weight * low_risk_intraday
            )
            for code in live_weights:
                if bool(missing_mark.loc[code]):
                    continue
                entry_open = position_entry_open.get(code)
                if entry_open is None or not np.isfinite(entry_open) or entry_open <= 0:
                    raise RuntimeError(f"持仓 {code} 缺少有效入场开盘价")
                nav = float(signal_close.loc[code] / entry_open)
                close_peak = max(position_close_peak.get(code, 1.0), nav)
                position_close_peak[code] = close_peak
                position_returns[code] = nav - 1.0
                position_drawdowns[code] = nav / close_peak - 1.0
        elif low_risk_weight > 0.0:
            marked_equity *= 1.0 + low_risk_weight * low_risk_intraday
        risk_peak = max(risk_peak, marked_equity)
        drawdown = marked_equity / risk_peak - 1.0
        if use_drawdown_cap:
            drawdown_state = advance_drawdown_state(drawdown_state, drawdown)
        day_source = daily_groups.get_group(signal_date)
        strength_override = (
            use_sector_strength_override
            and regime_state.current == "DEFENSIVE"
            and leading_sector_strength(day_source, product_score)
        )
        if use_market_regime and use_continuous_defensive_exposure:
            market_exposure = technical_regime_exposure(
                regime_state.current, market_row, regime_policy
            )
        elif use_market_regime:
            market_exposure = regime_exposure(regime_state.current, regime_policy)
        else:
            market_exposure = 1.0
        regime_base_exposure = market_exposure
        # 大盘防御但领先板块自身仍强时保留七成仓位；组合回撤上限仍可继续压低它。
        if strength_override:
            market_exposure = max(market_exposure, 0.7)
        drawdown_cap = drawdown_state.exposure_cap if use_drawdown_cap else 1.0
        exposure = min(market_exposure, drawdown_cap)
        target_positions = (
            base_policy.target_positions
            if strength_override and exposure > 0
            else REGIME_POSITION_LIMITS[regime_state.current]
            if use_market_regime and exposure > 0
            else base_policy.target_positions if exposure > 0 else 0
        )
        policy = replace(base_policy, target_positions=target_positions)
        columns = [
            "ts_code", product_score, "ret_5d_rank", "ret_20d",
            "volatility_20d", "volatility_20d_rank",
        ]
        available = [column for column in columns if column in day_source.columns]
        daily = day_source[available].copy().rename(columns={product_score: "score"})
        daily = daily.dropna(subset=["score"]).copy()
        # 计划在收盘生成，不能查看下一开盘是否缺价，更不能查看再下一开盘。
        # 缺价应在真实执行/结算事件发生时记录，而不是改变当日排名和目标。
        daily["position_return"] = daily["ts_code"].map(position_returns)
        daily["position_drawdown"] = daily["ts_code"].map(position_drawdowns)
        positions, targets, decisions = step_portfolio(signal_date, daily, positions, exposure, policy)
        risk_details = {
            "risk_score": float(market_row["risk_score"]),
            "risk_data_quality": str(market_row["risk_data_quality"]),
            "benchmark_trend_60d": float(market_row["benchmark_trend_60d"]),
            "breadth_positive_20d": float(market_row["risk_breadth_positive_20d"]),
            "breadth_positive_60d": float(market_row["risk_breadth_positive_60d"]),
            "breadth_20d_valid_count": int(market_row["breadth_20d_valid_count"]),
            "breadth_60d_valid_count": int(market_row["breadth_60d_valid_count"]),
            "breadth_20d_coverage": float(market_row["breadth_20d_coverage"]),
            "breadth_60d_coverage": float(market_row["breadth_60d_coverage"]),
            "market_vol_percentile": float(market_row["market_vol_percentile"]),
            "trend_health": float(market_row["trend_health"]),
            "breadth_20d_health": float(market_row["breadth_20d_health"]),
            "breadth_60d_health": float(market_row["breadth_60d_health"]),
            "volatility_health": float(market_row["volatility_health"]),
            "raw_regime": raw_regime if use_market_regime else "RISK_ON",
            "regime_pending": regime_state.pending,
            "regime_pending_sessions": regime_state.pending_sessions,
            "regime_base_exposure": regime_base_exposure,
            "market_exposure_after_override": market_exposure,
            "drawdown_cap": drawdown_cap,
            "risk_target_exposure": exposure,
        }
        return (
            _target_dict(targets),
            regime_state.current,
            exposure,
            strength_override,
            decisions,
            risk_details,
        )

    first_signal = score_dates[0]
    (
        first_target,
        first_regime,
        first_exposure,
        first_override,
        first_decisions,
        first_risk_details,
    ) = decide(first_signal)
    last_strength_override = first_override
    last_risk_details = first_risk_details
    first_execution_date = str(date_map.loc[first_signal, "next_open_date"])
    first_available = open_pivot.loc[first_execution_date].notna()
    first_unfilled = [code for code in first_target if not bool(first_available.get(code, False))]
    first_target = {
        code: weight for code, weight in first_target.items() if code not in first_unfilled
    }
    positions = {code: state for code, state in positions.items() if code in first_target}
    initial_turnover = _turnover({}, first_target)
    initial_net = (1.0 - initial_turnover * cost_rate) - 1.0
    equity *= 1.0 + initial_net
    peak = max(peak, equity)
    live_weights = first_target
    first_execution_prices = open_pivot.loc[first_execution_date].reindex(live_weights)
    position_entry_open = {code: float(first_execution_prices.loc[code]) for code in live_weights}
    position_close_peak = {code: 1.0 for code in live_weights}
    first_actions = _execution_actions(
        first_signal,
        first_execution_date,
        first_decisions,
        {},
        first_target,
        initial_turnover,
        cost_rate,
        first_regime,
        low_risk_code,
    )
    if not first_actions.empty:
        actions.append(first_actions)
    rows.append(
        {
            "date": first_execution_date,
            "signal_date": first_signal,
            "gross_return": 0.0,
            "sector_contribution": 0.0,
            "low_risk_contribution": 0.0,
            "turnover": initial_turnover,
            "cost": initial_turnover * cost_rate,
            "net_return": initial_net,
            "equity": equity,
            "drawdown": equity / peak - 1.0,
            "regime": first_regime,
            "exposure": sum(live_weights.values()),
            "sector_strength_override": first_override,
            "low_risk_code": low_risk_code,
            "low_risk_weight": 1.0 - sum(live_weights.values()),
            "low_risk_return": 0.0,
                "position_count": len(live_weights),
            "missing_valuation_count": 0,
            "unfilled_order_count": len(first_unfilled),
            **first_risk_details,
        }
    )

    for index in range(1, len(score_dates)):
        signal_date = score_dates[index]
        prior_positions = dict(positions)
        (
            desired_target,
            regime,
            exposure,
            strength_override,
            decisions,
            risk_details,
        ) = decide(signal_date)
        last_strength_override = strength_override
        last_risk_details = risk_details
        prior_signal = score_dates[index - 1]
        asset_returns = return_pivot.loc[prior_signal].reindex(live_weights)
        missing_valuation = asset_returns.isna()
        # 已持仓遇到停牌/缺报价时采用价格持平估值，并在账本中显式计数。
        asset_returns = asset_returns.fillna(0.0)
        low_risk_weight = 1.0 - float(sum(live_weights.values()))
        period_low_risk_return = low_risk_return(prior_signal, "forward_open_ret_1d")
        sector_contribution = float(
            sum(live_weights[code] * asset_returns.loc[code] for code in live_weights)
        )
        low_risk_contribution = low_risk_weight * period_low_risk_return
        gross_return = float(sector_contribution + low_risk_contribution)
        pretrade = _drift_weights(live_weights, asset_returns, gross_return)
        target = _apply_rebalance_band(
            pretrade,
            desired_target,
            base_policy.rebalance_tolerance,
        )
        execution_date = str(date_map.loc[signal_date, "next_open_date"])
        execution_available = open_pivot.loc[execution_date].notna()
        unfilled_codes = []
        for code in set(pretrade) | set(target):
            if not bool(execution_available.get(code, False)):
                unfilled_codes.append(code)
                if pretrade.get(code, 0.0) > 1e-12:
                    target[code] = pretrade[code]
                else:
                    target.pop(code, None)
        positions = {
            code: positions.get(code, prior_positions.get(code))
            for code in target
            if positions.get(code, prior_positions.get(code)) is not None
        }
        turnover = _turnover(pretrade, target)
        net_return = (1.0 + gross_return) * (1.0 - turnover * cost_rate) - 1.0
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        risk_peak = max(risk_peak, equity)
        prior_codes = set(live_weights)
        live_weights = target
        execution_prices = open_pivot.loc[execution_date].reindex(live_weights)
        for code in prior_codes - set(live_weights):
            position_entry_open.pop(code, None)
            position_close_peak.pop(code, None)
        for code in set(live_weights) - prior_codes:
            position_entry_open[code] = float(execution_prices.loc[code])
            position_close_peak[code] = 1.0
        executed = _execution_actions(
            signal_date,
            execution_date,
            decisions,
            pretrade,
            target,
            turnover,
            cost_rate,
            regime,
            low_risk_code,
        )
        if not executed.empty:
            actions.append(executed)
        rows.append(
            {
                "date": str(date_map.loc[prior_signal, "return_end_date"]),
                "signal_date": signal_date,
                "gross_return": gross_return,
                "sector_contribution": sector_contribution,
                "low_risk_contribution": low_risk_contribution,
                "turnover": turnover,
                "cost": turnover * cost_rate,
                "net_return": net_return,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "regime": regime,
                "exposure": sum(live_weights.values()),
                "sector_strength_override": strength_override,
                "low_risk_code": low_risk_code,
                "low_risk_weight": 1.0 - sum(live_weights.values()),
                "low_risk_return": period_low_risk_return,
                "position_count": len(live_weights),
                "missing_valuation_count": int(missing_valuation.sum()),
                "unfilled_order_count": len(unfilled_codes),
                **risk_details,
            }
        )

    # 只有再下一开盘已经出现时才结算最后一段；否则保留为executed_unsettled。
    last_signal = score_dates[-1]
    last_return_end = date_map.loc[last_signal, "return_end_date"]
    if pd.notna(last_return_end) and str(last_return_end) <= end:
        last_returns = return_pivot.loc[last_signal].reindex(live_weights)
        final_missing_valuation = last_returns.isna()
        last_returns = last_returns.fillna(0.0)
        final_low_risk_weight = 1.0 - float(sum(live_weights.values()))
        final_low_risk_return = low_risk_return(last_signal, "forward_open_ret_1d")
        final_sector_contribution = float(
            sum(live_weights[code] * last_returns.loc[code] for code in live_weights)
        )
        final_low_risk_contribution = final_low_risk_weight * final_low_risk_return
        final_gross = float(final_sector_contribution + final_low_risk_contribution)
        equity *= 1.0 + final_gross
        peak = max(peak, equity)
        rows.append(
            {
                "date": str(last_return_end),
                "signal_date": last_signal,
                "gross_return": final_gross,
                "sector_contribution": final_sector_contribution,
                "low_risk_contribution": final_low_risk_contribution,
                "turnover": 0.0,
                "cost": 0.0,
                "net_return": final_gross,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "regime": regime_state.current,
                "exposure": sum(live_weights.values()),
                "sector_strength_override": last_strength_override,
                "low_risk_code": low_risk_code,
                "low_risk_weight": final_low_risk_weight,
                "low_risk_return": final_low_risk_return,
                "position_count": len(live_weights),
                "missing_valuation_count": int(final_missing_valuation.sum()),
                "unfilled_order_count": 0,
                **last_risk_details,
            }
        )

    # 最新收盘只形成planned目标，不把未知下一开盘伪装成已成交或收益。
    latest_signal_date = str(sub.loc[sub["trade_date"].le(end), "trade_date"].max())
    if latest_plan_sink is not None and latest_signal_date > last_signal:
        current_weights = dict(live_weights)
        (
            planned_target,
            planned_regime,
            planned_exposure,
            planned_override,
            planned_decisions,
            planned_risk,
        ) = decide(latest_signal_date)
        preview_turnover = _turnover(current_weights, planned_target)
        preview_actions = _execution_actions(
            latest_signal_date,
            planned_execution_date or "",
            planned_decisions,
            current_weights,
            planned_target,
            preview_turnover,
            cost_rate,
            planned_regime,
            low_risk_code,
        )
        latest_plan_sink.update(
            {
                "stage": "planned" if planned_execution_date else "blocked_calendar_missing",
                "signal_date": latest_signal_date,
                "planned_execution_date": planned_execution_date,
                "simulated_portfolio_asof_date": str(date_map.loc[last_signal, "next_open_date"]),
                "current_weights": current_weights,
                "target_weights": planned_target,
                "preview_turnover": preview_turnover,
                "preview_cost": preview_turnover * cost_rate,
                "regime": planned_regime,
                "exposure": planned_exposure,
                "sector_strength_override": planned_override,
                "risk": planned_risk,
                "actions": preview_actions,
            }
        )

    daily_result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    action_result = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame()
    sell_holding_periods = (
        action_result.loc[action_result["action"].eq("sell"), "held_sessions"]
        if not action_result.empty
        else pd.Series(dtype=float)
    )
    metrics = annualized_metrics(daily_result.set_index("date")["net_return"])
    metrics.update(
        {
            "avg_turnover": float(daily_result["turnover"].mean()),
            "total_cost": float(daily_result["cost"].sum()),
            "median_position_count": float(daily_result["position_count"].median()),
            "median_holding_sessions": (
                float(sell_holding_periods.median()) if not sell_holding_periods.empty else np.nan
            ),
            **execution_stability_metrics(daily_result, action_result),
        }
    )
    return daily_result, action_result, metrics


def summarize_backtest_period(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """从连续状态路径切出报告区间，不在区间边界重新初始化持仓。"""
    period_daily = daily[daily["date"].between(start, end)].copy()
    if actions.empty or "execution_date" not in actions.columns:
        period_actions = actions.iloc[0:0].copy()
    else:
        period_actions = actions[actions["execution_date"].between(start, end)].copy()
    metrics = annualized_metrics(period_daily.set_index("date")["net_return"])
    sell_holding_periods = (
        period_actions.loc[period_actions["action"].eq("sell"), "held_sessions"]
        if not period_actions.empty
        else pd.Series(dtype=float)
    )
    metrics.update(
        {
            "avg_turnover": float(period_daily["turnover"].mean()),
            "total_cost": float(period_daily["cost"].sum()),
            "median_position_count": float(period_daily["position_count"].median()),
            "median_holding_sessions": (
                float(sell_holding_periods.median()) if not sell_holding_periods.empty else np.nan
            ),
            "avg_exposure": float(period_daily["exposure"].mean()),
            "avg_low_risk_weight": float(period_daily["low_risk_weight"].mean()),
            **execution_stability_metrics(period_daily, period_actions),
        }
    )
    return period_daily, period_actions, metrics


def build_cost_sensitivity_frame(
    daily_path: pd.DataFrame,
    actions_path: pd.DataFrame,
    *,
    cost_bps: float,
    score_name: str,
    policy_name: str,
) -> pd.DataFrame:
    """汇总单一成本路径；供隔离子进程回传小型结果表。"""
    rows = []
    for period, start, end in (
        ("development", PRODUCT_HISTORY_START, TRAIN_END),
        ("selection", VAL_START, VAL_END),
        ("full", PRODUCT_HISTORY_START, VAL_END),
        ("observation", OBSERVATION_START, OBSERVATION_END),
    ):
        daily, _, metrics = summarize_backtest_period(daily_path, actions_path, start, end)
        rows.append(
            {
                "period": period,
                "boundary_mode": "continuous_carry",
                "score_name": score_name,
                "policy_name": policy_name,
                "cost_bps": cost_bps,
                "full_path_rerun": True,
                "stress_kind": "full_system_replay_with_drawdown_feedback",
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "strategy_policy_version": STRATEGY_POLICY_VERSION,
                "low_risk_data_signature": low_risk_data_signature(),
                **metrics,
                "avg_turnover": float(daily["turnover"].mean()),
            }
        )
    history = daily_path.loc[daily_path["date"].le(VAL_END)]
    for year, year_daily in history.groupby(history["date"].str[:4]):
        metrics = annualized_metrics(year_daily.set_index("date")["net_return"])
        rows.append(
            {
                "period": f"year_{year}",
                "boundary_mode": "continuous_carry",
                "score_name": score_name,
                "policy_name": policy_name,
                "cost_bps": cost_bps,
                "full_path_rerun": True,
                "stress_kind": "full_system_replay_with_drawdown_feedback",
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "strategy_policy_version": STRATEGY_POLICY_VERSION,
                "low_risk_data_signature": low_risk_data_signature(),
                **metrics,
                "avg_turnover": float(year_daily["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_acceptance_gate(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    output_dir,
    *,
    policy_name: str,
    policy: StrategyPolicy,
    score_name: str,
    cost_sensitivity: pd.DataFrame | None = None,
) -> dict:
    """用固定2018—2025门槛评价候选；2026永不参与晋级判断。"""
    selection_daily, selection_actions, metrics = summarize_backtest_period(
        daily,
        actions,
        PRODUCT_HISTORY_START,
        VAL_END,
    )
    if selection_daily.empty or selection_daily["date"].max() > VAL_END:
        raise RuntimeError("验收账本必须严格截止于2025年末")
    year_counts = selection_daily.groupby(selection_daily["date"].str[:4]).size()
    expected_years = {str(year) for year in range(2018, 2026)}
    coverage_complete = bool(
        set(year_counts.index) == expected_years
        and selection_daily["date"].min().startswith("2018")
        and selection_daily["date"].max() == VAL_END
        and year_counts.ge(200).all()
    )
    annual_returns = {
        int(year): float((1.0 + part["net_return"]).prod() - 1.0)
        for year, part in selection_daily.groupby(selection_daily["date"].str[:4])
    }
    positive_years = sum(value > 0.0 for value in annual_returns.values())
    meaningful_positive_years = sum(value >= 0.03 for value in annual_returns.values())
    worst_year = min(annual_returns.values())
    development = summarize_backtest_period(
        daily, actions, PRODUCT_HISTORY_START, TRAIN_END
    )[2]
    selection = summarize_backtest_period(daily, actions, VAL_START, VAL_END)[2]
    cumulative_threshold = float((1.0 + 0.07) ** (len(selection_daily) / 252.0) - 1.0)

    def gate(name: str, threshold: str, value, passed: bool | None) -> dict:
        return {"name": name, "threshold": threshold, "value": value, "pass": passed}

    gates = [
        gate(
            "complete_2018_2025_coverage",
            "年份必须完整为2018—2025且每年>=200日",
            {
                "first": selection_daily["date"].min(),
                "last": selection_daily["date"].max(),
                "days": {str(year): int(count) for year, count in year_counts.items()},
            },
            coverage_complete,
        ),
        gate(
            "cumulative_return",
            f">= {cumulative_threshold:.2%}（与7%年化同口径）",
            metrics["total_ret"],
            metrics["total_ret"] >= cumulative_threshold,
        ),
        gate("annualized_return", ">= 7.00%", metrics["ann_ret"], metrics["ann_ret"] >= 0.07),
        gate(
            "development_return",
            ">= 20.00%（2018—2023）",
            development["total_ret"],
            development["total_ret"] >= 0.20,
        ),
        gate(
            "selection_return",
            ">= 30.00%（2024—2025）",
            selection["total_ret"],
            selection["total_ret"] >= 0.30,
        ),
        gate(
            "selection_years_positive",
            "2024与2025分别为正",
            {"2024": annual_returns.get(2024), "2025": annual_returns.get(2025)},
            annual_returns.get(2024, -1.0) > 0 and annual_returns.get(2025, -1.0) > 0,
        ),
        gate("maximum_drawdown", ">= -20.00%", metrics["max_drawdown"], metrics["max_drawdown"] >= -0.20),
        gate("positive_years", ">= 6 of 8", positive_years, positive_years >= 6),
        gate("years_above_3pct", ">= 5 of 8", meaningful_positive_years, meaningful_positive_years >= 5),
        gate("worst_calendar_year", ">= -15.00%", worst_year, worst_year >= -0.15),
        gate("average_daily_turnover", "<= 7.50%", metrics["avg_turnover"], metrics["avg_turnover"] <= 0.075),
        gate("annualized_turnover", "<= 19.0x", metrics["annualized_turnover"], metrics["annualized_turnover"] <= 19.0),
        gate("trade_day_ratio", "<= 40.00%", metrics["trade_day_ratio"], metrics["trade_day_ratio"] <= 0.40),
        gate(
            "median_holding_sessions",
            ">= 8",
            metrics["completed_position_holding_sessions_p50"],
            metrics["completed_position_holding_sessions_p50"] >= 8,
        ),
    ]

    risk_required = {
        "risk_score",
        "risk_data_quality",
        "raw_regime",
        "regime",
        "regime_base_exposure",
        "market_exposure_after_override",
        "drawdown_cap",
        "risk_target_exposure",
    }
    risk_schema_ok = risk_required.issubset(selection_daily.columns)
    risk_identity_ok = False
    if risk_schema_ok:
        expected_target = selection_daily[["market_exposure_after_override", "drawdown_cap"]].min(axis=1)
        risk_identity_ok = bool(
            np.allclose(expected_target, selection_daily["risk_target_exposure"], atol=1e-12)
        )
    gates.append(
        gate(
            "risk_explanation_schema",
            "风险字段齐全且risk_target=min(market_after_override, drawdown_cap)",
            {"required_fields_present": risk_schema_ok, "identity_holds": risk_identity_ok},
            risk_schema_ok and risk_identity_ok,
        )
    )
    latest_risk_path = output_dir / "LATEST_MARKET_RISK.json"
    latest_portfolio_path = output_dir / "LATEST_TARGET_PORTFOLIO.csv"
    latest_plan_path = output_dir / "LATEST_PLAN.json"
    execution_readiness_path = output_dir.parent / "etf_mapping" / "ETF_EXECUTION_READINESS.json"
    latest_actions_path = output_dir / "LATEST_ACTIONS.csv"
    latest_output_ok = False
    latest_output_detail: dict = {"files_present": False}
    if latest_risk_path.exists() and latest_portfolio_path.exists():
        latest_risk = json.loads(latest_risk_path.read_text(encoding="utf-8"))
        latest_portfolio = pd.read_csv(latest_portfolio_path)
        required_latest_risk = {
            "risk_asof_date",
            "risk_score",
            "risk_data_quality",
            "regime_base_exposure",
            "drawdown_cap",
            "risk_target_exposure",
            "actual_portfolio_exposure",
            "simulated_portfolio_exposure",
        }
        required_portfolio = {
            "模型相对排名",
            "模型相对强度",
            "板块广度健康分",
            "风险目标仓位",
            "连续风险调整强度",
        }
        latest_output_ok = bool(
            required_latest_risk.issubset(latest_risk)
            and required_portfolio.issubset(latest_portfolio.columns)
            and str(latest_risk.get("risk_asof_date")) == str(daily["date"].max())
        )
        latest_output_detail = {
            "files_present": True,
            "risk_asof_date": latest_risk.get("risk_asof_date"),
            "data_end_date": str(daily["date"].max()),
            "risk_schema_ok": required_latest_risk.issubset(latest_risk),
            "portfolio_schema_ok": required_portfolio.issubset(latest_portfolio.columns),
        }
    gates.append(
        gate(
            "latest_output_compatibility",
            "最新风险日等于数据截止日且风险/目标组合字段齐全",
            latest_output_detail,
            latest_output_ok,
        )
    )

    execution_safety_ok = False
    execution_safety_detail = {"files_present": False}
    if latest_plan_path.exists() and execution_readiness_path.exists() and latest_actions_path.exists():
        latest_plan = json.loads(latest_plan_path.read_text(encoding="utf-8"))
        execution_readiness = json.loads(execution_readiness_path.read_text(encoding="utf-8"))
        latest_actions = pd.read_csv(latest_actions_path)
        execution_ready = bool(execution_readiness.get("execution_ready"))
        blockers = execution_readiness.get("blockers", [])
        safe_block = bool(execution_ready or (blockers and latest_actions.empty))
        date_aligned = str(latest_plan.get("signal_date")) == str(latest_plan.get("market_data_asof"))
        execution_safety_ok = bool(safe_block and date_aligned)
        execution_safety_detail = {
            "files_present": True,
            "plan_stage": latest_plan.get("stage"),
            "signal_date": latest_plan.get("signal_date"),
            "market_data_asof": latest_plan.get("market_data_asof"),
            "execution_ready": execution_ready,
            "blockers": blockers,
            "latest_action_rows": int(len(latest_actions)),
            "safe_block_holds": safe_block,
        }
    gates.append(
        gate(
            "execution_safety_interlock",
            "最新计划与行情同日；执行层未就绪时LATEST_ACTIONS必须为空",
            execution_safety_detail,
            execution_safety_ok,
        )
    )

    cost_gate_status: bool | None = None
    cost_detail: dict = {"status": "not_evaluated"}
    if cost_sensitivity is not None and not cost_sensitivity.empty:
        lookup = cost_sensitivity.set_index(["cost_bps", "period"])
        required = [
            (20.0, "full"),
            (30.0, "development"),
            (30.0, "selection"),
            (30.0, "full"),
            (50.0, "full"),
        ]
        if all(key in lookup.index for key in required):
            dev30 = float(lookup.loc[(30.0, "development"), "total_ret"])
            selection30 = float(lookup.loc[(30.0, "selection"), "total_ret"])
            full20 = float(lookup.loc[(20.0, "full"), "total_ret"])
            full30 = float(lookup.loc[(30.0, "full"), "total_ret"])
            full50 = float(lookup.loc[(50.0, "full"), "total_ret"])
            cost_gate_status = bool(
                dev30 > 0.0
                and selection30 > 0.0
                and full30 >= 0.60 * full20
                and full50 > 0.0
            )
            cost_detail = {
                "status": "evaluated",
                "development_30bp": dev30,
                "selection_30bp": selection30,
                "full_20bp": full20,
                "full_30bp": full30,
                "full_50bp": full50,
                "retention_30bp_vs_20bp": full30 / full20 if full20 else None,
            }
    gates.append(
        gate(
            "cost_robustness",
            "30bp开发/选择期为正且保留20bp收益60%；50bp全期为正",
            cost_detail,
            cost_gate_status,
        )
    )
    evaluated = [item["pass"] for item in gates if item["pass"] is not None]
    accepted = bool(len(evaluated) == len(gates) and all(evaluated))
    policy_payload = json.dumps(asdict(policy), ensure_ascii=False, sort_keys=True)
    source_digest = hashlib.sha256()
    for source_name in (
        "product_backtest.py",
        "risk.py",
        "strategy.py",
        "low_risk.py",
        "etf_mapping.py",
        "refresh_data.py",
    ):
        source_path = Path(__file__).with_name(source_name)
        source_digest.update(source_name.encode("utf-8"))
        source_digest.update(source_path.read_bytes())
    payload = {
        "status": "pass" if accepted else "fail",
        "accepted": accepted,
        "selection_end": VAL_END,
        "observation_used_for_selection": False,
        "policy_name": policy_name,
        "policy_signature": hashlib.sha256(policy_payload.encode("utf-8")).hexdigest(),
        "regime_policy_signature": hashlib.sha256(
            json.dumps(asdict(RegimePolicy()), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "low_risk_data_signature": low_risk_data_signature(),
        "decision_source_signature": source_digest.hexdigest(),
        "score_name": score_name,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "feature_cache_signature": current_feature_cache_signature(),
        "gates": gates,
    }
    (output_dir / "ACCEPTANCE_GATE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(gates).to_csv(
        output_dir / "ACCEPTANCE_GATE.csv", index=False, encoding="utf-8-sig"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", help="可选人工公式评分；不提供时使用已冻结滚动LightGBM")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--rolling-lgbm-horizon", type=int, choices=[5, 10])
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--use-selected-adaptive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cost-sensitivity", action="store_true")
    parser.add_argument("--cost-worker", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cost-worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--precomputed-cost-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--boundary-sensitivity", action="store_true")
    parser.add_argument(
        "--policy",
        choices=sorted(strategy_policy_presets()),
        default="simple_v1",
    )
    args = parser.parse_args()
    if args.rolling_lgbm_horizon and (args.use_selected_adaptive or args.score):
        parser.error("滚动模型重训不能同时指定其他评分来源")
    if args.score and args.use_selected_adaptive:
        parser.error("人工公式评分和已选滚动评分不能同时使用")
    if (args.cost_worker is None) != (args.cost_worker_output is None):
        parser.error("成本隔离进程参数必须成对提供")
    if args.cost_worker is not None and args.cost_sensitivity:
        parser.error("成本隔离进程不能再次启动成本压力")
    use_selected_adaptive = bool(
        args.use_selected_adaptive or (args.score is None and args.rolling_lgbm_horizon is None)
    )
    precomputed_cost_frame = (
        pd.read_csv(args.precomputed_cost_file)
        if args.precomputed_cost_file is not None
        else None
    )
    if args.cost_sensitivity:
        # 每条成本路径使用独立进程，避免Pandas/Numpy长期重复回放产生内存碎片。
        isolated_frames = []
        worker_python = os.environ.get("GA_FACTOR_WORKER_PYTHON", sys.executable)
        with tempfile.TemporaryDirectory(prefix="sector-cost-") as temp_dir:
            temp_root = Path(temp_dir)
            for cost_bps in (10.0, 20.0, 30.0, 50.0):
                output_path = temp_root / f"cost_{int(cost_bps)}.csv"
                command = [
                    worker_python,
                    "-X",
                    "faulthandler",
                    "-m",
                    "ga_factor_mining.sector.rotation.product_backtest",
                    "--cost-worker",
                    str(cost_bps),
                    "--cost-worker-output",
                    str(output_path),
                    "--policy",
                    args.policy,
                ]
                if args.score:
                    command.extend(["--score", args.score])
                elif args.rolling_lgbm_horizon:
                    command.extend(["--rolling-lgbm-horizon", str(args.rolling_lgbm_horizon)])
                elif args.use_selected_adaptive:
                    command.append("--use-selected-adaptive")
                worker_env = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    worker_env[name] = "1"
                worker_env["PYTHONMALLOC"] = "malloc"
                source_root = str(Path(__file__).resolve().parents[3])
                worker_env["PYTHONPATH"] = os.pathsep.join(
                    value
                    for value in (source_root, worker_env.get("PYTHONPATH", ""))
                    if value
                )
                completed = None
                for attempt in range(1, 4):
                    output_path.unlink(missing_ok=True)
                    completed = subprocess.run(command, check=False, env=worker_env)
                    if completed.returncode == 0 and output_path.exists():
                        break
                    print(f"[cost] {cost_bps:.0f}bp 子进程异常，第{attempt}/3次")
                if completed is None or completed.returncode != 0 or not output_path.exists():
                    raise RuntimeError(f"{cost_bps:.0f}bp 成本隔离回放连续失败")
                isolated_frames.append(pd.read_csv(output_path))
                print(f"[cost] {cost_bps:.0f}bp 隔离回放完成")
            combined_path = temp_root / "cost_sensitivity.csv"
            pd.concat(isolated_frames, ignore_index=True).to_csv(
                combined_path, index=False, encoding="utf-8-sig"
            )
            final_command = [
                worker_python,
                "-X",
                "faulthandler",
                "-m",
                "ga_factor_mining.sector.rotation.product_backtest",
                "--precomputed-cost-file",
                str(combined_path),
                "--cost-bps",
                str(args.cost_bps),
                "--policy",
                args.policy,
            ]
            if args.score:
                final_command.extend(["--score", args.score])
            elif args.rolling_lgbm_horizon:
                final_command.extend(
                    ["--rolling-lgbm-horizon", str(args.rolling_lgbm_horizon)]
                )
            elif args.use_selected_adaptive:
                final_command.append("--use-selected-adaptive")
            final = None
            for attempt in range(1, 4):
                final = subprocess.run(final_command, check=False, env=worker_env)
                if final.returncode == 0:
                    break
                print(f"[product] 正式账本子进程异常，第{attempt}/3次")
            if final is None or final.returncode != 0:
                raise RuntimeError("正式账本隔离回放连续失败")
            print("[cost] 隔离成本压力与正式账本全部完成")
        return
    requested_score = args.score or "score_breakout"
    # 已有外部评分时只读取产品账本所需列；滚动模型重训仍需要完整特征面板。
    if args.rolling_lgbm_horizon:
        panel = load_or_build_features()
    else:
        panel = load_feature_subset(
            product_feature_columns(requested_score, external_score=use_selected_adaptive)
        )
    low_risk_frame = build_low_risk_return_frame(panel)
    score_name = requested_score
    if use_selected_adaptive:
        adaptation_dir = ensure_output_dir("sector", "adaptation")
        selected_path = adaptation_dir / "SELECTED_SCORES.parquet"
        selected_meta = json.loads((adaptation_dir / "SELECTED.json").read_text(encoding="utf-8"))
        if selected_meta.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
            raise RuntimeError("自适应评分缓存与当前特征协议不一致，请先重跑 adaptive_validation")
        if selected_meta.get("feature_cache_signature") != current_feature_cache_signature():
            raise RuntimeError("自适应评分缓存与当前特征数据不一致，请先重跑 adaptive_validation")
        predictions = pd.read_parquet(selected_path)
        score_columns = [column for column in predictions.columns if column not in {"ts_code", "trade_date"}]
        if len(score_columns) != 1:
            raise ValueError("SELECTED_SCORES.parquet 应只包含一个评分字段")
        score_name = score_columns[0]
        panel = panel.merge(predictions, on=["ts_code", "trade_date"], how="left")
        print(f"[adaptive] 使用2024-2025选择期确定的方案: {selected_meta['variant']}")
    elif args.rolling_lgbm_horizon:
        from .rolling_validation import fit_predict_lgbm

        horizon = args.rolling_lgbm_horizon
        score_path = OUT_DIR / f"rolling_lgbm_{horizon}d_scores.parquet"
        score_meta_path = OUT_DIR / f"rolling_lgbm_{horizon}d_scores.meta.json"
        score_meta = {}
        if score_meta_path.exists():
            try:
                score_meta = json.loads(score_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                score_meta = {}
        score_cache_current = (
            score_path.exists()
            and score_meta.get("feature_protocol_version") == FEATURE_PROTOCOL_VERSION
            and score_meta.get("fold_years") == [2024, 2025, 2026]
        )
        if score_cache_current and not args.refresh_scores:
            predictions = pd.read_parquet(score_path)
            print(f"[rolling-lgbm] 读取评分缓存: {score_path}")
        else:
            predictions = fit_predict_lgbm(panel, horizon, [2024, 2025, 2026])
            predictions.to_parquet(score_path, index=False)
            score_meta_path.write_text(
                json.dumps(
                    {
                        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                        "feature_cache_signature": current_feature_cache_signature(),
                        "horizon": horizon,
                        "fold_years": [2024, 2025, 2026],
                        "training": "expanding window, labels matured before each year",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        panel = panel.merge(predictions, on=["ts_code", "trade_date"], how="left")
        score_name = f"score_rolling_lgbm_{horizon}d"
    selected_policy = get_strategy_policy(args.policy)
    if args.cost_worker is not None:
        worker_daily, worker_actions, _ = run_product_backtest(
            panel,
            score_name,
            PRODUCT_HISTORY_START,
            OBSERVATION_END,
            cost_bps=args.cost_worker,
            strategy_policy=selected_policy,
            low_risk_frame=low_risk_frame,
        )
        worker_frame = build_cost_sensitivity_frame(
            worker_daily,
            worker_actions,
            cost_bps=args.cost_worker,
            score_name=score_name,
            policy_name=args.policy,
        )
        args.cost_worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_frame.to_csv(args.cost_worker_output, index=False, encoding="utf-8-sig")
        print(f"[cost-worker] {args.cost_worker:.0f}bp 完成")
        return

    output_dir = ensure_output_dir("sector", "strategy")
    (output_dir / "POLICY.json").write_text(
        json.dumps(
            {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "strategy_policy_version": STRATEGY_POLICY_VERSION,
                "policy_name": args.policy,
                "policy": asdict(selected_policy),
                "selection_period": "2018-2023 development and 2024-2025 model/policy selection",
                "independent_out_of_sample_available": False,
                "period_roles": {
                    "2018-2023": "development",
                    "2024-2025": "model_and_policy_selection",
                    "2026": "observed_diagnostic_only",
                    "future_unseen_data": "independent_out_of_sample",
                },
                "selection_rule": "frozen rolling-LightGBM product baseline; rejected challengers do not alter the default run",
                "parameter_search_performed": True,
                "observation_used_for_selection": False,
                "observation_hypothesis_disclosure": "2026 market structure motivated the review; 2026 metrics are diagnostic, not confirmatory evidence",
                "boundary_mode": "continuous_carry_from_2018",
                "low_risk_code": DEFAULT_LOW_RISK_CODE,
                "low_risk_data_signature": low_risk_data_signature(),
                "starting_capital_assumption": "fully invested in low-risk ETF before first signal",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summaries = []
    annual_rows = []
    market_data_end_date = str(panel["trade_date"].dropna().max())
    from .refresh_data import next_trade_date

    latest_plan: dict = {}
    continuous_daily, continuous_actions, _ = run_product_backtest(
        panel,
        score_name,
        PRODUCT_HISTORY_START,
        OBSERVATION_END,
        cost_bps=args.cost_bps,
        strategy_policy=selected_policy,
        low_risk_frame=low_risk_frame,
        latest_plan_sink=latest_plan,
        planned_execution_date=next_trade_date(market_data_end_date),
    )
    continuous_daily.to_parquet(output_dir / "HISTORY_DAILY.parquet", index=False)
    continuous_actions.to_parquet(output_dir / "HISTORY_ACTIONS.parquet", index=False)
    for period, start, end in (
        ("development", PRODUCT_HISTORY_START, TRAIN_END),
        ("selection", VAL_START, VAL_END),
        ("observation", OBSERVATION_START, OBSERVATION_END),
    ):
        daily, actions, metrics = summarize_backtest_period(
            continuous_daily, continuous_actions, start, end
        )
        if period != "development":
            daily.to_parquet(output_dir / f"{period}_daily.parquet", index=False)
            actions.to_parquet(output_dir / f"{period}_actions.parquet", index=False)
        summaries.append(
            {
                "period": period,
                "boundary_mode": "continuous_carry",
                "score_name": score_name,
                "policy_name": args.policy,
                "cost_bps": args.cost_bps,
                **metrics,
            }
        )
        if not daily.empty:
            for year, year_daily in daily.groupby(daily["date"].str[:4]):
                year_metrics = annualized_metrics(year_daily.set_index("date")["net_return"])
                annual_rows.append(
                    {
                        "period": period,
                        "year": int(year),
                        "score_name": score_name,
                        "policy_name": args.policy,
                        "cost_bps": args.cost_bps,
                        **year_metrics,
                        "avg_turnover": float(year_daily["turnover"].mean()),
                    }
                )
        gc.collect()
    pd.DataFrame(summaries).to_csv(output_dir / "SUMMARY.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(output_dir / "ANNUAL_RESULTS.csv", index=False, encoding="utf-8-sig")
    advice_status = write_latest_advice(
        continuous_daily,
        continuous_actions,
        output_dir,
        market_data_end_date=market_data_end_date,
        latest_plan=latest_plan or None,
    )
    from .etf_mapping import write_latest_execution_readiness

    etf_execution_readiness = write_latest_execution_readiness(
        output_dir,
        ensure_output_dir("sector", "etf_mapping"),
    )
    risk_snapshot = latest_market_risk_snapshot(
        panel,
        continuous_daily,
        latest_plan=latest_plan or None,
    )
    write_latest_market_risk(
        risk_snapshot,
        output_dir,
        data_freshness=advice_status,
    )
    append_latest_signal_strength(
        panel,
        score_name,
        selected_policy,
        continuous_daily,
        risk_snapshot,
        output_dir,
    )
    cost_sensitivity_frame: pd.DataFrame | None = None
    if precomputed_cost_frame is not None:
        cost_sensitivity_frame = precomputed_cost_frame
        cost_sensitivity_frame.to_csv(
            output_dir / "COST_SENSITIVITY.csv", index=False, encoding="utf-8-sig"
        )

    write_acceptance_gate(
        continuous_daily,
        continuous_actions,
        output_dir,
        policy_name=args.policy,
        policy=selected_policy,
        score_name=score_name,
        cost_sensitivity=cost_sensitivity_frame,
    )

    if args.boundary_sensitivity:
        # 独立重置只用于研究边界，不进入默认产品运行。
        reset_daily, reset_actions, _ = run_product_backtest(
            panel,
            score_name,
            OBSERVATION_START,
            OBSERVATION_END,
            cost_bps=args.cost_bps,
            strategy_policy=selected_policy,
            low_risk_frame=low_risk_frame,
        )
        _, _, reset_metrics = summarize_backtest_period(
            reset_daily, reset_actions, OBSERVATION_START, OBSERVATION_END
        )
        continuous_observation = next(row for row in summaries if row["period"] == "observation")
        pd.DataFrame(
            [
                {
                    "period": "observation",
                    "boundary_mode": "continuous_carry",
                    **{
                        key: value
                        for key, value in continuous_observation.items()
                        if key not in {"period", "boundary_mode"}
                    },
                },
                {
                    "period": "observation",
                    "boundary_mode": "standalone_reset",
                    "score_name": score_name,
                    "policy_name": args.policy,
                    "cost_bps": args.cost_bps,
                    **reset_metrics,
                },
            ]
        ).to_csv(output_dir / "BOUNDARY_SENSITIVITY.csv", index=False, encoding="utf-8-sig")
    (output_dir / "RUN.json").write_text(
        json.dumps(
            {
                "default_prototype": use_selected_adaptive and args.policy == "simple_v1",
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_cache_signature": current_feature_cache_signature(),
                "strategy_policy_version": STRATEGY_POLICY_VERSION,
                "score_name": score_name,
                "policy_name": args.policy,
                "cost_bps": args.cost_bps,
                "cost_sensitivity_run": cost_sensitivity_frame is not None,
                "boundary_sensitivity_run": args.boundary_sensitivity,
                "product_history_start": PRODUCT_HISTORY_START,
                "boundary_mode": "continuous_carry_from_2018",
                "independent_out_of_sample_available": False,
                "period_roles": {
                    "development": "2018-2023",
                    "selection": "2024-2025",
                    "observation": "2026_seen_but_not_selectable",
                },
                "memory_mode": "projected feature columns; one default product replay",
                "data_freshness": advice_status,
                "etf_execution_readiness": etf_execution_readiness,
                "runtime": {
                    "python": platform.python_version(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    from .forward_monitor import record_forward_snapshot

    forward_status = record_forward_snapshot(output_dir)
    if forward_status["status"] == "protocol_mismatch":
        print("[forward] 冻结协议已变化，本次没有续接旧前向证据")
    if advice_status["data_stale"]:
        print(
            "[stale] 数据截止"
            f"{advice_status['data_end_date']}，已过期{advice_status['data_age_days']}天；"
            "本次仅完成历史复现，LATEST_ACTIONS.csv为空，禁止执行。"
        )
    else:
        print(
            f"[advice] {advice_status['strategy_action']}；"
            f"{advice_status['execution_message']}"
        )
    print(f"[done] {output_dir / 'SUMMARY.csv'}")


if __name__ == "__main__":
    main()
