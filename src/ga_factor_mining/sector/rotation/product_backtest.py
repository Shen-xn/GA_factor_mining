#!/usr/bin/env python3
"""把每日板块评分转换为带现金、成本和风险约束的产品策略。"""

from __future__ import annotations

import argparse
import gc
import json
import platform
from dataclasses import asdict, replace

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
            "strategy_action": "无可用数据",
            "execution_message": "无可用数据，禁止执行",
        }

    if sector_name_lookup is None:
        names = pd.read_parquet(DATA_DIR / "ths_index.parquet", columns=["ts_code", "name"])
        name_lookup = names.drop_duplicates("ts_code").set_index("ts_code")["name"].to_dict()
    else:
        name_lookup = dict(sector_name_lookup)
    latest_strategy_date = str(daily["signal_date"].dropna().max())
    latest_daily = daily.sort_values(["signal_date", "date"]).iloc[-1]
    data_end_date = str(daily["date"].dropna().max())
    reference_date = pd.Timestamp(asof_date or pd.Timestamp.today()).normalize()
    data_date = pd.to_datetime(data_end_date, format="%Y%m%d").normalize()
    data_age_days = int((reference_date - data_date).days)
    data_stale = data_age_days > 7
    if actions.empty:
        current_actions = pd.DataFrame()
        last_rebalance_actions = pd.DataFrame()
        last_order_date = ""
        target_state: dict[str, tuple[float, str | None]] = {}
    else:
        ordered = actions.sort_values(["signal_date", "execution_date", "ts_code"])
        last_order_date = str(ordered["signal_date"].max())
        current_actions = ordered.loc[ordered["signal_date"].eq(latest_strategy_date)].copy()
        last_rebalance_actions = ordered.loc[ordered["signal_date"].eq(last_order_date)].copy()
        target_state = {}
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
        current_actions = current_actions.loc[current_actions["weight_change"].abs().ge(0.0005)]
        last_rebalance_actions = last_rebalance_actions.loc[
            last_rebalance_actions["weight_change"].abs().ge(0.0005)
        ]

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

    strategy_action = "有新调仓建议" if not current_actions.empty else "持有不动"
    actionable_actions = current_actions if not data_stale else current_actions.iloc[0:0]
    execution_message = (
        "数据已过期，禁止执行"
        if data_stale
        else "按LATEST_ACTIONS执行"
        if not actionable_actions.empty
        else "无需交易"
    )
    readable_action_frame(actionable_actions).to_csv(
        output_dir / "LATEST_ACTIONS.csv", index=False, encoding="utf-8-sig"
    )
    readable_action_frame(last_rebalance_actions).to_csv(
        output_dir / "LAST_REBALANCE_ACTIONS.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        [
            {
                "数据截止日": data_end_date,
                "数据年龄(天)": data_age_days,
                "数据状态": "已过期" if data_stale else "可用",
                "策略日期": latest_strategy_date,
                "最近调仓信号日": last_order_date,
                "策略动作": strategy_action,
                "执行提示": execution_message,
                "市场状态": latest_daily["regime"],
                "当前板块仓位": f"{float(latest_daily['exposure']):.2%}",
                "当前低风险仓位": f"{float(latest_daily['low_risk_weight']):.2%}",
                "当前持仓数量": int(latest_daily["position_count"]),
            }
        ],
        columns=status_columns,
    ).to_csv(output_dir / "LATEST_STATUS.csv", index=False, encoding="utf-8-sig")

    portfolio_rows = []
    for code, (weight, etf_code) in sorted(target_state.items(), key=lambda item: item[1][0], reverse=True):
        portfolio_rows.append(
            {
                "策略日期": latest_strategy_date,
                "目标形成日期": last_order_date,
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
        "strategy_action": strategy_action,
        "execution_message": execution_message,
    }


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
    score_dates = sub.loc[
        (sub["trade_date"] >= start)
        & (sub["trade_date"] <= end)
        & sub["next_open_date"].notna()
        & sub["return_end_date"].notna(),
        "trade_date",
    ].drop_duplicates().tolist()
    valid_end_dates = set(sub.loc[sub["return_end_date"].le(end), "trade_date"])
    score_dates = [date for date in score_dates if date in valid_end_dates]
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

    def decide(signal_date: str) -> tuple[dict[str, float], str, float, pd.DataFrame]:
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
            if missing_mark.any():
                missing_codes = missing_mark[missing_mark].index.tolist()
                raise RuntimeError(f"{signal_date} 持仓缺少开盘或收盘估值: {missing_codes}")
            intraday_returns = signal_close / signal_open - 1.0
            marked_equity *= 1.0 + float(
                sum(live_weights[code] * intraday_returns.loc[code] for code in live_weights)
                + low_risk_weight * low_risk_intraday
            )
            for code in live_weights:
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
        if use_market_regime and use_continuous_defensive_exposure:
            market_exposure = technical_regime_exposure(
                regime_state.current, market_row, regime_policy
            )
        elif use_market_regime:
            market_exposure = regime_exposure(regime_state.current, regime_policy)
        else:
            market_exposure = 1.0
        drawdown_cap = drawdown_state.exposure_cap if use_drawdown_cap else 1.0
        exposure = min(market_exposure, drawdown_cap)
        target_positions = (
            REGIME_POSITION_LIMITS[regime_state.current]
            if use_market_regime and exposure > 0
            else base_policy.target_positions if exposure > 0 else 0
        )
        policy = replace(base_policy, target_positions=target_positions)
        columns = [
            "ts_code", product_score, "ret_5d_rank", "ret_20d",
            "volatility_20d", "volatility_20d_rank",
        ]
        day_source = daily_groups.get_group(signal_date)
        available = [column for column in columns if column in day_source.columns]
        daily = day_source[available].copy().rename(columns={product_score: "score"})
        daily = daily.dropna(subset=["score"]).copy()
        execution_date = str(date_map.loc[signal_date, "next_open_date"])
        valuation_date = str(date_map.loc[signal_date, "return_end_date"])
        daily["execution_allowed"] = daily["ts_code"].map(open_pivot.loc[execution_date].notna())
        daily["valuation_available"] = daily["ts_code"].map(open_pivot.loc[valuation_date].notna())
        daily["position_return"] = daily["ts_code"].map(position_returns)
        daily["position_drawdown"] = daily["ts_code"].map(position_drawdowns)
        positions, targets, decisions = step_portfolio(signal_date, daily, positions, exposure, policy)
        return _target_dict(targets), regime_state.current, exposure, decisions

    first_signal = score_dates[0]
    first_target, first_regime, first_exposure, first_decisions = decide(first_signal)
    initial_turnover = _turnover({}, first_target)
    initial_net = (1.0 - initial_turnover * cost_rate) - 1.0
    equity *= 1.0 + initial_net
    peak = max(peak, equity)
    live_weights = first_target
    first_execution_date = str(date_map.loc[first_signal, "next_open_date"])
    first_execution_prices = open_pivot.loc[first_execution_date].reindex(live_weights)
    if first_execution_prices.isna().any():
        raise RuntimeError("首次建仓仍包含缺失执行开盘价")
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
            "exposure": first_exposure,
            "low_risk_code": low_risk_code,
            "low_risk_weight": 1.0 - sum(live_weights.values()),
            "low_risk_return": 0.0,
            "position_count": len(live_weights),
        }
    )

    for index in range(1, len(score_dates)):
        signal_date = score_dates[index]
        desired_target, regime, exposure, decisions = decide(signal_date)
        prior_signal = score_dates[index - 1]
        asset_returns = return_pivot.loc[prior_signal].reindex(live_weights)
        if asset_returns.isna().any():
            missing_codes = asset_returns[asset_returns.isna()].index.tolist()
            raise RuntimeError(
                f"{prior_signal} 至 {date_map.loc[prior_signal, 'return_end_date']} "
                f"持仓缺少开盘收益: {missing_codes}"
            )
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
        turnover = _turnover(pretrade, target)
        net_return = (1.0 + gross_return) * (1.0 - turnover * cost_rate) - 1.0
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        risk_peak = max(risk_peak, equity)
        prior_codes = set(live_weights)
        live_weights = target
        execution_date = str(date_map.loc[signal_date, "next_open_date"])
        execution_prices = open_pivot.loc[execution_date].reindex(live_weights)
        if execution_prices.isna().any():
            missing_codes = execution_prices[execution_prices.isna()].index.tolist()
            raise RuntimeError(f"{execution_date} 目标持仓缺少执行开盘价: {missing_codes}")
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
                "low_risk_code": low_risk_code,
                "low_risk_weight": 1.0 - sum(live_weights.values()),
                "low_risk_return": period_low_risk_return,
                "position_count": len(live_weights),
            }
        )

    # 最后一个已执行目标仍持有到下一开盘，不凭空截掉尾部收益。
    last_signal = score_dates[-1]
    last_returns = return_pivot.loc[last_signal].reindex(live_weights)
    if last_returns.isna().any():
        missing_codes = last_returns[last_returns.isna()].index.tolist()
        raise RuntimeError(f"最后持有期缺少开盘收益: {missing_codes}")
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
            "date": str(date_map.loc[last_signal, "return_end_date"]),
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
            "low_risk_code": low_risk_code,
            "low_risk_weight": final_low_risk_weight,
            "low_risk_return": final_low_risk_return,
            "position_count": len(live_weights),
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
        }
    )
    return period_daily, period_actions, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", help="可选人工公式评分；不提供时使用已冻结滚动LightGBM")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--rolling-lgbm-horizon", type=int, choices=[5, 10])
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--use-selected-adaptive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cost-sensitivity", action="store_true")
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
    use_selected_adaptive = bool(
        args.use_selected_adaptive or (args.score is None and args.rolling_lgbm_horizon is None)
    )
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
    output_dir = ensure_output_dir("sector", "strategy")
    selected_policy = get_strategy_policy(args.policy)
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
    continuous_daily, continuous_actions, _ = run_product_backtest(
        panel,
        score_name,
        PRODUCT_HISTORY_START,
        OBSERVATION_END,
        cost_bps=args.cost_bps,
        strategy_policy=selected_policy,
        low_risk_frame=low_risk_frame,
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
    advice_status = write_latest_advice(continuous_daily, continuous_actions, output_dir)
    if args.cost_sensitivity:
        sensitivity_rows = []
        for cost_bps in [10.0, 20.0, 30.0]:
            if cost_bps == args.cost_bps:
                cost_daily, cost_actions = continuous_daily, continuous_actions
            else:
                cost_daily, cost_actions, _ = run_product_backtest(
                    panel,
                    score_name,
                    PRODUCT_HISTORY_START,
                    OBSERVATION_END,
                    cost_bps=cost_bps,
                    strategy_policy=selected_policy,
                    low_risk_frame=low_risk_frame,
                )
            for period, start, end in (
                ("development", PRODUCT_HISTORY_START, TRAIN_END),
                ("selection", VAL_START, VAL_END),
                ("observation", OBSERVATION_START, OBSERVATION_END),
            ):
                daily, _, metrics = summarize_backtest_period(cost_daily, cost_actions, start, end)
                sensitivity_rows.append(
                    {
                        "period": period,
                        "boundary_mode": "continuous_carry",
                        "score_name": score_name,
                        "policy_name": args.policy,
                        "cost_bps": cost_bps,
                        "full_path_rerun": True,
                        **metrics,
                        "avg_turnover": float(daily["turnover"].mean()),
                    }
                )
            gc.collect()
        pd.DataFrame(sensitivity_rows).to_csv(
            output_dir / "COST_SENSITIVITY.csv", index=False, encoding="utf-8-sig"
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
                "cost_sensitivity_run": args.cost_sensitivity,
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
