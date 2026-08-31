"""把每日板块评分转换为低频、可解释的持仓建议。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyPolicy:
    """简化执行层：评分决定候选，少量规则决定是否真的换仓。"""

    target_positions: int = 5
    entry_rank: int = 5
    retain_rank: int = 10
    min_hold_sessions: int = 5
    hard_drawdown: float = -0.12
    weak_momentum_rank: float = 0.20
    extreme_volatility_rank: float = 0.95
    score_smoothing_sessions: int = 3
    rebalance_tolerance: float = 0.03


@dataclass(frozen=True)
class PositionState:
    entry_date: str
    held_sessions: int = 0


STRATEGY_POLICY_VERSION = 5


def strategy_policy_presets() -> dict[str, StrategyPolicy]:
    """日常只保留一个策略，研究候选不进入默认入口。"""
    return {"simple_v1": StrategyPolicy()}


def get_strategy_policy(name: str) -> StrategyPolicy:
    """按稳定名称取得策略参数，输出文件只记录名称和完整快照。"""
    presets = strategy_policy_presets()
    if name not in presets:
        raise ValueError(f"未知策略参数组: {name}")
    return presets[name]


def _number(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    return float(value) if pd.notna(value) else default


def risk_exit_reason(row: pd.Series, policy: StrategyPolicy) -> str | None:
    """只处理足以覆盖最短持有期的硬风险，普通走弱交给保留区判断。"""
    position_return = _number(row, "position_return")
    position_drawdown = _number(row, "position_drawdown")
    volatility = _number(row, "volatility_20d")
    momentum = _number(row, "ret_5d_rank")
    volatility_rank = _number(row, "volatility_20d_rank")
    if pd.notna(position_return) and position_return <= policy.hard_drawdown:
        return "hard_position_loss"
    if pd.notna(position_drawdown) and pd.notna(volatility):
        dynamic_stop = max(0.06, min(0.12, 2.5 * volatility * sqrt(5.0)))
        if position_drawdown <= -dynamic_stop:
            return "trailing_stop"
    if (
        pd.notna(volatility_rank)
        and volatility_rank >= policy.extreme_volatility_rank
        and pd.notna(momentum)
        and momentum <= policy.weak_momentum_rank
    ):
        return "volatility_shock"
    return None


def step_portfolio(
    signal_date: str,
    daily_scores: pd.DataFrame,
    positions: dict[str, PositionState],
    market_exposure: float,
    policy: StrategyPolicy | None = None,
) -> tuple[dict[str, PositionState], pd.DataFrame, pd.DataFrame]:
    """推进一个信号日，返回新状态、目标权重和买卖建议。

    ``daily_scores`` 至少包含 ``ts_code`` 和 ``score``。保留区使用
    ``score_rank``；硬风险使用入场后的收益/回撤和20日波动率。
    """
    policy = policy or StrategyPolicy()
    required = {"ts_code", "score"}
    missing = required - set(daily_scores.columns)
    if missing:
        raise ValueError(f"daily_scores 缺少字段: {sorted(missing)}")
    if not 0.0 <= market_exposure <= 1.0:
        raise ValueError("market_exposure 必须在 0 到 1 之间")

    daily = daily_scores.drop_duplicates("ts_code", keep="last").copy()
    daily = daily.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    daily["score_rank"] = np.arange(1, len(daily) + 1)
    daily = daily.set_index("ts_code", drop=False)

    current = {
        code: PositionState(state.entry_date, state.held_sessions + 1)
        for code, state in positions.items()
    }
    decisions: list[dict] = []
    sold_today: set[str] = set()

    # 极端市场状态允许全部降到现金；无法在次日开盘成交时保留原仓，避免虚构交易。
    for code in list(current):
        if code not in daily.index:
            continue
        row = daily.loc[code]
        execution_allowed = bool(row.get("execution_allowed", True))
        if not execution_allowed:
            continue
        if not bool(row.get("valuation_available", True)):
            reason = "data_unavailable_next_valuation"
        elif market_exposure == 0.0:
            reason = "market_risk_off"
        else:
            reason = risk_exit_reason(row, policy)
        if reason:
            decisions.append({
                "signal_date": signal_date,
                "ts_code": code,
                "action": "sell",
                "reason": reason,
                "held_sessions": current[code].held_sessions,
            })
            sold_today.add(code)
            del current[code]

    # 市场降级时主动收缩持仓数量，不等待普通退出条件触发。
    if len(current) > policy.target_positions:
        executable_current = [
            code for code in current
            if code in daily.index and bool(daily.loc[code].get("execution_allowed", True))
        ]
        ranked_current = sorted(
            executable_current,
            key=lambda code: _number(daily.loc[code], "score", -np.inf) if code in daily.index else -np.inf,
        )
        for code in ranked_current[: max(0, len(current) - policy.target_positions)]:
            decisions.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "action": "sell",
                    "reason": "position_limit_reduction",
                    "held_sessions": current[code].held_sessions,
                }
            )
            sold_today.add(code)
            del current[code]

    # 至少持有5日；跌出Top10才退出，但不拖延已经走弱的持仓。
    for code in list(current):
        if code not in daily.index:
            # 当日无报价时只延续持仓，不能用未来可用性提前卖出。
            continue
        row = daily.loc[code]
        state = current[code]
        if not bool(row.get("execution_allowed", True)):
            continue
        rank = int(row["score_rank"])
        if state.held_sessions >= policy.min_hold_sessions and rank > policy.retain_rank:
            decisions.append({
                "signal_date": signal_date,
                "ts_code": code,
                "action": "sell",
                "reason": "left_retain_zone",
                "held_sessions": state.held_sessions,
            })
            sold_today.add(code)
            del current[code]

    execution_allowed = (
        daily["execution_allowed"].fillna(False).astype(bool)
        if "execution_allowed" in daily.columns
        else pd.Series(True, index=daily.index)
    )
    valuation_available = (
        daily["valuation_available"].fillna(False).astype(bool)
        if "valuation_available" in daily.columns
        else pd.Series(True, index=daily.index)
    )
    eligible_mask = (
        (daily["score_rank"] <= policy.entry_rank)
        & execution_allowed
        & valuation_available
        & (~daily["ts_code"].isin(current))
        & (~daily["ts_code"].isin(sold_today))
    )
    eligible = daily.loc[eligible_mask]

    # 只在出现空位时买入Top5，不为小幅分数变化主动替换现有持仓。
    # 这里只需要代码列；避免每日创建namedtuple类型，降低Windows长回放的原生堆压力。
    for code in eligible["ts_code"].astype(str).tolist():
        if len(current) >= policy.target_positions:
            break
        current[code] = PositionState(entry_date=signal_date)
        decisions.append({
            "signal_date": signal_date,
            "ts_code": code,
            "action": "buy",
            "reason": "vacancy_and_strong_signal",
            "held_sessions": 0,
        })

    weight = market_exposure / len(current) if current else 0.0
    targets = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "ts_code": code,
                "target_weight": weight,
                "held_sessions": state.held_sessions,
                "entry_date": state.entry_date,
            }
            for code, state in sorted(current.items())
        ]
    )
    return current, targets, pd.DataFrame(decisions)
