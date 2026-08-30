"""透明的市场状态和组合回撤仓位约束。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIME_ORDER = {"CASH": 0, "DEFENSIVE": 1, "NEUTRAL": 2, "RISK_ON": 3}
REGIME_EXPOSURE = {"CASH": 0.0, "DEFENSIVE": 0.3, "NEUTRAL": 0.7, "RISK_ON": 1.0}


@dataclass(frozen=True)
class RegimePolicy:
    risk_on_breadth_60d: float = 0.55
    defensive_breadth_60d: float = 0.45
    cash_breadth_20d: float = 0.30
    defensive_vol_percentile: float = 0.80
    cash_vol_percentile: float = 0.90
    worsen_confirm_sessions: int = 2
    improve_confirm_sessions: int = 5
    cash_exposure: float = 0.0
    defensive_exposure: float = 0.3
    neutral_exposure: float = 0.7
    risk_on_exposure: float = 1.0


@dataclass(frozen=True)
class RegimeState:
    current: str = "NEUTRAL"
    pending: str | None = None
    pending_sessions: int = 0


@dataclass(frozen=True)
class DrawdownState:
    """组合回撤保护状态，避免清仓后因净值不动而永久锁死。"""

    exposure_cap: float = 1.0
    cooldown_remaining: int = 0
    recovery_sessions: int = 0
    trigger_drawdown: float | None = None


def build_market_state(panel: pd.DataFrame, types: tuple[str, ...] = ("I", "N")) -> pd.DataFrame:
    """从板块横截面构造不依赖未来数据的市场状态表。"""
    sub = panel[panel["type"].isin(types)].copy()
    daily = sub.groupby("trade_date", sort=True).agg(
        benchmark_ret_1d=("ret_1d", "mean"),
        breadth_positive_20d=("ret_20d", lambda x: float((x > 0).mean())),
        breadth_positive_60d=("ret_60d", lambda x: float((x > 0).mean())),
        cross_section_dispersion_20d=("ret_20d", "std"),
    )
    daily["benchmark_equity"] = (1.0 + daily["benchmark_ret_1d"].fillna(0.0)).cumprod()
    daily["benchmark_trend_60d"] = daily["benchmark_equity"] / daily["benchmark_equity"].shift(60) - 1.0
    daily["market_volatility_20d"] = daily["benchmark_ret_1d"].rolling(20, min_periods=15).std()

    def last_percentile(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if not len(valid):
            return np.nan
        return float((valid <= valid[-1]).mean())

    daily["market_vol_percentile"] = daily["market_volatility_20d"].rolling(
        252,
        min_periods=60,
    ).apply(last_percentile, raw=True)
    return daily.reset_index()


def classify_market(row: pd.Series, policy: RegimePolicy | None = None) -> str:
    policy = policy or RegimePolicy()
    trend = float(row.get("benchmark_trend_60d", np.nan))
    breadth20 = float(row.get("breadth_positive_20d", np.nan))
    breadth60 = float(row.get("breadth_positive_60d", np.nan))
    vol_pct = float(row.get("market_vol_percentile", np.nan))
    if not all(np.isfinite([trend, breadth20, breadth60, vol_pct])):
        return "NEUTRAL"
    if trend < 0 and breadth20 < policy.cash_breadth_20d and vol_pct >= policy.cash_vol_percentile:
        return "CASH"
    defensive_votes = sum(
        [
            trend < 0,
            breadth60 < policy.defensive_breadth_60d,
            vol_pct >= policy.defensive_vol_percentile,
        ]
    )
    if defensive_votes >= 2:
        return "DEFENSIVE"
    if trend > 0 and breadth60 >= policy.risk_on_breadth_60d and vol_pct < policy.defensive_vol_percentile:
        return "RISK_ON"
    return "NEUTRAL"


def advance_regime(
    state: RegimeState,
    raw_regime: str,
    policy: RegimePolicy | None = None,
) -> RegimeState:
    """市场恶化快确认、改善慢确认，降低边界反复切换。"""
    policy = policy or RegimePolicy()
    if raw_regime not in REGIME_ORDER:
        raise ValueError(f"未知市场状态: {raw_regime}")
    if raw_regime == state.current:
        return RegimeState(current=state.current)
    pending_sessions = state.pending_sessions + 1 if state.pending == raw_regime else 1
    worsening = REGIME_ORDER[raw_regime] < REGIME_ORDER[state.current]
    required = policy.worsen_confirm_sessions if worsening else policy.improve_confirm_sessions
    if pending_sessions < required:
        return RegimeState(state.current, raw_regime, pending_sessions)
    if worsening:
        return RegimeState(current=raw_regime)
    next_level = min(REGIME_ORDER[state.current] + 1, REGIME_ORDER[raw_regime])
    next_regime = next(name for name, level in REGIME_ORDER.items() if level == next_level)
    return RegimeState(current=next_regime)


def drawdown_exposure_cap(drawdown: float) -> float:
    """在目标20%回撤之前主动降仓，阈值是产品约束而非收益保证。"""
    if drawdown <= -0.15:
        return 0.0
    if drawdown <= -0.12:
        return 0.3
    if drawdown <= -0.08:
        return 0.7
    return 1.0


def regime_exposure(regime: str, policy: RegimePolicy | None = None) -> float:
    policy = policy or RegimePolicy()
    return {
        "CASH": policy.cash_exposure,
        "DEFENSIVE": policy.defensive_exposure,
        "NEUTRAL": policy.neutral_exposure,
        "RISK_ON": policy.risk_on_exposure,
    }[regime]


def technical_regime_exposure(
    regime: str,
    market_row: pd.Series,
    policy: RegimePolicy | None = None,
) -> float:
    """只在DEFENSIVE内按趋势和宽度连续缩放0%-30%风险仓位。"""
    policy = policy or RegimePolicy()
    if regime != "DEFENSIVE":
        return regime_exposure(regime, policy)
    trend = float(market_row.get("benchmark_trend_60d", np.nan))
    breadth = float(market_row.get("breadth_positive_20d", np.nan))
    if not np.isfinite(trend) or not np.isfinite(breadth):
        return policy.defensive_exposure
    trend_strength = float(np.clip((trend + 0.12) / 0.12, 0.0, 1.0))
    breadth_strength = float(np.clip((breadth - 0.20) / 0.25, 0.0, 1.0))
    return policy.defensive_exposure * float(np.sqrt(trend_strength * breadth_strength))


def leading_sector_strength(
    daily_scores: pd.DataFrame,
    score_column: str,
    *,
    top_k: int = 5,
    minimum_strong: int = 3,
) -> bool:
    """判断领先板块是否自身仍保持正趋势，仅使用当日及历史信息。"""
    required = {score_column, "ret_20d", "ret_5d_rank"}
    if required - set(daily_scores.columns):
        return False
    leaders = daily_scores.dropna(subset=[score_column]).nlargest(top_k, score_column)
    if len(leaders) < top_k:
        return False
    strong = leaders["ret_20d"].gt(0.0) & leaders["ret_5d_rank"].ge(0.5)
    return int(strong.sum()) >= minimum_strong


def effective_exposure(
    regime: str,
    drawdown: float,
    policy: RegimePolicy | None = None,
) -> float:
    return min(regime_exposure(regime, policy), drawdown_exposure_cap(drawdown))


def advance_drawdown_state(state: DrawdownState, drawdown: float) -> DrawdownState:
    """回撤恶化立即降仓，清仓冷静期结束后分级恢复。"""
    if state.exposure_cap == 0.0:
        if state.cooldown_remaining > 1:
            return DrawdownState(0.0, state.cooldown_remaining - 1, 0, state.trigger_drawdown)
        return DrawdownState(0.3, 0, 0, state.trigger_drawdown)

    # 试探仓只对触发后的新增损失再次清仓，避免同一历史回撤永久锁死。
    if state.exposure_cap <= 0.3 and state.trigger_drawdown is not None:
        if drawdown < state.trigger_drawdown - 0.02:
            return DrawdownState(0.0, 20, 0, drawdown)
    if state.exposure_cap > 0.3 and drawdown <= -0.15:
        return DrawdownState(0.0, 10, 0, drawdown)
    if state.exposure_cap > 0.7 and drawdown <= -0.12:
        return DrawdownState(0.3, 0, 0, drawdown)
    if state.exposure_cap > 0.3 and drawdown <= -0.08:
        return DrawdownState(0.7, 0, 0, drawdown)

    default_threshold = -0.12 if state.exposure_cap <= 0.3 else -0.08
    improvement_threshold = (
        max(default_threshold, state.trigger_drawdown + 0.03)
        if state.trigger_drawdown is not None
        else default_threshold
    )
    if state.exposure_cap < 1.0 and drawdown > improvement_threshold:
        recovery_sessions = state.recovery_sessions + 1
        if recovery_sessions >= 5:
            next_cap = 0.7 if state.exposure_cap <= 0.3 else 1.0
            next_trigger = None if next_cap == 1.0 else state.trigger_drawdown
            return DrawdownState(next_cap, 0, 0, next_trigger)
        return DrawdownState(state.exposure_cap, 0, recovery_sessions, state.trigger_drawdown)
    return DrawdownState(state.exposure_cap, 0, 0, state.trigger_drawdown)
