import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.risk import (
    DrawdownState,
    RegimePolicy,
    RegimeState,
    advance_drawdown_state,
    advance_regime,
    classify_market,
    effective_exposure,
    technical_regime_exposure,
)


class MarketRiskTests(unittest.TestCase):
    def test_market_deterioration_requires_two_confirmations(self):
        state = RegimeState("RISK_ON")
        state = advance_regime(state, "DEFENSIVE")
        self.assertEqual(state.current, "RISK_ON")
        state = advance_regime(state, "DEFENSIVE")
        self.assertEqual(state.current, "DEFENSIVE")

    def test_market_improvement_is_gradual(self):
        state = RegimeState("CASH")
        for _ in range(5):
            state = advance_regime(state, "RISK_ON")
        self.assertEqual(state.current, "DEFENSIVE")

    def test_cash_regime_requires_trend_breadth_and_volatility(self):
        row = pd.Series(
            {
                "benchmark_trend_60d": -0.05,
                "breadth_positive_20d": 0.20,
                "breadth_positive_60d": 0.25,
                "market_vol_percentile": 0.95,
            }
        )
        self.assertEqual(classify_market(row), "CASH")

    def test_drawdown_cap_overrides_market_regime(self):
        self.assertEqual(effective_exposure("RISK_ON", -0.13), 0.3)
        self.assertEqual(effective_exposure("RISK_ON", -0.16), 0.0)

    def test_regime_policy_can_reduce_defensive_exposure(self):
        policy = RegimePolicy(defensive_exposure=0.15)
        self.assertEqual(effective_exposure("DEFENSIVE", 0.0, policy), 0.15)

    def test_defensive_exposure_changes_continuously_with_technical_health(self):
        middle = pd.Series(
            {"benchmark_trend_60d": -0.06, "breadth_positive_20d": 0.325}
        )
        self.assertAlmostEqual(technical_regime_exposure("DEFENSIVE", middle), 0.15)
        self.assertEqual(
            technical_regime_exposure(
                "DEFENSIVE",
                pd.Series({"benchmark_trend_60d": -0.12, "breadth_positive_20d": 0.45}),
            ),
            0.0,
        )
        self.assertEqual(
            technical_regime_exposure(
                "DEFENSIVE",
                pd.Series({"benchmark_trend_60d": 0.0, "breadth_positive_20d": 0.45}),
            ),
            0.3,
        )
        self.assertEqual(technical_regime_exposure("RISK_ON", middle), 1.0)

    def test_drawdown_cash_cooldown_can_restart_at_small_exposure(self):
        state = advance_drawdown_state(DrawdownState(), -0.16)
        self.assertEqual(state.exposure_cap, 0.0)
        for _ in range(10):
            state = advance_drawdown_state(state, -0.16)
        self.assertEqual(state.exposure_cap, 0.3)
        state = advance_drawdown_state(state, -0.16)
        self.assertEqual(state.exposure_cap, 0.3)

    def test_small_exposure_rearms_only_after_new_loss(self):
        state = DrawdownState(exposure_cap=0.3, trigger_drawdown=-0.16)
        self.assertEqual(advance_drawdown_state(state, -0.17).exposure_cap, 0.3)
        self.assertEqual(advance_drawdown_state(state, -0.19).exposure_cap, 0.0)

    def test_recovery_requires_five_confirmations(self):
        state = DrawdownState(exposure_cap=0.3)
        for _ in range(4):
            state = advance_drawdown_state(state, -0.10)
        self.assertEqual(state.exposure_cap, 0.3)
        state = advance_drawdown_state(state, -0.10)
        self.assertEqual(state.exposure_cap, 0.7)


if __name__ == "__main__":
    unittest.main()
