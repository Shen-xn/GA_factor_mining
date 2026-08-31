import unittest

import numpy as np
import pandas as pd

from ga_factor_mining.sector.rotation.risk import (
    DrawdownState,
    RegimePolicy,
    RegimeState,
    advance_drawdown_state,
    advance_regime,
    build_market_state,
    classify_market,
    effective_exposure,
    leading_sector_strength,
    market_risk_components,
    technical_regime_exposure,
)
from ga_factor_mining.sector.rotation.reference_outputs import (
    build_broad_market_diagnostic_state,
)


class MarketRiskTests(unittest.TestCase):
    def test_broad_market_diagnostic_uses_indices_and_31_industries(self):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range("2023-01-02", periods=320).strftime("%Y%m%d")
        common = rng.normal(0.0003, 0.008, len(dates))
        index_frames = []
        for number in range(5):
            returns = common + rng.normal(0.0, 0.002, len(dates))
            index_frames.append(
                pd.DataFrame(
                    {
                        "ts_code": f"INDEX{number}",
                        "trade_date": dates,
                        "close": 100.0 * np.cumprod(1.0 + returns),
                    }
                )
            )
        industry_frames = []
        for number in range(31):
            returns = common + rng.normal(0.0, 0.006, len(dates))
            industry_frames.append(
                pd.DataFrame(
                    {
                        "ts_code": f"SW{number:02d}",
                        "trade_date": dates,
                        "close": 100.0 * np.cumprod(1.0 + returns),
                    }
                )
            )
        state = build_broad_market_diagnostic_state(
            pd.concat(index_frames, ignore_index=True),
            pd.concat(industry_frames, ignore_index=True),
        )
        latest = state.iloc[-1]
        self.assertEqual(latest["trade_date"], dates[-1])
        self.assertEqual(int(latest["sector_count"]), 31)
        self.assertEqual(latest["risk_data_quality"], "complete")
        self.assertIn(latest["raw_regime"], {"CASH", "DEFENSIVE", "NEUTRAL", "RISK_ON"})

    def test_market_risk_score_is_monotonic_and_bounded(self):
        weak = pd.Series(
            {
                "benchmark_trend_60d": -0.10,
                "market_volatility_20d": 0.02,
                "breadth_positive_20d": 0.25,
                "breadth_positive_60d": 0.30,
                "market_vol_percentile": 0.90,
            }
        )
        strong = weak.copy()
        strong["benchmark_trend_60d"] = 0.10
        strong["breadth_positive_20d"] = 0.70
        strong["breadth_positive_60d"] = 0.75
        strong["market_vol_percentile"] = 0.20
        weak_score = float(market_risk_components(weak)["risk_score"])
        strong_score = float(market_risk_components(strong)["risk_score"])
        self.assertTrue(0.0 <= weak_score < strong_score <= 100.0)

    def test_market_risk_missing_data_is_not_optimistic(self):
        result = market_risk_components(pd.Series(dtype=float))
        self.assertEqual(result["risk_data_quality"], "insufficient")
        self.assertLessEqual(float(result["risk_score"]), 50.0)

    def test_zero_volatility_is_not_complete_market_data(self):
        result = market_risk_components(
            pd.Series(
                {
                    "benchmark_trend_60d": 0.20,
                    "market_volatility_20d": 0.0,
                    "risk_breadth_positive_20d": 1.0,
                    "risk_breadth_positive_60d": 1.0,
                    "market_vol_percentile": 0.0,
                    "sector_count": 100,
                    "breadth_20d_coverage": 1.0,
                    "breadth_60d_coverage": 1.0,
                }
            )
        )
        self.assertEqual(result["risk_data_quality"], "insufficient")
        self.assertLessEqual(float(result["risk_score"]), 50.0)

    def test_risk_breadth_excludes_missing_values_from_denominator(self):
        panel = pd.DataFrame(
            {
                "trade_date": ["20240102", "20240102"],
                "ts_code": ["A", "B"],
                "type": ["I", "I"],
                "ret_1d": [0.01, -0.01],
                "ret_20d": [0.10, float("nan")],
                "ret_60d": [0.10, float("nan")],
            }
        )
        row = build_market_state(panel).iloc[0]
        self.assertEqual(row["breadth_positive_20d"], 0.5)
        self.assertEqual(row["risk_breadth_positive_20d"], 1.0)
        self.assertEqual(row["breadth_20d_coverage"], 0.5)

    def test_leading_sector_strength_requires_three_strong_top_five(self):
        frame = pd.DataFrame(
            {
                "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                "ret_20d": [0.1, 0.08, 0.03, -0.02, -0.03, 0.2],
                "ret_5d_rank": [0.9, 0.8, 0.6, 0.7, 0.4, 1.0],
            }
        )
        self.assertTrue(leading_sector_strength(frame, "score"))
        frame.loc[2, "ret_20d"] = -0.01
        self.assertFalse(leading_sector_strength(frame, "score"))

    def test_leading_sector_strength_does_not_use_assets_below_top_five(self):
        frame = pd.DataFrame(
            {
                "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                "ret_20d": [-0.1, -0.1, -0.1, -0.1, -0.1, 0.2],
                "ret_5d_rank": [0.9, 0.8, 0.7, 0.6, 0.5, 1.0],
            }
        )
        self.assertFalse(leading_sector_strength(frame, "score"))

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
