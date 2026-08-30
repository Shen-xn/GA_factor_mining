import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.adaptive_validation import passes_frequency_gate
from ga_factor_mining.sector.rotation.refresh_data import _prediction_windows
from ga_factor_mining.sector.rotation.rolling_validation import (
    recency_sample_weights,
    periodic_prediction_bounds,
    training_window_mask,
)


class AdaptiveTrainingTests(unittest.TestCase):
    def test_recent_window_still_requires_mature_labels(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20181231", "20190102", "20231228"],
                "future_ret_5d_end_date": ["20190108", "20190109", "20240105"],
                "future_ret_5d_rank": [0.1, 0.2, 0.3],
            }
        )
        mask = training_window_mask(frame, "20231231", 5, 5)
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_recent_observation_has_larger_weight(self):
        dates = pd.Series(["20200101", "20230101"])
        weights = recency_sample_weights(dates, "20231231", 2.0)
        self.assertGreater(weights[1], weights[0])

    def test_quarterly_prediction_bounds_do_not_overlap(self):
        bounds = periodic_prediction_bounds("20240101", "20240531", 3)
        self.assertEqual(
            bounds,
            [
                ("20231231", "20240101", "20240331"),
                ("20240331", "20240401", "20240531"),
            ],
        )

    def test_frequency_gate_requires_real_product_improvement(self):
        baseline = {
            "total_ret": 0.20,
            "sharpe": 1.0,
            "max_drawdown": -0.10,
            "avg_turnover": 0.08,
        }
        candidate = {
            "total_ret": 0.23,
            "sharpe": 1.01,
            "max_drawdown": -0.11,
            "avg_turnover": 0.085,
        }
        self.assertTrue(
            passes_frequency_gate(baseline, candidate, {2024: 0.1, 2025: 0.1})
        )
        candidate["total_ret"] = 0.21
        self.assertFalse(
            passes_frequency_gate(baseline, candidate, {2024: 0.1, 2025: 0.1})
        )

    def test_incremental_quarterly_windows_cross_boundary_without_leakage(self):
        self.assertEqual(
            _prediction_windows("20260929", "20261002", 3),
            [
                ("20260630", "20260930", "20260930"),
                ("20260930", "20261001", "20261002"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
