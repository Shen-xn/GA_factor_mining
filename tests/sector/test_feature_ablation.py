import unittest

from ga_factor_mining.sector.rotation.feature_ablation import (
    CANDIDATES,
    passes_development_gate,
    passes_selection_gate,
)


class FeatureAblationTests(unittest.TestCase):
    def test_only_three_predeclared_derived_candidates_are_tested(self):
        self.assertEqual(
            set(CANDIDATES.values()),
            {
                "risk_adj_5_20_rank",
                "risk_adj_10_20_rank",
                "risk_adj_20_60_rank",
            },
        )

    def test_development_gate_rejects_fewer_positive_years(self):
        baseline = {
            "total_ret": 0.10,
            "sharpe": 0.30,
            "max_drawdown": -0.18,
            "avg_turnover": 0.08,
        }
        candidate = {
            "total_ret": 0.10,
            "sharpe": 0.30,
            "max_drawdown": -0.18,
            "avg_turnover": 0.08,
        }
        baseline_years = {2018: -0.10, 2019: 0.05, 2020: 0.03, 2022: -0.05}
        candidate_years = {2018: -0.10, 2019: -0.01, 2020: 0.09, 2022: -0.05}
        self.assertFalse(
            passes_development_gate(
                baseline, candidate, baseline_years, candidate_years
            )
        )

    def test_selection_gate_requires_both_years_positive(self):
        baseline = {
            "total_ret": 0.30,
            "sharpe": 1.00,
            "max_drawdown": -0.10,
            "avg_turnover": 0.10,
        }
        candidate = {
            "total_ret": 0.29,
            "sharpe": 0.98,
            "max_drawdown": -0.11,
            "avg_turnover": 0.10,
        }
        self.assertFalse(
            passes_selection_gate(baseline, candidate, {2024: 0.31, 2025: -0.01})
        )
        self.assertTrue(
            passes_selection_gate(baseline, candidate, {2024: 0.20, 2025: 0.08})
        )


if __name__ == "__main__":
    unittest.main()
