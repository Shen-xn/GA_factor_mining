import unittest

from ga_factor_mining.sector.rotation.defensive_exposure_validation import (
    development_gate_failures,
    selection_gate_failures,
)


class DefensiveExposureValidationTests(unittest.TestCase):
    def test_development_gate_requires_worst_year_improvement(self):
        baseline = {"total_ret": 0.10, "max_drawdown": -0.18, "avg_turnover": 0.08}
        candidate = {"total_ret": 0.12, "max_drawdown": -0.18, "avg_turnover": 0.08}
        baseline_years = {
            2018: -0.12,
            2019: 0.05,
            2020: 0.04,
            2021: 0.10,
            2022: -0.08,
            2023: 0.01,
        }
        candidate_years = {**baseline_years, 2018: -0.10}
        failures = development_gate_failures(
            baseline, candidate, baseline_years, candidate_years
        )
        self.assertIn(
            "worst year did not improve by at least four percentage points", failures
        )

    def test_selection_gate_accepts_retained_positive_candidate(self):
        baseline = {"total_ret": 0.30, "max_drawdown": -0.10}
        candidate = {"total_ret": 0.25, "max_drawdown": -0.11, "avg_turnover": 0.09}
        self.assertEqual(
            selection_gate_failures(
                baseline, candidate, {2024: 0.15, 2025: 0.09}, 0.18
            ),
            [],
        )

    def test_selection_gate_rejects_negative_cost_stress(self):
        failures = selection_gate_failures(
            {"total_ret": 0.30, "max_drawdown": -0.10},
            {"total_ret": 0.30, "max_drawdown": -0.10, "avg_turnover": 0.09},
            {2024: 0.15, 2025: 0.09},
            -0.01,
        )
        self.assertIn("30bp selection cumulative return was not positive", failures)


if __name__ == "__main__":
    unittest.main()
