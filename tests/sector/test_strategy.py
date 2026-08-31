import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.strategy import (
    PositionState,
    StrategyPolicy,
    get_strategy_policy,
    step_portfolio,
)


class StatefulStrategyTests(unittest.TestCase):
    def test_default_policy_preserves_v1_and_promotes_v2(self):
        baseline = get_strategy_policy("simple_v1")
        promoted = get_strategy_policy("simple_v2")
        self.assertEqual(baseline.retain_rank, 10)
        self.assertEqual(baseline.min_hold_sessions, 5)
        self.assertEqual(promoted.target_positions, 5)
        self.assertEqual(promoted.entry_rank, 5)
        self.assertEqual(promoted.retain_rank, 20)
        self.assertEqual(promoted.min_hold_sessions, 10)
        self.assertEqual(promoted.risk_reference_cost_bps, 20.0)

    def test_small_score_change_does_not_force_replacement(self):
        policy = StrategyPolicy(target_positions=2, entry_rank=2, retain_rank=4, min_hold_sessions=3)
        positions = {
            "A": PositionState("20240101", 3),
            "B": PositionState("20240101", 3),
        }
        daily = pd.DataFrame({"ts_code": ["C", "A", "B"], "score": [0.81, 0.80, 0.79]})
        new_positions, _, decisions = step_portfolio("20240105", daily, positions, 1.0, policy)
        self.assertEqual(set(new_positions), {"A", "B"})
        self.assertTrue(decisions.empty)

    def test_hard_risk_exit_ignores_minimum_holding_period(self):
        positions = {"A": PositionState("20240104", 0)}
        daily = pd.DataFrame(
            {
                "ts_code": ["B", "A"],
                "score": [0.8, 0.7],
                "position_return": [0.0, -0.15],
                "position_drawdown": [0.0, -0.15],
                "volatility_20d": [0.02, 0.02],
            }
        )
        new_positions, _, decisions = step_portfolio("20240105", daily, positions, 1.0)
        self.assertNotIn("A", new_positions)
        self.assertIn("hard_position_loss", decisions["reason"].tolist())

    def test_volatility_shock_branch_is_executable(self):
        positions = {"A": PositionState("20240104", 0)}
        daily = pd.DataFrame(
            {
                "ts_code": ["B", "A"],
                "score": [0.8, 0.7],
                "ret_5d_rank": [0.8, 0.2],
                "volatility_20d_rank": [0.2, 0.99],
            }
        )
        new_positions, _, decisions = step_portfolio("20240105", daily, positions, 1.0)
        self.assertNotIn("A", new_positions)
        self.assertIn("volatility_shock", decisions["reason"].tolist())

    def test_market_exposure_controls_total_weight(self):
        daily = pd.DataFrame({"ts_code": ["A", "B", "C", "D", "E"], "score": [0.9, 0.8, 0.7, 0.6, 0.5]})
        _, targets, _ = step_portfolio("20240105", daily, {}, 0.3)
        self.assertAlmostEqual(targets["target_weight"].sum(), 0.3)

    def test_reduced_position_limit_sells_the_weakest_holding(self):
        policy = StrategyPolicy(target_positions=2, entry_rank=2, retain_rank=5)
        positions = {
            "A": PositionState("20240101", 5),
            "B": PositionState("20240101", 5),
            "C": PositionState("20240101", 5),
        }
        daily = pd.DataFrame({"ts_code": ["A", "B", "C"], "score": [0.9, 0.8, 0.7]})
        new_positions, _, decisions = step_portfolio("20240110", daily, positions, 0.3, policy)
        self.assertEqual(set(new_positions), {"A", "B"})
        self.assertIn("position_limit_reduction", decisions["reason"].tolist())

    def test_new_positions_fill_top_five_vacancies(self):
        policy = StrategyPolicy(target_positions=5)
        daily = pd.DataFrame(
            {"ts_code": list("ABCDE"), "score": [0.9, 0.8, 0.7, 0.6, 0.5]}
        )
        positions, _, _ = step_portfolio("20240110", daily, {}, 1.0, policy)
        self.assertEqual(len(positions), 5)

    def test_weak_rank_exits_after_minimum_holding_period(self):
        policy = StrategyPolicy(target_positions=1, min_hold_sessions=1, retain_rank=1)
        positions = {"A": PositionState("20240101", 5)}
        daily = pd.DataFrame(
            {
                "ts_code": ["B", "A"],
                "score": [0.9, 0.1],
            }
        )
        new_positions, _, decisions = step_portfolio("20240110", daily, positions, 1.0, policy)
        self.assertNotIn("A", new_positions)
        self.assertIn("left_retain_zone", decisions["reason"].tolist())

    def test_untradable_candidate_is_not_bought(self):
        daily = pd.DataFrame(
            {
                "ts_code": ["A", "B"],
                "score": [0.9, 0.8],
                "execution_allowed": [False, True],
                "valuation_available": [True, True],
            }
        )
        positions, _, decisions = step_portfolio("20240110", daily, {}, 1.0)
        self.assertNotIn("A", positions)
        self.assertEqual(decisions["ts_code"].tolist(), ["B"])

    def test_missing_next_valuation_forces_data_quality_exit(self):
        positions = {"A": PositionState("20240101", 5)}
        daily = pd.DataFrame(
            {
                "ts_code": ["A"],
                "score": [0.9],
                "execution_allowed": [True],
                "valuation_available": [False],
            }
        )
        new_positions, _, decisions = step_portfolio("20240110", daily, positions, 1.0)
        self.assertNotIn("A", new_positions)
        self.assertIn("data_unavailable_next_valuation", decisions["reason"].tolist())

    def test_untradable_holding_is_not_fictitiously_sold(self):
        positions = {"A": PositionState("20240101", 5)}
        daily = pd.DataFrame(
            {
                "ts_code": ["A"],
                "score": [0.1],
                "execution_allowed": [False],
                "valuation_available": [False],
            }
        )
        new_positions, _, decisions = step_portfolio("20240110", daily, positions, 0.0)
        self.assertIn("A", new_positions)
        self.assertTrue(decisions.empty)


if __name__ == "__main__":
    unittest.main()
