import unittest

from ga_factor_mining.sector.rotation.ga_ablation import passes_incremental_gate


class GaAblationTests(unittest.TestCase):
    def test_candidate_must_improve_without_worsening_product_risk(self):
        baseline = {"ann_ret": 0.10, "sharpe": 1.0, "max_drawdown": -0.10, "avg_turnover": 0.08}
        candidate = {"ann_ret": 0.12, "sharpe": 1.06, "max_drawdown": -0.11, "avg_turnover": 0.085}
        self.assertTrue(passes_incremental_gate(baseline, candidate))
        candidate["max_drawdown"] = -0.13
        self.assertFalse(passes_incremental_gate(baseline, candidate))


if __name__ == "__main__":
    unittest.main()
