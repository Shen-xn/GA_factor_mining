import unittest

import numpy as np
import pandas as pd

from ga_factor_mining.sector.factor_mining.run_factor_mining import (
    Evaluator,
    load_config,
    simple_derived_duplicate,
    valid_expr,
)
from ga_factor_mining.sector.rotation.run_experiments import rank_cs, safe_div


class SectorTests(unittest.TestCase):
    def test_cross_section_helpers(self):
        values = pd.Series([3.0, 1.0, 2.0])
        self.assertEqual(rank_cs(values).tolist(), [1.0, 1 / 3, 2 / 3])
        result = safe_div(pd.Series([2.0, 1.0]), pd.Series([2.0, 0.0]))
        self.assertEqual(result.iloc[0], 1.0)
        self.assertTrue(np.isnan(result.iloc[1]))

    def test_temporal_expression_is_backward_only(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["A"] * 6,
                "trade_date": [str(index) for index in range(6)],
                "ret": np.arange(6, dtype=float),
            }
        )
        value = Evaluator(frame).eval(["delta_5", "ret"])
        self.assertTrue(value.iloc[:5].isna().all())
        self.assertEqual(value.iloc[5], 5.0)
        self.assertTrue(valid_expr(["add", ["mean_5", "ret"], "ret"]))
        self.assertFalse(valid_expr(["mean_5", ["add", "ret", "ret"]]))

    def test_protected_division_is_finite_and_clipped(self):
        frame = pd.DataFrame({"a": [1.0], "b": [0.0]})
        value = Evaluator(frame, max_abs=10.0, div_epsilon=0.1).eval(["div", "a", "b"])
        self.assertEqual(value.iloc[0], 10.0)

    def test_ga_uses_only_rank_domain_terminals(self):
        cfg = load_config("configs/sector/factor_mining.json")
        terminals = [term for group in cfg["terminal_groups"].values() for term in group]
        self.assertTrue(terminals)
        self.assertTrue(all(term.endswith("_rank") for term in terminals))

    def test_simple_manual_linear_rewrite_is_structural_duplicate(self):
        self.assertTrue(simple_derived_duplicate(["sub", "ret_10d_rank", "ret_20d_rank"]))
        self.assertFalse(simple_derived_duplicate(["std_20", "ret_3d_rank"]))


if __name__ == "__main__":
    unittest.main()
