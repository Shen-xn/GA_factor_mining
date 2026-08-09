import unittest

import numpy as np
import pandas as pd

from ga_factor_mining.sector.factor_mining.run_factor_mining import Evaluator, valid_expr
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


if __name__ == "__main__":
    unittest.main()
