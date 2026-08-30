import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.low_risk import (
    DEFAULT_LOW_RISK_CODE,
    build_low_risk_return_frame,
    build_selection_audit,
)


class LowRiskTests(unittest.TestCase):
    def test_selection_uses_only_2017_and_freezes_expected_etf(self):
        audit = build_selection_audit()
        selected = audit.loc[audit["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["ts_code"], DEFAULT_LOW_RISK_CODE)
        self.assertTrue(bool(selected.iloc[0]["eligible"]))

    def test_forward_return_uses_two_execution_opens(self):
        panel = pd.DataFrame(
            {
                "trade_date": ["20240102"],
                "next_open_date": ["20240103"],
                "return_end_date": ["20240104"],
            }
        )
        frame = build_low_risk_return_frame(panel)
        self.assertEqual(frame.loc[0, "low_risk_code"], DEFAULT_LOW_RISK_CODE)
        self.assertTrue(pd.notna(frame.loc[0, "forward_open_ret_1d"]))
        self.assertTrue(pd.notna(frame.loc[0, "intraday_return"]))


if __name__ == "__main__":
    unittest.main()
