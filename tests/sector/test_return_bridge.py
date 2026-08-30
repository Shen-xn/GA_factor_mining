import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.return_bridge import _direct_topk_backtest


class ReturnBridgeTests(unittest.TestCase):
    def test_direct_topk_uses_one_shared_execution_ledger(self):
        rows = []
        dates = ["20240101", "20240102", "20240103", "20240104"]
        opens = {
            "A": [10.0, 10.0, 11.0, 11.0],
            "B": [10.0, 10.0, 9.0, 10.0],
        }
        scores = {
            "A": [0.9, 0.1, 0.1, 0.1],
            "B": [0.1, 0.9, 0.9, 0.9],
        }
        for code in ("A", "B"):
            for index, date in enumerate(dates):
                next_date = dates[index + 1] if index + 1 < len(dates) else None
                end_date = dates[index + 2] if index + 2 < len(dates) else None
                forward_return = (
                    opens[code][index + 2] / opens[code][index + 1] - 1.0
                    if end_date is not None
                    else None
                )
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": date,
                        "type": "I",
                        "open": opens[code][index],
                        "next_open_date": next_date,
                        "return_end_date": end_date,
                        "forward_open_ret_1d": forward_return,
                        "score": scores[code][index],
                    }
                )

        daily = _direct_topk_backtest(
            pd.DataFrame(rows),
            "score",
            "20240101",
            "20240104",
            smoothing_sessions=1,
            cost_bps=20.0,
            top_k=1,
        )

        self.assertEqual(daily["date"].tolist(), ["20240102", "20240103", "20240104"])
        self.assertFalse(daily["date"].duplicated().any())
        self.assertAlmostEqual(
            float(
                (
                    daily["net_return"]
                    - ((1.0 + daily["gross_return"]) * (1.0 - daily["cost"]) - 1.0)
                ).abs().max()
            ),
            0.0,
        )
        self.assertEqual(daily["position_count"].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
