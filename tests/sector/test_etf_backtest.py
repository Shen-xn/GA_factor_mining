import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.etf_backtest import (
    run_resolved_etf_backtest,
)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asof_date": ["20240131"],
            "effective_from": ["20240201"],
            "effective_to": [None],
            "sector_code": ["A.TI"],
            "etf_code": ["A.SH"],
            "selected": [True],
            "mapping_score": [0.8],
            "median_amount20": [100_000.0],
        }
    )


class EtfBacktestTests(unittest.TestCase):
    def test_true_etf_open_replay_preserves_weights_and_coverage(self):
        timeline = pd.DataFrame(
            {
                "signal_date": ["20240131", "20240131", "20240201", "20240201"],
                "execution_date": ["20240201", "20240201", "20240202", "20240202"],
                "stage": ["executed"] * 4,
                "asset_code": ["A.TI", "LOW_RISK", "A.TI", "LOW_RISK"],
                "target_weight": [0.5, 0.5, 0.5, 0.5],
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": ["20240201", "20240202", "20240201", "20240202"],
                "ts_code": ["A.SH", "A.SH", "511880.SH", "511880.SH"],
                "adj_open": [100.0, 110.0, 100.0, 100.0],
            }
        )
        result = run_resolved_etf_backtest(timeline, _mapping(), prices, cost_bps=0.0)
        self.assertAlmostEqual(result.summary["total_ret"], 0.05)
        self.assertAlmostEqual(result.summary["average_mapping_coverage"], 1.0)
        self.assertAlmostEqual(
            result.daily.iloc[-1]["mapped_equity_exposure"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
