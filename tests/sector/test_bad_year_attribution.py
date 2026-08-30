import unittest

import pandas as pd

from ga_factor_mining.sector.rotation.bad_year_attribution import (
    add_daily_components,
    link_period,
)


class BadYearAttributionTests(unittest.TestCase):
    def test_components_exactly_link_to_period_return(self):
        history = pd.DataFrame(
            {
                "date": ["20220103", "20220104"],
                "sector_contribution": [0.0, 0.05],
                "low_risk_contribution": [0.0, 0.01],
                "cost": [0.001, 0.002],
                "net_return": [-0.001, (1.06 * 0.998) - 1.0],
                "exposure": [0.5, 0.0],
                "regime": ["NEUTRAL", "CASH"],
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": ["20220104"],
                "benchmark_return": [0.08],
                "benchmark_member_count": [100],
            }
        )
        attributed = add_daily_components(history, benchmark)
        self.assertAlmostEqual(float(attributed.loc[1, "held_exposure"]), 0.5)
        self.assertAlmostEqual(float(attributed.loc[1, "market_gross_component"]), 0.04)
        self.assertAlmostEqual(float(attributed.loc[1, "selection_gross_component"]), 0.01)
        components, total_return = link_period(attributed)
        self.assertAlmostEqual(sum(components.values()), total_return)
        self.assertAlmostEqual(total_return, (0.999 * 1.05788) - 1.0)

    def test_nonzero_sector_return_requires_benchmark(self):
        history = pd.DataFrame(
            {
                "date": ["20220103"],
                "sector_contribution": [0.01],
                "low_risk_contribution": [0.0],
                "cost": [0.0],
                "net_return": [0.01],
                "exposure": [1.0],
                "regime": ["RISK_ON"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "缺少基准"):
            add_daily_components(
                history,
                pd.DataFrame(
                    columns=["date", "benchmark_return", "benchmark_member_count"]
                ),
            )

    def test_year_filter_preserves_previous_year_exposure(self):
        history = pd.DataFrame(
            {
                "date": ["20211231", "20220104"],
                "sector_contribution": [0.0, 0.0],
                "low_risk_contribution": [0.0, 0.0],
                "cost": [0.0, 0.0],
                "net_return": [0.0, 0.0],
                "exposure": [0.7, 0.3],
                "regime": ["RISK_ON", "DEFENSIVE"],
            }
        )
        attributed = add_daily_components(
            history,
            pd.DataFrame(
                {
                    "date": ["20220104"],
                    "benchmark_return": [0.01],
                    "benchmark_member_count": [100],
                }
            ),
            attribution_years=(2022,),
        )
        self.assertEqual(attributed["date"].tolist(), ["20220104"])
        self.assertAlmostEqual(float(attributed.iloc[0]["held_exposure"]), 0.7)
        self.assertEqual(attributed.iloc[0]["held_regime"], "RISK_ON")


if __name__ == "__main__":
    unittest.main()
