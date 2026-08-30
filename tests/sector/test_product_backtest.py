import unittest
import tempfile
from pathlib import Path

import pandas as pd

from ga_factor_mining.sector.rotation.product_backtest import (
    _apply_rebalance_band,
    _drift_weights,
    _execution_actions,
    _turnover,
    prepare_product_panel,
    product_feature_columns,
    summarize_backtest_period,
    write_latest_advice,
)


class ProductBacktestTests(unittest.TestCase):
    def test_latest_advice_reconstructs_last_target_portfolio(self):
        daily = pd.DataFrame(
            {
                "date": ["20240104"],
                "signal_date": ["20240103"],
                "regime": ["NEUTRAL"],
                "exposure": [0.3],
                "low_risk_weight": [0.7],
                "position_count": [1],
            }
        )
        actions = pd.DataFrame(
            [
                {
                    "signal_date": "20240102",
                    "execution_date": "20240103",
                    "ts_code": "A.TI",
                    "action": "buy",
                    "trade_type": "entry",
                    "reason": "vacancy_and_strong_signal",
                    "current_weight": 0.0,
                    "target_weight": 0.3,
                    "weight_change": 0.3,
                    "regime": "NEUTRAL",
                    "etf_code": None,
                },
                {
                    "signal_date": "20240102",
                    "execution_date": "20240103",
                    "ts_code": "LOW_RISK",
                    "action": "sell",
                    "trade_type": "rebalance",
                    "reason": "low_risk_residual_allocation",
                    "current_weight": 1.0,
                    "target_weight": 0.7,
                    "weight_change": -0.3,
                    "regime": "NEUTRAL",
                    "etf_code": "511880.SH",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_latest_advice(
                daily,
                actions,
                output_dir,
                {"A.TI": "A板块"},
                asof_date="20240105",
            )
            latest = pd.read_csv(output_dir / "LATEST_ACTIONS.csv")
            last_rebalance = pd.read_csv(output_dir / "LAST_REBALANCE_ACTIONS.csv")
            portfolio = pd.read_csv(output_dir / "LATEST_TARGET_PORTFOLIO.csv")
            status = pd.read_csv(output_dir / "LATEST_STATUS.csv")
        self.assertTrue(latest.empty)
        self.assertEqual(last_rebalance["指令"].tolist(), ["买入", "卖出"])
        self.assertEqual(status.loc[0, "策略动作"], "持有不动")
        self.assertEqual(status.loc[0, "执行提示"], "无需交易")
        self.assertEqual(portfolio["目标权重"].tolist(), ["70.00%", "30.00%"])
        self.assertEqual(set(portfolio["板块代码"]), {"A.TI", "LOW_RISK"})

    def test_stale_data_suppresses_current_orders(self):
        daily = pd.DataFrame(
            {
                "date": ["20240104"],
                "signal_date": ["20240103"],
                "regime": ["RISK_ON"],
                "exposure": [1.0],
                "low_risk_weight": [0.0],
                "position_count": [1],
            }
        )
        actions = pd.DataFrame(
            [
                {
                    "signal_date": "20240103",
                    "execution_date": "20240104",
                    "ts_code": "A.TI",
                    "action": "buy",
                    "trade_type": "entry",
                    "reason": "vacancy_and_strong_signal",
                    "current_weight": 0.0,
                    "target_weight": 1.0,
                    "weight_change": 1.0,
                    "regime": "RISK_ON",
                    "etf_code": None,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = write_latest_advice(
                daily,
                actions,
                output_dir,
                {"A.TI": "A板块"},
                asof_date="20240201",
            )
            latest = pd.read_csv(output_dir / "LATEST_ACTIONS.csv")
            status = pd.read_csv(output_dir / "LATEST_STATUS.csv")
        self.assertTrue(result["data_stale"])
        self.assertTrue(latest.empty)
        self.assertEqual(status.loc[0, "策略动作"], "有新调仓建议")
        self.assertEqual(status.loc[0, "执行提示"], "数据已过期，禁止执行")

    def test_prepare_product_panel_drops_unused_features(self):
        panel = pd.DataFrame(
            {
                "ts_code": ["A.TI"],
                "trade_date": ["20240102"],
                "type": ["I"],
                "open": [10.0],
                "close": [10.2],
                "forward_open_ret_1d": [0.01],
                "next_open_date": ["20240103"],
                "return_end_date": ["20240104"],
                "ret_1d": [0.02],
                "ret_5d": [0.03],
                "ret_20d": [0.04],
                "ret_60d": [0.05],
                "volatility_20d": [0.01],
                "ret_5d_rank": [0.7],
                "volatility_20d_rank": [0.2],
                "frozen_score": [0.8],
                "unused_feature": [999.0],
            }
        )
        slim = prepare_product_panel(panel, "frozen_score", "industry_concept")
        self.assertIn("frozen_score", slim.columns)
        self.assertNotIn("unused_feature", slim.columns)
        self.assertEqual(float(slim.loc[0, "frozen_score"]), 0.8)

    def test_external_product_projection_excludes_score_column(self):
        columns = product_feature_columns("frozen_score", external_score=True)
        self.assertNotIn("frozen_score", columns)
        self.assertIn("forward_open_ret_1d", columns)

    def test_period_summary_slices_continuous_path_without_reset(self):
        daily = pd.DataFrame(
            {
                "date": ["20251231", "20260105", "20260106"],
                "net_return": [0.10, 0.02, -0.01],
                "turnover": [0.0, 0.2, 0.0],
                "cost": [0.0, 0.0004, 0.0],
                "position_count": [5, 5, 5],
                "exposure": [1.0, 0.7, 0.7],
                "low_risk_weight": [0.0, 0.3, 0.3],
            }
        )
        period, _, metrics = summarize_backtest_period(
            daily, pd.DataFrame(), "20260101", "20261231"
        )
        self.assertEqual(period["date"].tolist(), ["20260105", "20260106"])
        self.assertAlmostEqual(metrics["total_ret"], (1.02 * 0.99) - 1.0)
        self.assertAlmostEqual(metrics["avg_exposure"], 0.7)

    def test_turnover_includes_cash(self):
        self.assertAlmostEqual(_turnover({}, {"A": 0.3}), 0.3)
        self.assertAlmostEqual(_turnover({"A": 0.3}, {}), 0.3)

    def test_drifted_weights_are_self_financing(self):
        drifted = _drift_weights({"A": 0.5, "B": 0.5}, {"A": 0.1, "B": 0.0}, 0.05)
        self.assertAlmostEqual(drifted["A"], 0.55 / 1.05)
        self.assertAlmostEqual(drifted["B"], 0.50 / 1.05)
        self.assertAlmostEqual(sum(drifted.values()), 1.0)

    def test_rebalance_band_preserves_small_weight_drift(self):
        pretrade = {"A": 0.51, "B": 0.49}
        desired = {"A": 0.50, "B": 0.50}
        self.assertEqual(_apply_rebalance_band(pretrade, desired, 0.03), pretrade)
        self.assertEqual(
            _apply_rebalance_band({"A": 0.55, "B": 0.45}, desired, 0.03),
            desired,
        )

    def test_execution_actions_include_actual_weight_change(self):
        actions = _execution_actions(
            "20240105",
            "20240108",
            decisions=pd.DataFrame(
                [{"ts_code": "A", "action": "buy", "reason": "strong_signal", "held_sessions": 0}]
            ),
            pretrade={},
            target={"A": 0.3},
            turnover=0.3,
            cost_rate=0.002,
            regime="NEUTRAL",
        )
        self.assertEqual(actions.loc[0, "trade_type"], "entry")
        self.assertAlmostEqual(actions.loc[0, "target_weight"], 0.3)
        self.assertAlmostEqual(actions.loc[0, "portfolio_expected_cost"], 0.0006)

    def test_execution_actions_expose_low_risk_etf_weight_change(self):
        actions = _execution_actions(
            "20240105",
            "20240108",
            decisions=pd.DataFrame(),
            pretrade={},
            target={"A": 0.3},
            turnover=0.3,
            cost_rate=0.002,
            regime="NEUTRAL",
            low_risk_code="511880.SH",
        )
        low_risk = actions.loc[actions["ts_code"].eq("LOW_RISK")].iloc[0]
        self.assertEqual(low_risk["etf_code"], "511880.SH")
        self.assertAlmostEqual(low_risk["current_weight"], 1.0)
        self.assertAlmostEqual(low_risk["target_weight"], 0.7)


if __name__ == "__main__":
    unittest.main()
