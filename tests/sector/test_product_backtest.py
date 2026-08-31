import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ga_factor_mining.sector.rotation.product_backtest import (
    _apply_rebalance_band,
    _drift_weights,
    _execution_actions,
    _turnover,
    append_latest_signal_strength,
    cost_worker_frame_is_current,
    latest_market_risk_snapshot,
    prepare_product_panel,
    product_feature_columns,
    run_product_backtest,
    summarize_backtest_period,
    write_latest_advice,
    write_latest_market_risk,
)
from ga_factor_mining.sector.rotation.strategy import StrategyPolicy


class ProductBacktestTests(unittest.TestCase):
    def test_cost_worker_cache_requires_matching_signatures(self):
        frame = pd.DataFrame(
            {
                "period": ["development", "selection", "full", "observation"],
                "cost_bps": [20.0] * 4,
                "policy_name": ["simple_v1"] * 4,
                "score_name": ["score_x"] * 4,
                "feature_protocol_version": [4] * 4,
                "feature_cache_signature": ["feature"] * 4,
                "strategy_policy_version": [5] * 4,
                "low_risk_data_signature": ["low_risk"] * 4,
                "full_path_rerun": [True] * 4,
                "stress_kind": ["full_system_replay_with_drawdown_feedback"] * 4,
            }
        )
        self.assertTrue(
            cost_worker_frame_is_current(
                frame,
                cost_bps=20.0,
                policy_name="simple_v1",
                feature_signature="feature",
                low_risk_signature="low_risk",
                expected_score_name="score_x",
            )
        )
        self.assertFalse(
            cost_worker_frame_is_current(
                frame,
                cost_bps=30.0,
                policy_name="simple_v1",
                feature_signature="feature",
                low_risk_signature="low_risk",
                expected_score_name="score_x",
            )
        )

    def test_market_risk_is_not_executable_when_product_layer_blocks(self):
        snapshot = {
            "status": "ready",
            "risk_data_quality": "complete",
            "risk_score": 60.0,
            "trend_health": 0.6,
            "breadth_positive_20d": 0.6,
            "breadth_positive_60d": 0.6,
            "breadth_20d_coverage": 1.0,
            "breadth_60d_coverage": 1.0,
            "volatility_health": 0.6,
            "pending_regime": None,
            "drawdown_cap": 1.0,
            "regime_base_exposure": 0.7,
            "actual_portfolio_exposure": 0.7,
            "risk_target_exposure": 0.7,
        }
        freshness = {
            "data_age_days": 0,
            "data_stale": False,
            "instruction_current": True,
            "action_plan_valid": True,
            "execution_allowed": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = write_latest_market_risk(
                snapshot,
                Path(temp_dir),
                data_freshness=freshness,
            )
        self.assertFalse(payload["execution_allowed"])
        self.assertIn("ETF执行层未就绪", payload["reason"])

    @staticmethod
    def _three_day_live_tail_panel() -> pd.DataFrame:
        rows = []
        dates = ["20240102", "20240103", "20240104"]
        for day_index, date in enumerate(dates):
            for sector_index in range(5):
                rows.append(
                    {
                        "ts_code": f"S{sector_index}",
                        "trade_date": date,
                        "type": "I",
                        "open": 10.0 + day_index + sector_index,
                        "close": 10.1 + day_index + sector_index,
                        "forward_open_ret_1d": 0.01 if day_index == 0 else np.nan,
                        "next_open_date": dates[day_index + 1] if day_index < 2 else np.nan,
                        "return_end_date": dates[day_index + 2] if day_index == 0 else np.nan,
                        "ret_1d": 0.01,
                        "ret_5d": 0.02,
                        "ret_20d": 0.03,
                        "ret_60d": 0.04,
                        "volatility_20d": 0.02,
                        "ret_5d_rank": (sector_index + 1) / 5,
                        "volatility_20d_rank": (sector_index + 1) / 5,
                        "model_score": float(sector_index + day_index / 10),
                    }
                )
        return pd.DataFrame(rows)

    def test_live_tail_separates_executed_unsettled_and_planned(self):
        panel = self._three_day_live_tail_panel()
        plan = {}
        daily, _, _ = run_product_backtest(
            panel,
            "model_score",
            "20240102",
            "20240104",
            use_market_regime=False,
            use_drawdown_cap=False,
            latest_plan_sink=plan,
            planned_execution_date="20240105",
        )
        self.assertEqual(daily.iloc[-1]["date"], "20240104")
        self.assertEqual(daily.iloc[-1]["signal_date"], "20240103")
        self.assertEqual(plan["signal_date"], "20240104")
        self.assertEqual(plan["planned_execution_date"], "20240105")
        self.assertEqual(plan["stage"], "planned")
        self.assertEqual(plan["simulated_portfolio_asof_date"], "20240104")

        poisoned = panel.copy()
        poisoned.loc[poisoned["trade_date"].eq("20240103"), "return_end_date"] = "20991231"
        poisoned_plan = {}
        run_product_backtest(
            poisoned,
            "model_score",
            "20240102",
            "20240104",
            use_market_regime=False,
            use_drawdown_cap=False,
            latest_plan_sink=poisoned_plan,
            planned_execution_date="20240105",
        )
        self.assertEqual(plan["target_weights"], poisoned_plan["target_weights"])

    def test_latest_risk_uses_panel_asof_not_last_backtest_signal(self):
        dates = pd.bdate_range("2024-01-02", periods=90).strftime("%Y%m%d")
        rows = []
        for day_index, date in enumerate(dates):
            for sector_index in range(30):
                rows.append(
                    {
                        "trade_date": date,
                        "ts_code": f"S{sector_index:02d}",
                        "type": "I",
                        "ret_1d": 0.001 * np.sin(day_index + sector_index),
                        "ret_20d": 0.01 if sector_index < 20 else -0.01,
                        "ret_60d": 0.02 if sector_index < 18 else -0.02,
                    }
                )
        daily = pd.DataFrame(
            {"signal_date": [dates[-3]], "drawdown_cap": [0.7], "exposure": [0.69]}
        )
        snapshot = latest_market_risk_snapshot(pd.DataFrame(rows), daily)
        self.assertEqual(snapshot["risk_asof_date"], dates[-1])
        self.assertAlmostEqual(
            snapshot["risk_target_exposure"],
            min(snapshot["regime_base_exposure"], snapshot["drawdown_cap"]),
        )

    def test_latest_absolute_strength_combines_rank_and_risk_exposure(self):
        panel = pd.DataFrame(
            {
                "ts_code": ["A.TI", "B.TI", "A.TI", "B.TI"],
                "trade_date": ["20240102", "20240102", "20240103", "20240103"],
                "type": ["I", "I", "I", "I"],
                "model_score": [0.8, 0.2, 0.9, 0.1],
            }
        )
        daily = pd.DataFrame(
            {"signal_date": ["20240103"], "risk_target_exposure": [0.7]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "板块代码": ["A.TI", "LOW_RISK"],
                    "目标权重": ["70.00%", "30.00%"],
                }
            ).to_csv(
                output_dir / "LATEST_TARGET_PORTFOLIO.csv",
                index=False,
                encoding="utf-8-sig",
            )
            append_latest_signal_strength(
                panel,
                "model_score",
                StrategyPolicy(score_smoothing_sessions=2),
                daily,
                {
                    "risk_asof_date": "20240103",
                    "risk_target_exposure": 0.7,
                    "risk_score": 60.0,
                },
                output_dir,
            )
            result = pd.read_csv(output_dir / "LATEST_TARGET_PORTFOLIO.csv")
        sector = result.loc[result["板块代码"].eq("A.TI")].iloc[0]
        self.assertEqual(sector["模型相对排名"], 1.0)
        self.assertEqual(sector["风险目标仓位"], "70.00%")
        self.assertEqual(sector["连续风险调整强度"], "60.00%")

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
        self.assertEqual(status.loc[0, "策略动作"], "信号尚未推进到最新行情")
        self.assertEqual(status.loc[0, "执行提示"], "策略信号滞后于行情，禁止执行")
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
        self.assertEqual(status.loc[0, "策略动作"], "信号尚未推进到最新行情")
        self.assertEqual(status.loc[0, "执行提示"], "数据已过期，禁止执行")

    def test_historical_action_on_data_end_is_never_reissued(self):
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
            result = write_latest_advice(
                daily,
                actions,
                Path(temp_dir),
                {"A.TI": "A板块"},
                asof_date="20240104",
                market_data_end_date="20240104",
            )
            latest = pd.read_csv(Path(temp_dir) / "LATEST_ACTIONS.csv")
            status = pd.read_csv(Path(temp_dir) / "LATEST_STATUS.csv")
        self.assertFalse(result["instruction_current"])
        self.assertFalse(result["execution_allowed"])
        self.assertTrue(latest.empty)
        self.assertEqual(status.loc[0, "指令状态"], "禁止执行")

    def test_only_latest_close_future_plan_can_be_reviewed(self):
        daily = pd.DataFrame(
            {
                "date": ["20240103"],
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
            result = write_latest_advice(
                daily,
                actions,
                Path(temp_dir),
                {"A.TI": "A板块"},
                asof_date="20240103",
                market_data_end_date="20240103",
                etf_execution_ready=True,
            )
            latest = pd.read_csv(Path(temp_dir) / "LATEST_ACTIONS.csv")
        self.assertTrue(result["instruction_current"])
        self.assertTrue(result["action_plan_valid"])
        self.assertTrue(result["execution_allowed"])
        self.assertEqual(latest["板块代码"].tolist(), ["A.TI"])

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
