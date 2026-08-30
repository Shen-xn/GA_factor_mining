import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from ga_factor_mining.sector.rotation.run_experiments import (
    StrategyConfig,
    FEATURE_LOGIC_SIGNATURE,
    FEATURE_PROTOCOL_VERSION,
    _metadata_signature,
    add_forward_open_returns,
    add_cross_sectional_ranks,
    backtest_one_cached,
    feature_cache_is_current,
    matured_training_mask,
    select_best_result,
    MARKET_CONTEXT_COLS,
)
from ga_factor_mining.sector.rotation.refresh_data import (
    _append_parquet,
    _replace_parquet_tail,
)


class RotationProtocolTests(unittest.TestCase):
    def test_ranks_are_computed_inside_the_requested_universe(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20240102"] * 4,
                "ts_code": ["I1", "I2", "N1", "R1"],
                "type": ["I", "I", "N", "R"],
                "ret_5d": [1.0, 2.0, 10.0, 100.0],
            }
        )
        industry_concept = add_cross_sectional_ranks(frame, ["I", "N"])
        self.assertEqual(industry_concept["ret_5d_rank"].tolist(), [1 / 3, 2 / 3, 1.0])
        industry = add_cross_sectional_ranks(frame, ["I"])
        self.assertEqual(industry["ret_5d_rank"].tolist(), [0.5, 1.0])

    def test_training_requires_the_label_to_be_realized(self):
        frame = pd.DataFrame(
            {
                "trade_date": ["20231220", "20231228"],
                "future_ret_5d_end_date": ["20231227", "20240105"],
                "future_ret_5d_rank": [0.8, 0.9],
            }
        )
        mask = matured_training_mask(frame, "20231231", 5)
        self.assertEqual(mask.tolist(), [True, False])

    def test_close_signal_uses_next_open_to_following_open_return(self):
        score = pd.DataFrame(
            {"A": [0.9, 0.8], "B": [0.1, 0.2]},
            index=["20240101", "20240102"],
        )
        forward_returns = pd.DataFrame(
            {"A": [0.10, 0.20], "B": [0.0, 0.0]},
            index=["20240101", "20240102"],
        )
        cache = {
            "score_pivots": {"score": score},
            "ret_pivot": forward_returns,
            "name_map": {"A": "甲", "B": "乙"},
            "return_date_map": {"20240101": "20240103", "20240102": "20240104"},
            "execution_date_map": {"20240101": "20240102", "20240102": "20240103"},
        }
        config = StrategyConfig("test", "industry", "score", 1, 1, 1)
        returns, positions, aux = backtest_one_cached(cache, config, "20240101", "20240104")
        self.assertEqual(returns.index.tolist(), ["20240103", "20240104"])
        self.assertEqual(returns.tolist(), [0.10, 0.20])
        self.assertEqual(positions["execution_date"].tolist(), ["20240102", "20240103"])
        # 首次从现金建仓换手为1，随后同一单资产组合无需再平衡。
        self.assertEqual(aux["avg_turnover"], 0.5)

    def test_global_calendar_does_not_skip_missing_instrument_date(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["A", "A", "A", "B", "B"],
                "trade_date": ["20240101", "20240102", "20240103", "20240101", "20240103"],
                "open": [100.0, 110.0, 121.0, 50.0, 55.0],
            }
        )
        result = add_forward_open_returns(frame, horizons=(1,))
        b_first = result[(result["ts_code"] == "B") & (result["trade_date"] == "20240101")].iloc[0]
        self.assertEqual(b_first["next_open_date"], "20240102")
        self.assertTrue(pd.isna(b_first["forward_open_ret_1d"]))

    def test_five_day_label_excludes_signal_to_entry_overnight_move(self):
        dates = [f"2024010{i}" for i in range(1, 8)]
        frame = pd.DataFrame(
            {
                "ts_code": ["A"] * 7,
                "trade_date": dates,
                "open": [10.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            }
        )
        result = add_forward_open_returns(frame, horizons=(5,))
        first = result.iloc[0]
        self.assertEqual(first["future_ret_5d_end_date"], "20240107")
        self.assertAlmostEqual(first["future_ret_5d"], 30.0 / 20.0 - 1.0)

    def test_turnover_includes_weight_drift_even_when_membership_is_unchanged(self):
        score = pd.DataFrame(
            {"A": [0.9, 0.9], "B": [0.8, 0.8]},
            index=["20240101", "20240102"],
        )
        forward_returns = pd.DataFrame(
            {"A": [0.10, 0.0], "B": [0.0, 0.0]},
            index=score.index,
        )
        cache = {
            "score_pivots": {"score": score},
            "ret_pivot": forward_returns,
            "name_map": {},
            "return_date_map": {"20240101": "20240103", "20240102": "20240104"},
            "execution_date_map": {"20240101": "20240102", "20240102": "20240103"},
        }
        config = StrategyConfig("test", "industry", "score", 2, 1, 1)
        _, _, aux = backtest_one_cached(cache, config, "20240101", "20240104")
        drift_turnover = abs(0.5 - 0.55 / 1.05)
        self.assertAlmostEqual(aux["avg_turnover"], (1.0 + drift_turnover) / 2.0)

    def test_theoretical_engine_does_not_turn_missing_return_into_zero(self):
        score = pd.DataFrame(
            {"A": [0.9], "B": [0.8]},
            index=["20240101"],
        )
        forward_returns = pd.DataFrame(
            {"A": [float("nan")], "B": [0.02]},
            index=score.index,
        )
        cache = {
            "score_pivots": {"score": score},
            "ret_pivot": forward_returns,
            "name_map": {},
            "return_date_map": {"20240101": "20240103"},
            "execution_date_map": {"20240101": "20240102"},
        }
        config = StrategyConfig("test", "industry", "score", 1, 1, 1)
        returns, positions, _ = backtest_one_cached(cache, config, "20240101", "20240103")
        self.assertEqual(positions["ts_code"].tolist(), ["B"])
        self.assertAlmostEqual(float(returns.iloc[0]), 0.02)

    def test_theoretical_engine_fails_if_live_holding_loses_return_data(self):
        score = pd.DataFrame(
            {"A": [0.9, 0.9]},
            index=["20240101", "20240102"],
        )
        forward_returns = pd.DataFrame(
            {"A": [0.01, float("nan")]},
            index=score.index,
        )
        cache = {
            "score_pivots": {"score": score},
            "ret_pivot": forward_returns,
            "name_map": {},
            "return_date_map": {"20240101": "20240103", "20240102": "20240104"},
            "execution_date_map": {"20240101": "20240102", "20240102": "20240103"},
        }
        config = StrategyConfig("test", "industry", "score", 1, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "持仓缺少次日开盘收益"):
            backtest_one_cached(cache, config, "20240101", "20240104")

    def test_observation_metrics_cannot_change_best_selection(self):
        base = pd.DataFrame(
            {
                "strategy_id": ["A", "B"],
                "val_sharpe": [1.0, 1.0],
                "val_ann_ret": [0.2, 0.2],
                "val_excess_ann_ret": [0.1, 0.1],
                "observation_sharpe": [-10.0, 10.0],
            }
        )
        self.assertEqual(select_best_result(base).iloc[0]["strategy_id"], "A")
        changed = base.copy()
        changed["observation_sharpe"] *= -1
        self.assertEqual(select_best_result(changed).iloc[0]["strategy_id"], "A")

    def test_feature_cache_requires_matching_protocol_metadata(self):
        with TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.parquet"
            meta_path = Path(directory) / "features.meta.json"
            feature_path.touch()
            meta_path.write_text('{"feature_protocol_version": 1}', encoding="utf-8")
            self.assertFalse(feature_cache_is_current(feature_path, meta_path))
            sources = {
                "source.parquet": {
                    "path": "C:/old-machine/source.parquet",
                    "sha256": "abc",
                    "size": 1,
                    "mtime_ns": 1,
                }
            }
            metadata = {
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
                "feature_logic_signature": FEATURE_LOGIC_SIGNATURE,
                "sources": sources,
            }
            metadata["feature_cache_signature"] = _metadata_signature(metadata)
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch(
                "ga_factor_mining.sector.rotation.run_experiments.source_data_fingerprints",
                return_value={
                    "source.parquet": {
                        "path": "data/sector/source.parquet",
                        "sha256": "abc",
                        "size": 1,
                    }
                },
            ):
                self.assertTrue(feature_cache_is_current(feature_path, meta_path))

    def test_market_context_columns_are_explicit_model_inputs(self):
        self.assertEqual(len(MARKET_CONTEXT_COLS), 5)
        self.assertEqual(len(MARKET_CONTEXT_COLS), len(set(MARKET_CONTEXT_COLS)))

    def test_incremental_parquet_write_appends_and_replaces_only_tail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            appended = root / "appended.parquet"
            replaced = root / "replaced.parquet"
            base = pd.DataFrame(
                {
                    "ts_code": ["A", "A", "A"],
                    "trade_date": ["20240101", "20240102", "20240103"],
                    "value": [1.0, 2.0, 3.0],
                }
            )
            base.to_parquet(source, index=False)
            _append_parquet(
                source,
                pd.DataFrame(
                    {"ts_code": ["A"], "trade_date": ["20240104"], "value": [4.0]}
                ),
                appended,
            )
            self.assertEqual(pd.read_parquet(appended)["value"].tolist(), [1.0, 2.0, 3.0, 4.0])

            _replace_parquet_tail(
                appended,
                pd.DataFrame(
                    {
                        "ts_code": ["A", "A"],
                        "trade_date": ["20240103", "20240104"],
                        "value": [30.0, 40.0],
                    }
                ),
                "20240103",
                replaced,
            )
            result = pd.read_parquet(replaced)
            self.assertEqual(result["trade_date"].tolist(), ["20240101", "20240102", "20240103", "20240104"])
            self.assertEqual(result["value"].tolist(), [1.0, 2.0, 30.0, 40.0])


if __name__ == "__main__":
    unittest.main()
