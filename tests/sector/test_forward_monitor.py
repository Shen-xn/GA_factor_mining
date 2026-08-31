import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ga_factor_mining.sector.rotation.forward_monitor import record_forward_snapshot


class ForwardMonitorTests(unittest.TestCase):
    def test_snapshot_is_idempotent_and_source_change_stops_append(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            strategy = root / "strategy"
            forward = root / "forward"
            strategy.mkdir()
            source = root / "decision.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            config = root / "forward.json"
            config.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "freeze_data_end": "20260810",
                        "first_unseen_after": "20260810",
                        "evidence_name": "test",
                        "cost_bps": 20.0,
                    }
                ),
                encoding="utf-8",
            )
            (strategy / "POLICY.json").write_text(
                json.dumps(
                    {
                        "strategy_policy_version": 4,
                        "policy_name": "simple_v1",
                        "policy": {"target_positions": 5},
                        "boundary_mode": "continuous_carry_from_2018",
                        "low_risk_code": "511880.SH",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "数据截止日": "20260810",
                        "策略日期": "20260806",
                        "最近调仓信号日": "20260714",
                        "策略动作": "持有不动",
                        "市场状态": "CASH",
                        "当前板块仓位": "0.00%",
                        "当前低风险仓位": "100.00%",
                        "当前持仓数量": 0,
                    }
                ]
            ).to_csv(strategy / "LATEST_STATUS.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "策略日期": "20260806",
                        "目标形成日期": "20260714",
                        "板块代码": "LOW_RISK",
                        "板块名称": "货币ETF",
                        "目标权重": "100.00%",
                        "ETF代码": "511880.SH",
                    }
                ]
            ).to_csv(strategy / "LATEST_TARGET_PORTFOLIO.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {
                    "date": ["20260810"],
                    "net_return": [0.0],
                    "turnover": [0.0],
                }
            ).to_parquet(strategy / "HISTORY_DAILY.parquet", index=False)

            first = record_forward_snapshot(
                strategy, forward, config, source_paths=[source]
            )
            second = record_forward_snapshot(
                strategy, forward, config, source_paths=[source]
            )
            self.assertEqual(first["status"], "recorded")
            self.assertEqual(second["protocol_hash"], first["protocol_hash"])
            self.assertEqual(len(pd.read_csv(forward / "SNAPSHOTS.csv")), 1)

            source.write_text("VERSION = 2\n", encoding="utf-8")
            mismatch = record_forward_snapshot(
                strategy, forward, config, source_paths=[source]
            )
            self.assertEqual(mismatch["status"], "protocol_mismatch")
            self.assertEqual(len(pd.read_csv(forward / "SNAPSHOTS.csv")), 1)

            source.write_text("VERSION = 1\n", encoding="utf-8")
            recovered = record_forward_snapshot(
                strategy, forward, config, source_paths=[source]
            )
            self.assertEqual(recovered["status"], "recorded")
            self.assertEqual(recovered["protocol_hash"], first["protocol_hash"])
            self.assertEqual(len(pd.read_csv(forward / "SNAPSHOTS.csv")), 1)


if __name__ == "__main__":
    unittest.main()
