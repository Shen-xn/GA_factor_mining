import unittest

import numpy as np
import pandas as pd

from ga_factor_mining.sector.rotation.etf_mapping import (
    MappingPolicy,
    build_latest_execution_readiness,
    candidate_fetch_ranges,
    build_strategy_allocation_audit,
    build_strategy_coverage_audit,
    build_monthly_mapping,
    build_strict_candidates,
    normalize_theme_name,
    resolve_target_weights,
)


class EtfMappingTests(unittest.TestCase):
    def test_candidate_fetch_ranges_resume_from_last_overlap_day(self):
        candidates = pd.DataFrame({"etf_code": ["A.SH", "B.SZ", "C.SH"]})
        daily = pd.DataFrame(
            {
                "ts_code": ["A.SH", "C.SH"],
                "trade_date": ["20260529", "20260829"],
            }
        )
        adj = pd.DataFrame(
            {
                "ts_code": ["A.SH", "C.SH"],
                "trade_date": ["20260528", "20260829"],
            }
        )
        ranges = candidate_fetch_ranges(
            candidates,
            daily,
            adj,
            "20150101",
            "20260828",
        )
        self.assertEqual(ranges, [("A.SH", "20260528"), ("B.SZ", "20150101")])

    def test_normalized_name_is_conservative_but_handles_common_suffixes(self):
        self.assertEqual(normalize_theme_name("芯片概念"), "芯片")
        self.assertEqual(normalize_theme_name("国证芯片指数"), "芯片")
        self.assertEqual(normalize_theme_name("证券Ⅲ"), "证券")

    def test_strict_candidates_are_not_created_from_unrelated_names(self):
        sectors = pd.DataFrame(
        {
            "ts_code": ["A.TI", "B.TI"],
            "name": ["半导体", "银行"],
            "type": ["N", "I"],
        }
    )
        etfs = pd.DataFrame(
        {
            "ts_code": ["512480.SH", "510300.SH", "159999.SZ"],
            "csname": ["半导体ETF", "沪深300ETF", "待上市半导体ETF"],
            "index_code": ["931865.CSI", "000300.SH", "931865.CSI"],
            "index_name": ["中证半导体", "沪深300", "中证半导体"],
            "list_date": ["20190612", "20120528", "20270101"],
            "list_status": ["L", "L", "P"],
            "etf_type": ["纯境内", "纯境内", "纯境内"],
        }
    )
        candidates = build_strict_candidates(sectors, etfs, "20260529")
        self.assertEqual(
            candidates[["sector_code", "etf_code"]].to_dict("records"),
            [{"sector_code": "A.TI", "etf_code": "512480.SH"}],
        )

    def test_monthly_mapping_uses_past_window_and_selects_stable_liquid_etf(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2023-01-02", periods=190).strftime("%Y%m%d")
        sector_a = rng.normal(0.0005, 0.012, len(dates))
        sector_b = rng.normal(0.0001, 0.010, len(dates))
        etf_a = sector_a + rng.normal(0.0, 0.001, len(dates))
        panel = pd.concat(
        [
            pd.DataFrame(
                {"trade_date": dates, "ts_code": "A.TI", "type": "N", "ret_1d": sector_a}
            ),
            pd.DataFrame(
                {"trade_date": dates, "ts_code": "B.TI", "type": "I", "ret_1d": sector_b}
            ),
        ],
        ignore_index=True,
    )
        prices = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": "512480.SH",
            "etf_ret_1d": etf_a,
            "amount": 200_000.0,
        }
    )
        candidates = pd.DataFrame(
        {
            "sector_code": ["A.TI"],
            "sector_name": ["半导体"],
            "sector_type": ["N"],
            "etf_code": ["512480.SH"],
            "etf_name": ["半导体ETF"],
            "index_code": ["931865.CSI"],
            "index_name": ["中证半导体"],
            "list_date": ["20190612"],
        }
    )
        mapping = build_monthly_mapping(
            panel,
            prices,
            candidates,
            MappingPolicy(minimum_median_amount_thousand_rmb=50_000.0),
        )
        selected = mapping[mapping["selected"]]
        self.assertFalse(selected.empty)
        self.assertLess(selected.iloc[-1]["asof_date"], selected.iloc[-1]["effective_from"])
        self.assertGreater(selected.iloc[-1]["corr120"], 0.9)

    def test_unmapped_and_duplicate_etf_weights_go_to_low_risk(self):
        mapping = pd.DataFrame(
        {
            "sector_code": ["A.TI", "B.TI"],
            "etf_code": ["512480.SH", "512480.SH"],
            "effective_from": ["20240101", "20240101"],
            "effective_to": [pd.NA, pd.NA],
            "mapping_score": [0.9, 0.8],
            "median_amount20": [200_000.0, 180_000.0],
            "selected": [True, True],
        }
    )
        resolved = resolve_target_weights(
            {"A.TI": 0.2, "B.TI": 0.2, "C.TI": 0.2}, mapping, "20240501"
        )
        self.assertEqual(
            resolved.loc[resolved["sector_code"].eq("A.TI"), "etf_code"].iloc[0],
            "512480.SH",
        )
        self.assertEqual(
            set(resolved.loc[resolved["sector_code"].isin(["B.TI", "C.TI"]), "etf_code"]),
            {"511880.SH"},
        )
        self.assertAlmostEqual(resolved.attrs["low_risk_weight_from_unmapped"], 0.4)

    def test_expired_monthly_mapping_is_not_carried_forward(self):
        mapping = pd.DataFrame(
            {
                "sector_code": ["A.TI"],
                "etf_code": ["512480.SH"],
                "effective_from": ["20240102"],
                "effective_to": ["20240201"],
                "mapping_score": [0.9],
                "median_amount20": [200_000.0],
                "selected": [True],
            }
        )
        resolved = resolve_target_weights({"A.TI": 0.5}, mapping, "20240201")
        self.assertEqual(resolved.iloc[0]["etf_code"], "511880.SH")
        self.assertEqual(resolved.iloc[0]["allocation_reason"], "unmapped_to_low_risk")

    def test_last_mapping_expires_after_its_monthly_review_cycle(self):
        mapping = pd.DataFrame(
            {
                "asof_date": ["20260430"],
                "sector_code": ["A.TI"],
                "etf_code": ["512480.SH"],
                "effective_from": ["20260506"],
                "effective_to": [pd.NA],
                "mapping_score": [0.9],
                "median_amount20": [200_000.0],
                "selected": [True],
            }
        )
        current = resolve_target_weights({"A.TI": 0.5}, mapping, "20260529")
        stale = resolve_target_weights({"A.TI": 0.5}, mapping, "20260810")
        self.assertEqual(current.iloc[0]["final_asset_code"], "512480.SH")
        self.assertEqual(stale.iloc[0]["final_asset_code"], "511880.SH")
        self.assertEqual(stale.iloc[0]["allocation_reason"], "unmapped_to_low_risk")

    def test_duplicate_etf_resolution_is_independent_of_input_order(self):
        mapping = pd.DataFrame(
            {
                "sector_code": ["A.TI", "B.TI"],
                "etf_code": ["512480.SH", "512480.SH"],
                "effective_from": ["20240101", "20240101"],
                "effective_to": [pd.NA, pd.NA],
                "mapping_score": [0.9, 0.8],
                "median_amount20": [200_000.0, 180_000.0],
                "selected": [True, True],
            }
        )
        first = resolve_target_weights({"A.TI": 0.2, "B.TI": 0.2}, mapping, "20240501")
        reversed_input = resolve_target_weights(
            {"B.TI": 0.2, "A.TI": 0.2}, mapping, "20240501"
        )
        columns = ["sector_code", "final_asset_code", "allocation_reason"]
        self.assertEqual(first[columns].to_dict("records"), reversed_input[columns].to_dict("records"))
        self.assertEqual(
            first.set_index("sector_code").loc["A.TI", "final_asset_code"],
            "512480.SH",
        )

    def test_strategy_coverage_audit_reconstructs_target_weights(self):
        mapping = pd.DataFrame(
            {
                "sector_code": ["A.TI"],
                "etf_code": ["512480.SH"],
                "effective_from": ["20240102"],
                "effective_to": [pd.NA],
                "mapping_score": [0.9],
                "median_amount20": [200_000.0],
                "selected": [True],
            }
        )
        actions = pd.DataFrame(
            {
                "signal_date": ["20240102", "20240102", "20240103"],
                "execution_date": ["20240103", "20240103", "20240104"],
                "ts_code": ["A.TI", "B.TI", "B.TI"],
                "target_weight": [0.3, 0.3, 0.0],
            }
        )
        audit = build_strategy_coverage_audit(actions, mapping)
        self.assertAlmostEqual(audit.iloc[0]["risk_weight_coverage"], 0.5)
        self.assertAlmostEqual(audit.iloc[1]["risk_weight_coverage"], 1.0)
        allocations = build_strategy_allocation_audit(actions, mapping)
        first_day = allocations[allocations["execution_date"].eq("20240103")]
        self.assertEqual(
            first_day.set_index("sector_code").loc["A.TI", "etf_code"], "512480.SH"
        )
        self.assertEqual(
            first_day.set_index("sector_code").loc["B.TI", "etf_code"], "511880.SH"
        )

    def test_execution_readiness_blocks_stale_mapping_and_preserves_weights(self):
        mapping = pd.DataFrame(
            {
                "asof_date": ["20260430"],
                "sector_code": ["A.TI"],
                "sector_name": ["半导体"],
                "etf_code": ["512480.SH"],
                "etf_name": ["半导体ETF"],
                "effective_from": ["20260506"],
                "effective_to": [pd.NA],
                "mapping_score": [0.9],
                "median_amount20": [200_000.0],
                "selected": [True],
            }
        )
        plan = {
            "stage": "planned",
            "market_data_asof": "20260810",
            "signal_date": "20260810",
            "planned_execution_date": "20260811",
            "target_weights": {"A.TI": 0.3, "LOW_RISK": 0.7},
        }
        payload, resolution, portfolio = build_latest_execution_readiness(
            plan,
            mapping,
            equity_quote_date="20260529",
            low_risk_quote_date="20260810",
            reference_date="20260810",
        )
        self.assertEqual(payload["overall_status"], "blocked")
        self.assertIn("mapping_stale", payload["operational_blockers"])
        self.assertIn("equity_etf_quotes_stale", payload["operational_blockers"])
        self.assertEqual(resolution.iloc[0]["final_asset_code"], "511880.SH")
        self.assertAlmostEqual(float(portfolio["final_target_weight"].sum()), 1.0)
        self.assertEqual(portfolio["etf_code"].tolist(), ["511880.SH"])

    def test_mapping_expiry_is_checked_on_planned_execution_date(self):
        mapping = pd.DataFrame(
            {
                "asof_date": ["20260731"],
                "sector_code": ["A.TI"],
                "sector_name": ["半导体"],
                "etf_code": ["512480.SH"],
                "etf_name": ["半导体ETF"],
                "effective_from": ["20260803"],
                "effective_to": [pd.NA],
                "mapping_score": [0.9],
                "median_amount20": [200_000.0],
                "selected": [True],
            }
        )
        plan = {
            "stage": "planned",
            "market_data_asof": "20260831",
            "signal_date": "20260831",
            "planned_execution_date": "20260901",
            "target_weights": {"A.TI": 0.3, "LOW_RISK": 0.7},
        }
        payload, resolution, _ = build_latest_execution_readiness(
            plan,
            mapping,
            equity_quote_date="20260831",
            low_risk_quote_date="20260831",
            reference_date="20260831",
        )
        self.assertIn("mapping_stale", payload["operational_blockers"])
        self.assertEqual(resolution.iloc[0]["final_asset_code"], "511880.SH")
