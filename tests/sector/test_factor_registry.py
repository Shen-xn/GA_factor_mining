import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ga_factor_mining.sector.factor_registry import (
    build_streaming_redundancy_report,
    dependency_profiles,
    load_registry,
    validate_registry,
)


class FactorRegistryTests(unittest.TestCase):
    def test_project_registry_is_valid(self):
        ids = validate_registry(load_registry())
        self.assertEqual(len(ids), 23)

    def test_cycle_is_rejected(self):
        registry = copy.deepcopy(load_registry())
        registry["factors"][0]["dependencies"] = ["ret_3d"]
        registry["factors"][1]["dependencies"] = ["ret_1d"]
        with self.assertRaisesRegex(ValueError, "循环依赖"):
            validate_registry(registry)

    def test_dependency_profile_expands_to_raw_sources(self):
        profile = dependency_profiles(load_registry())["risk_adj_5_20"]
        self.assertEqual(profile["direct_dependencies"], ["ret_5d", "volatility_20d"])
        self.assertEqual(
            profile["factor_dependencies"],
            ["ret_1d", "ret_5d", "volatility_20d"],
        )
        self.assertEqual(profile["raw_sources"], ["close"])
        self.assertTrue(profile["derived_only_from_registered_factors"])

    def test_only_development_correlation_flags_a_pair(self):
        registry = load_registry()
        specs = {factor["factor_id"]: factor for factor in registry["factors"]}
        dates = [
            "20200101", "20200102", "20200103", "20200104",
            "20240101", "20240102", "20240103", "20240104",
            "20260101", "20260102", "20260103", "20260104",
        ]
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"trade_date": dates})
        for factor_id, spec in specs.items():
            column = spec.get("model_column", f"{factor_id}_rank")
            frame[column] = rng.normal(size=len(frame))
        frame["volume_z_20d_rank"] = [1, 2, 3, 4] * 3
        frame["turnover_z_20d_rank"] = [1, 2, 3, 4, 1, 4, 2, 3, -1, -2, -3, -4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            frame.to_parquet(path, index=False)
            report = build_streaming_redundancy_report(
                path,
                registry,
                {
                    "development": ("20200101", "20201231"),
                    "selection": ("20240101", "20241231"),
                    "observation": ("20260101", "20261231"),
                },
                sector_min_periods=3,
                context_min_periods=3,
                batch_size=5,
            )
        pair = report.loc[
            report["left_factor"].eq("volume_z_20d")
            & report["right_factor"].eq("turnover_z_20d")
        ].iloc[0]
        self.assertTrue(bool(pair["flagged_by_development"]))
        self.assertLess(float(pair["abs_correlation_selection"]), 0.8)
        self.assertAlmostEqual(float(pair["correlation_observation"]), -1.0)


if __name__ == "__main__":
    unittest.main()
