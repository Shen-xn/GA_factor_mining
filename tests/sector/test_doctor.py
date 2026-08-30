import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ga_factor_mining.sector.doctor import (
    REQUIRED_DISTRIBUTIONS,
    RUNTIME_FILES,
    collect_static_checks,
)


class DoctorTests(unittest.TestCase):
    def _ready_root(self, root: Path) -> None:
        for relative_path in RUNTIME_FILES:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({"ready": True}), encoding="utf-8")
            else:
                path.write_bytes(b"ready")

    def test_ready_runtime_pack_passes_static_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._ready_root(root)
            installed = {name: "1.0" for name in REQUIRED_DISTRIBUTIONS}
            checks = collect_static_checks(root, installed=installed)
        self.assertTrue(all(item.ok for item in checks))

    def test_missing_runtime_file_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._ready_root(root)
            missing = root / "outputs/sector/adaptation/SELECTED_SCORES.parquet"
            missing.unlink()
            installed = {name: "1.0" for name in REQUIRED_DISTRIBUTIONS}
            checks = collect_static_checks(root, installed=installed)
        failed = [item for item in checks if not item.ok]
        self.assertEqual([item.name for item in failed], ["冻结模型评分"])

    def test_update_check_accepts_environment_token(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._ready_root(root)
            installed = {name: "1.0" for name in REQUIRED_DISTRIBUTIONS}
            with patch.dict(os.environ, {"TUSHARE_TOKEN": "secret"}, clear=False):
                checks = collect_static_checks(
                    root,
                    include_update=True,
                    installed=installed,
                )
        token_check = next(item for item in checks if item.name == "Tushare更新凭据")
        self.assertTrue(token_check.ok)


if __name__ == "__main__":
    unittest.main()
