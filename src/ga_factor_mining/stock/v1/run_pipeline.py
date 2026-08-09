"""统一执行遗传因子搜索与 2026 回测。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "mine", "validate", "backtest", "tree", "lightgbm",
            "lightgbm_ranker", "all",
        ),
    )
    parser.add_argument("--config", default="configs/stock/v1.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[4]

    def run_module(module: str, *extra: str) -> None:
        subprocess.run(
            [sys.executable, "-u", "-m", module, "--config", args.config, *extra],
            cwd=repository_root,
            check=True,
        )

    if args.action in ("mine", "all"):
        command = [
            sys.executable, "-u", "-m",
            "ga_factor_mining.stock.v1.build_factor_library",
            "--config", args.config,
        ]
        if args.quick:
            command.append("--quick")
        subprocess.run(command, cwd=repository_root, check=True)

    if args.action in ("validate", "all") and not args.quick:
        run_module("ga_factor_mining.stock.v1.validate_factor_library")

    if args.action in ("backtest", "all"):
        run_module("ga_factor_mining.stock.v1.backtest_2026")

    if args.action in ("tree", "all"):
        run_module("ga_factor_mining.stock.v1.backtest_tree_2026")

    if args.action in ("lightgbm", "all"):
        run_module("ga_factor_mining.stock.v1.backtest_lightgbm_2026")

    if args.action in ("lightgbm_ranker", "all"):
        run_module("ga_factor_mining.stock.v1.backtest_lightgbm_2026", "--mode", "ranker")


if __name__ == "__main__":
    main()
