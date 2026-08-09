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
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    if args.action in ("mine", "all"):
        command = [
            sys.executable,
            "-u",
            str(root / "build_factor_library.py"),
            "--config",
            args.config,
        ]
        if args.quick:
            command.append("--quick")
        subprocess.run(command, cwd=root, check=True)

    if args.action in ("validate", "all") and not args.quick:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "validate_factor_library.py"),
                "--config",
                args.config,
            ],
            cwd=root,
            check=True,
        )

    if args.action in ("backtest", "all"):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "backtest_2026.py"),
                "--config",
                args.config,
            ],
            cwd=root,
            check=True,
        )

    if args.action in ("tree", "all"):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "backtest_tree_2026.py"),
                "--config",
                args.config,
            ],
            cwd=root,
            check=True,
        )

    if args.action in ("lightgbm", "all"):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "backtest_lightgbm_2026.py"),
                "--config",
                args.config,
            ],
            cwd=root,
            check=True,
        )

    if args.action in ("lightgbm_ranker", "all"):
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(root / "backtest_lightgbm_2026.py"),
                "--config",
                args.config,
                "--mode",
                "ranker",
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
