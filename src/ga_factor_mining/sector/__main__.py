"""板块策略的简洁运行入口。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    # 统一命令行编码，避免Windows重定向输出时中文乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="运行板块轮动原型")
    parser.add_argument("--check", action="store_true", help="只检查依赖、数据和缓存，不运行策略")
    parser.add_argument("--update", action="store_true", help="先增量更新行情、特征和冻结模型评分")
    parser.add_argument("--end-date", help="更新截止日，格式YYYYMMDD；默认今天")
    parser.add_argument("--token-file", help="可选Tushare token文件")
    args, product_args = parser.parse_known_args()

    if args.check:
        from .doctor import run_preflight

        ready = run_preflight(include_update=args.update, token_file=args.token_file)
        raise SystemExit(0 if ready else 2)

    if args.update:
        from .rotation.refresh_data import refresh

        end_date = args.end_date
        if end_date is None:
            import pandas as pd

            end_date = pd.Timestamp.today().strftime("%Y%m%d")
        result = refresh(end_date, args.token_file)
        if result["updated"]:
            print(f"[update] 数据已从{result['old_end']}更新到{result['new_end']}")
        else:
            print(f"[update] 已是最新可得数据: {result['new_end']}")

    # 大面板账本和ETF回放必须分进程串行执行，避免Pandas/Numpy内存碎片累积。
    worker_env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        worker_env[name] = "1"
    # Windows下让CPython使用默认分配器；进程隔离已能完整释放Pandas内存。
    worker_env.pop("PYTHONMALLOC", None)
    python = worker_env.get("GA_FACTOR_WORKER_PYTHON", sys.executable)
    commands = (
        [python, "-X", "faulthandler", "-m", "ga_factor_mining.sector.rotation.product_backtest", *product_args],
        [python, "-X", "faulthandler", "-m", "ga_factor_mining.sector.rotation.etf_backtest"],
    )
    for label, command in zip(("product", "etf-replay"), commands, strict=True):
        completed = None
        for attempt in range(1, 4):
            completed = subprocess.run(command, check=False, env=worker_env)
            if completed.returncode == 0:
                break
            print(f"[{label}] 隔离进程异常，第{attempt}/3次")
        if completed is None or completed.returncode != 0:
            raise SystemExit(f"{label} 隔离进程连续失败")


if __name__ == "__main__":
    main()
