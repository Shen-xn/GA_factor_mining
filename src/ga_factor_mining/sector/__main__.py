"""板块策略的简洁运行入口。"""

from __future__ import annotations

import argparse
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

    # 高级研究参数仍交给产品模块，但不占用默认入口的帮助页面。
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *product_args]
        from .rotation.product_backtest import main as product_main

        product_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
