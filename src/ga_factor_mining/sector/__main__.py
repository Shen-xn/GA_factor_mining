"""板块策略的简洁运行入口。"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="运行板块轮动原型")
    parser.add_argument("--update", action="store_true", help="先增量更新行情、特征和冻结模型评分")
    parser.add_argument("--end-date", help="更新截止日，格式YYYYMMDD；默认今天")
    parser.add_argument("--token-file", help="可选Tushare token文件")
    args, product_args = parser.parse_known_args()

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
