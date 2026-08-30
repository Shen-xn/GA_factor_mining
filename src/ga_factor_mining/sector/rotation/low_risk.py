#!/usr/bin/env python3
"""选择并构造真实可交易的低风险ETF收益序列。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ...common.paths import DATA_ROOT, ensure_output_dir


LOW_RISK_DATA_DIR = DATA_ROOT / "sector"
FUND_BASIC_PATH = LOW_RISK_DATA_DIR / "low_risk_fund_basic.parquet"
FUND_DAILY_PATH = LOW_RISK_DATA_DIR / "low_risk_fund_daily.parquet"
FUND_ADJ_PATH = LOW_RISK_DATA_DIR / "low_risk_fund_adj.parquet"
FUND_NAV_PATH = LOW_RISK_DATA_DIR / "low_risk_fund_nav.parquet"

# 只用2017年信息冻结：511990复权价格漏掉货币收益，511880是通过净值核验后流动性最高者。
DEFAULT_LOW_RISK_CODE = "511880.SH"
SELECTION_END = "20171231"
MIN_2017_TRADING_DAYS = 200
MIN_MEDIAN_AMOUNT_THOUSAND_RMB = 50_000.0
MAX_PRICE_NAV_GAP = 0.01


def _required_paths() -> tuple[Path, ...]:
    return FUND_BASIC_PATH, FUND_DAILY_PATH, FUND_ADJ_PATH, FUND_NAV_PATH


def low_risk_data_signature() -> str:
    """为低风险原始数据生成可复核指纹。"""
    digest = hashlib.sha256()
    for path in _required_paths():
        if not path.exists():
            raise FileNotFoundError(f"缺少低风险数据: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_adjusted_prices() -> pd.DataFrame:
    daily = pd.read_parquet(FUND_DAILY_PATH)
    adj = pd.read_parquet(FUND_ADJ_PATH)
    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if merged["adj_factor"].isna().any():
        missing = merged.loc[merged["adj_factor"].isna(), "ts_code"].value_counts().to_dict()
        raise RuntimeError(f"基金行情缺少复权因子: {missing}")
    for column in ("open", "close", "adj_factor"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["adj_open"] = merged["open"] * merged["adj_factor"]
    merged["adj_close"] = merged["close"] * merged["adj_factor"]
    return merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def build_selection_audit() -> pd.DataFrame:
    """用2017年流动性和净值一致性选择货币ETF，不查看后续收益。"""
    basic = pd.read_parquet(FUND_BASIC_PATH)
    prices = _load_adjusted_prices()
    nav = pd.read_parquet(FUND_NAV_PATH)

    cutoff = basic["list_date"].fillna("99999999").le(SELECTION_END)
    money = basic["fund_type"].eq("货币型")
    candidates = basic.loc[cutoff & money, ["ts_code", "name", "list_date", "delist_date"]]

    price_2017 = prices.loc[
        prices["trade_date"].between("20170101", SELECTION_END)
    ].groupby("ts_code", sort=False).agg(
        trading_days=("trade_date", "size"),
        median_amount=("amount", "median"),
        first_adjusted_open=("adj_open", "first"),
        last_adjusted_open=("adj_open", "last"),
    )
    price_2017["adjusted_price_return"] = (
        price_2017["last_adjusted_open"] / price_2017["first_adjusted_open"] - 1.0
    )

    nav_2017 = nav.loc[nav["nav_date"].between("20170101", SELECTION_END)].copy()
    nav_2017["adj_nav"] = pd.to_numeric(nav_2017["adj_nav"], errors="coerce")
    nav_summary = nav_2017.sort_values("nav_date").groupby("ts_code", sort=False).agg(
        first_adjusted_nav=("adj_nav", "first"),
        last_adjusted_nav=("adj_nav", "last"),
    )
    nav_summary["adjusted_nav_return"] = (
        nav_summary["last_adjusted_nav"] / nav_summary["first_adjusted_nav"] - 1.0
    )

    audit = candidates.merge(price_2017, left_on="ts_code", right_index=True, how="left")
    audit = audit.merge(nav_summary, left_on="ts_code", right_index=True, how="left")
    audit["price_nav_gap"] = (
        audit["adjusted_price_return"] - audit["adjusted_nav_return"]
    ).abs()
    audit["eligible"] = (
        audit["trading_days"].ge(MIN_2017_TRADING_DAYS)
        & audit["median_amount"].ge(MIN_MEDIAN_AMOUNT_THOUSAND_RMB)
        & audit["price_nav_gap"].le(MAX_PRICE_NAV_GAP)
    )
    eligible = audit.loc[audit["eligible"]].sort_values(
        ["median_amount", "list_date", "ts_code"], ascending=[False, True, True]
    )
    if eligible.empty:
        raise RuntimeError("2017年没有通过流动性和净值一致性门槛的货币ETF")
    selected = str(eligible.iloc[0]["ts_code"])
    audit["selected"] = audit["ts_code"].eq(selected)
    return audit.sort_values(["eligible", "median_amount"], ascending=[False, False]).reset_index(drop=True)


def build_low_risk_return_frame(
    panel: pd.DataFrame,
    code: str = DEFAULT_LOW_RISK_CODE,
) -> pd.DataFrame:
    """按板块统一交易日历构造ETF次日开盘到下一开盘的真实收益。"""
    audit = build_selection_audit()
    selected = audit.loc[audit["selected"], "ts_code"].iloc[0]
    if code != selected:
        raise ValueError(f"低风险ETF必须使用冻结选择 {selected}，收到 {code}")

    date_map = (
        panel[["trade_date", "next_open_date", "return_end_date"]]
        .drop_duplicates()
        .sort_values("trade_date")
    )
    if date_map["trade_date"].duplicated().any():
        raise RuntimeError("板块交易日历存在一日多组执行日期")

    prices = _load_adjusted_prices()
    prices = prices.loc[prices["ts_code"].eq(code)].copy()
    signal_prices = prices[["trade_date", "open", "close"]].rename(
        columns={"open": "signal_open", "close": "signal_close"}
    )
    entry_prices = prices[["trade_date", "adj_open"]].rename(
        columns={"trade_date": "next_open_date", "adj_open": "entry_adjusted_open"}
    )
    exit_prices = prices[["trade_date", "adj_open"]].rename(
        columns={"trade_date": "return_end_date", "adj_open": "exit_adjusted_open"}
    )
    result = date_map.merge(signal_prices, on="trade_date", how="left", validate="one_to_one")
    # 面板尾部可能有多条信号映射到同一退出日，价格表本身仍须一日唯一。
    result = result.merge(entry_prices, on="next_open_date", how="left", validate="many_to_one")
    result = result.merge(exit_prices, on="return_end_date", how="left", validate="many_to_one")
    result["intraday_return"] = result["signal_close"] / result["signal_open"] - 1.0
    result["forward_open_ret_1d"] = (
        result["exit_adjusted_open"] / result["entry_adjusted_open"] - 1.0
    )
    result["low_risk_code"] = code
    return result


def main() -> None:
    output_dir = ensure_output_dir("sector", "low_risk")
    audit = build_selection_audit()
    audit.to_csv(output_dir / "SELECTION_AUDIT.csv", index=False, encoding="utf-8-sig")
    selected = audit.loc[audit["selected"]].iloc[0]
    (output_dir / "SELECTED.json").write_text(
        json.dumps(
            {
                "selected_code": selected["ts_code"],
                "selected_name": selected["name"],
                "selection_period": ["20170101", SELECTION_END],
                "post_2017_returns_used_for_selection": False,
                "minimum_trading_days": MIN_2017_TRADING_DAYS,
                "minimum_median_amount_thousand_rmb": MIN_MEDIAN_AMOUNT_THOUSAND_RMB,
                "maximum_adjusted_price_nav_gap": MAX_PRICE_NAV_GAP,
                "data_signature": low_risk_data_signature(),
                "starting_capital_assumption": "backtest starts fully invested in the selected money ETF",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] selected_low_risk={selected['ts_code']}")


if __name__ == "__main__":
    main()
