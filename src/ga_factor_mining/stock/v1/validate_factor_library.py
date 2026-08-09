"""在全部 2026 年前横截面上复核遗传因子库。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_pipeline import load_config, prepare_data
from .factor_engine import evaluate, fast_daily_rank_ic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stock/v1.json")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    data, _ = prepare_data(config)
    target = config["target"]["name"]
    end = config["split"]["factor_library_end"]
    data = data[(data["trade_date"] <= end) & data[target].notna()].copy()
    data["_date_code"] = pd.factorize(data["trade_date"], sort=False)[0]
    data["_target_rank"] = data.groupby("_date_code", sort=False)[target].rank(pct=True)

    output_dir = Path(config["data"]["output_dir"])
    library_path = output_dir / "factor_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    monthly_rows = []
    for number, item in enumerate(library["factors"], start=1):
        values = evaluate(
            item["expression"], data,
            float(config["factor_mining"]["max_abs_value"]),
        )
        daily = fast_daily_rank_ic(
            values, data, int(config["target"]["min_daily_stocks"])
        ).dropna()
        monthly = daily.groupby(daily.index.str[:6]).agg(["mean", "count"])
        monthly = monthly[
            monthly["count"] >= int(config["factor_mining"]["min_month_days"])
        ]
        top = monthly["mean"].abs().nlargest(3)
        item["full_peak_month"] = str(top.index[0])
        item["full_best_month_abs_ic"] = float(top.iloc[0])
        item["full_top3_month_abs_ic"] = float(top.mean())
        item["full_mean_abs_month_ic"] = float(monthly["mean"].abs().mean())
        item["full_months_abs_ic_ge_005"] = int((monthly["mean"].abs() >= 0.05).sum())
        for month, row in monthly.iterrows():
            monthly_rows.append({
                "factor": item["name"], "month": month,
                "mean_rank_ic": row["mean"], "valid_days": int(row["count"]),
            })
        print(
            f"[validate] {number:02d}/{len(library['factors'])} {item['name']} "
            f"peak={item['full_peak_month']} abs_ic={item['full_best_month_abs_ic']:.4f}"
        )

    library["full_history_validated"] = True
    library_path.write_text(
        json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [key for key in library["factors"][0] if key != "expression"]
    pd.DataFrame(library["factors"])[columns].to_csv(
        output_dir / "factor_library.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(monthly_rows).to_csv(
        output_dir / "factor_monthly_ic_full_history.csv",
        index=False, encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
