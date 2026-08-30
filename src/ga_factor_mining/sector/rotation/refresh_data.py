#!/usr/bin/env python3
"""低内存增量更新板块原型所需的行情、特征和冻结模型评分。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...common.paths import REPOSITORY_ROOT
from .low_risk import DEFAULT_LOW_RISK_CODE, low_risk_data_signature
from .rolling_validation import FEATURE_COLS, make_lgbm_model
from .run_experiments import (
    DATA_DIR,
    FEATURE_PATH,
    MARKET_CONTEXT_COLS,
    UNIVERSES,
    build_feature_frame,
    current_feature_cache_signature,
    write_feature_metadata,
)


INDEX_PATH = DATA_DIR / "ths_index.parquet"
SECTOR_DAILY_PATH = DATA_DIR / "ths_daily.parquet"
LOW_RISK_DAILY_PATH = DATA_DIR / "low_risk_fund_daily.parquet"
LOW_RISK_ADJ_PATH = DATA_DIR / "low_risk_fund_adj.parquet"
SELECTED_DIR = REPOSITORY_ROOT / "outputs" / "sector" / "adaptation"
SELECTED_SCORE_PATH = SELECTED_DIR / "SELECTED_SCORES.parquet"
SELECTED_META_PATH = SELECTED_DIR / "SELECTED.json"


def _max_date(path: Path, column: str = "trade_date") -> str:
    values = pq.read_table(path, columns=[column]).column(column)
    value = pc.max(values).as_py()
    if value is None:
        raise ValueError(f"{path} 没有日期")
    return str(value)


def _token_from(token_file: str | None) -> str:
    if token := os.environ.get("TUSHARE_TOKEN"):
        return token.strip()
    candidates = []
    if token_file:
        candidates.append(Path(token_file))
    candidates.extend(
        [
            REPOSITORY_ROOT / "tushare_token.txt",
            Path(r"D:\Users\s1171\qt_MLE\tushare_token.txt"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise RuntimeError("未找到Tushare token；请设置TUSHARE_TOKEN或传入--token-file")


def _query(callable_, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return callable_(**kwargs)
        except Exception as exc:  # 网络波动只做有限重试，不隐藏最终错误。
            last_error = exc
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(f"Tushare请求失败: {last_error}") from last_error


def _table_for_schema(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    missing = set(schema.names) - set(frame.columns)
    if missing:
        raise ValueError(f"增量数据缺少字段: {sorted(missing)}")
    table = pa.Table.from_pandas(frame[schema.names], preserve_index=False)
    return table.cast(schema, safe=False)


def _append_parquet(source: Path, new_rows: pd.DataFrame, destination: Path) -> None:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    with pq.ParquetWriter(destination, schema, compression="snappy") as writer:
        for batch in parquet.iter_batches(batch_size=100_000):
            writer.write_batch(batch)
        if not new_rows.empty:
            table = _table_for_schema(new_rows, schema)
            writer.write_table(table, row_group_size=100_000)


def _replace_parquet_tail(
    source: Path,
    tail: pd.DataFrame,
    cutoff: str,
    destination: Path,
) -> None:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    with pq.ParquetWriter(destination, schema, compression="snappy") as writer:
        for batch in parquet.iter_batches(batch_size=100_000):
            mask = pc.less(batch.column(batch.schema.get_field_index("trade_date")), cutoff)
            kept = pa.Table.from_batches([batch]).filter(mask)
            if kept.num_rows:
                writer.write_table(kept)
        table = _table_for_schema(tail, schema)
        writer.write_table(table, row_group_size=100_000)


def _fetch_sector_updates(pro, dates: list[str]) -> pd.DataFrame:
    rows = []
    for number, trade_date in enumerate(dates, start=1):
        frame = _query(pro.ths_daily, trade_date=trade_date)
        if not frame.empty:
            rows.append(frame)
        if number % 10 == 0 or number == len(dates):
            print(f"[download] 板块行情 {number}/{len(dates)}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["ts_code", "trade_date"], keep="last").sort_values(
        ["ts_code", "trade_date"]
    )


def _updated_index(pro, old_end: str) -> pd.DataFrame:
    current = pd.read_parquet(INDEX_PATH)
    latest = pd.concat(
        [_query(pro.ths_index, exchange="A", type=kind) for kind in UNIVERSES["industry_concept"]],
        ignore_index=True,
    )
    # 只补原数据截止日以后上市的新板块，避免用当前目录重写历史分类。
    additions = latest[
        (~latest["ts_code"].isin(current["ts_code"])) & (latest["list_date"].astype(str) > old_end)
    ]
    return pd.concat([current, additions], ignore_index=True).drop_duplicates("ts_code", keep="first")


def _calendar_and_dates(pro, old_end: str, requested_end: str) -> tuple[list[str], list[str]]:
    calendar = _query(
        pro.trade_cal,
        exchange="SSE",
        start_date="20240101",
        end_date=requested_end,
        is_open="1",
    )
    open_dates = sorted(calendar["cal_date"].astype(str).unique().tolist())
    return open_dates, [date for date in open_dates if date > old_end]


def _validate_overlap(tail: pd.DataFrame, cutoff: str, old_end: str) -> None:
    model_cols = [
        "ts_code", "trade_date", "type", "open", "close", *FEATURE_COLS, *MARKET_CONTEXT_COLS
    ]
    old = pd.read_parquet(
        FEATURE_PATH,
        columns=model_cols,
        filters=[("trade_date", ">=", cutoff), ("trade_date", "<=", old_end)],
    )
    new = tail.loc[tail["trade_date"].le(old_end), model_cols]
    old_keys = old[["ts_code", "trade_date"]].sort_values(["ts_code", "trade_date"])
    new_keys = new[["ts_code", "trade_date"]].sort_values(["ts_code", "trade_date"])
    if not old_keys.reset_index(drop=True).equals(new_keys.reset_index(drop=True)):
        raise RuntimeError("增量特征与旧缓存的重叠键不一致，已停止替换")
    merged = old.merge(new, on=["ts_code", "trade_date"], suffixes=("_old", "_new"))
    for column in ["open", "close", *FEATURE_COLS, *MARKET_CONTEXT_COLS]:
        # 滚动方差在不同预热起点可能产生极小浮点差，最多造成横截面约两个名次变化。
        tolerance = 0.0021 if column.endswith("_rank") else 2e-6
        if not np.allclose(
            merged[f"{column}_old"],
            merged[f"{column}_new"],
            equal_nan=True,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise RuntimeError(f"增量特征重叠校验失败: {column}")


def _prediction_windows(
    score_start: str,
    data_end: str,
    retrain_months: int,
) -> list[tuple[str, str, str]]:
    """返回新增评分使用的(train_end, predict_start, predict_end)。"""
    if retrain_months <= 0 or 12 % retrain_months:
        raise ValueError("retrain_months必须是12的正因数")
    first_new = pd.Timestamp(score_start) + pd.Timedelta(days=1)
    final_end = pd.Timestamp(data_end)
    if first_new > final_end:
        return []
    anchor_month = ((first_new.month - 1) // retrain_months) * retrain_months + 1
    window_start = pd.Timestamp(first_new.year, anchor_month, 1)
    windows = []
    while window_start <= final_end:
        window_end = window_start + pd.DateOffset(months=retrain_months) - pd.Timedelta(days=1)
        predict_start = max(first_new, window_start)
        predict_end = min(final_end, window_end)
        windows.append(
            (
                (window_start - pd.Timedelta(days=1)).strftime("%Y%m%d"),
                predict_start.strftime("%Y%m%d"),
                predict_end.strftime("%Y%m%d"),
            )
        )
        window_start = window_start + pd.DateOffset(months=retrain_months)
    return windows


def _new_scores(
    feature_path: Path,
    score_start: str,
    data_end: str,
    score_name: str,
    retrain_months: int,
) -> pd.DataFrame:
    """按冻结重训频率生成新增评分；每段只使用段首前已兑现标签。"""
    train_cols = [
        "ts_code", "trade_date", "type", "future_ret_5d_rank", "future_ret_5d_end_date", *FEATURE_COLS
    ]
    score_frames = []
    for train_end, predict_start, predict_end in _prediction_windows(
        score_start, data_end, retrain_months
    ):
        train_panel = pd.read_parquet(
            feature_path,
            columns=list(dict.fromkeys(train_cols)),
            filters=[("trade_date", "<=", train_end)],
        )
        train_panel = train_panel[
            train_panel["type"].isin(UNIVERSES["industry_concept"])
        ]
        train = train_panel[
            train_panel["future_ret_5d_end_date"].le(train_end)
            & train_panel["future_ret_5d_rank"].notna()
        ][["trade_date", *FEATURE_COLS, "future_ret_5d_rank"]]
        train = train.replace([np.inf, -np.inf], np.nan).dropna()
        model = make_lgbm_model(5).set_params(n_jobs=4)
        model.fit(train[FEATURE_COLS], train["future_ret_5d_rank"])
        del train, train_panel
        gc.collect()

        pred = pd.read_parquet(
            feature_path,
            columns=["ts_code", "trade_date", "type", *FEATURE_COLS],
            filters=[
                ("trade_date", ">=", predict_start),
                ("trade_date", "<=", predict_end),
            ],
        )
        pred = pred[pred["type"].isin(UNIVERSES["industry_concept"])].copy()
        values = pred[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        pred[score_name] = model.predict(values).astype("float32")
        score_frames.append(pred[["ts_code", "trade_date", score_name]])
        del pred, values, model
        gc.collect()
    if not score_frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", score_name])
    return pd.concat(score_frames, ignore_index=True).sort_values(["ts_code", "trade_date"])


def _transactional_replace(replacements: list[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    try:
        for target, staged in replacements:
            backup = target.with_name(f".{target.name}.refresh-backup")
            if backup.exists():
                raise RuntimeError(f"发现未清理的更新备份: {backup}")
            os.replace(target, backup)
            backups.append((target, backup))
            os.replace(staged, target)
    except Exception:
        for target, backup in reversed(backups):
            if target.exists():
                target.unlink()
            if backup.exists():
                os.replace(backup, target)
        raise
    for _, backup in backups:
        backup.unlink(missing_ok=True)


def refresh(end_date: str, token_file: str | None = None) -> dict:
    import tushare as ts

    old_end = _max_date(SECTOR_DAILY_PATH)
    if end_date <= old_end:
        return {"updated": False, "old_end": old_end, "new_end": old_end, "new_dates": 0}
    pro = ts.pro_api(_token_from(token_file))
    open_dates, dates = _calendar_and_dates(pro, old_end, end_date)
    if not dates:
        return {"updated": False, "old_end": old_end, "new_end": old_end, "new_dates": 0}

    sector_updates = _fetch_sector_updates(pro, dates)
    if sector_updates.empty:
        return {"updated": False, "old_end": old_end, "new_end": old_end, "new_dates": 0}
    new_end = str(sector_updates["trade_date"].max())
    dates = [date for date in dates if date <= new_end]
    low_daily = _query(
        pro.fund_daily, ts_code=DEFAULT_LOW_RISK_CODE, start_date=dates[0], end_date=new_end
    ).sort_values(["ts_code", "trade_date"])
    low_adj = _query(
        pro.fund_adj, ts_code=DEFAULT_LOW_RISK_CODE, start_date=dates[0], end_date=new_end
    ).sort_values(["ts_code", "trade_date"])
    if set(dates) - set(low_daily["trade_date"].astype(str)):
        raise RuntimeError("低风险ETF行情未覆盖全部新增交易日")
    if set(dates) - set(low_adj["trade_date"].astype(str)):
        raise RuntimeError("低风险ETF复权因子未覆盖全部新增交易日")

    old_score_end = _max_date(SELECTED_SCORE_PATH)
    old_meta = json.loads(SELECTED_META_PATH.read_text(encoding="utf-8"))
    retrain_months = int(old_meta.get("retrain_months", 0))
    supported_model = (
        old_meta.get("variant") in {"expanding", "expanding_quarterly"}
        and retrain_months in {3, 12}
    )
    if not supported_model:
        raise RuntimeError("日常增量更新仅支持冻结的年度或季度扩展窗口模型")

    old_end_index = open_dates.index(old_end)
    replace_index = max(0, old_end_index - 15)
    warmup_index = max(0, replace_index - 400)
    replace_start = open_dates[replace_index]
    warmup_start = open_dates[warmup_index]

    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sector-refresh-", dir=temp_root) as folder:
        staging = Path(folder)
        staged_sector = staging / SECTOR_DAILY_PATH.name
        staged_index = staging / INDEX_PATH.name
        staged_low_daily = staging / LOW_RISK_DAILY_PATH.name
        staged_low_adj = staging / LOW_RISK_ADJ_PATH.name
        staged_feature = staging / FEATURE_PATH.name
        staged_scores = staging / SELECTED_SCORE_PATH.name

        _append_parquet(SECTOR_DAILY_PATH, sector_updates, staged_sector)
        _append_parquet(LOW_RISK_DAILY_PATH, low_daily, staged_low_daily)
        _append_parquet(LOW_RISK_ADJ_PATH, low_adj, staged_low_adj)
        updated_index = _updated_index(pro, old_end)
        updated_index.to_parquet(staged_index, index=False)

        print(f"[features] 尾部重算 {warmup_start} 起，替换 {replace_start} 起")
        raw_tail = pd.read_parquet(staged_sector, filters=[("trade_date", ">=", warmup_start)])
        feature_tail = build_feature_frame(raw_tail, updated_index)
        feature_tail = feature_tail[feature_tail["trade_date"].ge(replace_start)].copy()
        _validate_overlap(feature_tail, replace_start, old_end)
        _replace_parquet_tail(FEATURE_PATH, feature_tail, replace_start, staged_feature)
        del raw_tail, feature_tail
        gc.collect()

        print(f"[model] 使用冻结超参数按{retrain_months}个月频率预测新增日期")
        score_rows = _new_scores(
            staged_feature,
            old_score_end,
            new_end,
            str(old_meta["score_column"]),
            retrain_months,
        )
        if score_rows.empty or str(score_rows["trade_date"].max()) != new_end:
            raise RuntimeError("新增评分未覆盖最新行情日")
        _append_parquet(SELECTED_SCORE_PATH, score_rows, staged_scores)
        del score_rows
        gc.collect()

        _transactional_replace(
            [
                (SECTOR_DAILY_PATH, staged_sector),
                (INDEX_PATH, staged_index),
                (LOW_RISK_DAILY_PATH, staged_low_daily),
                (LOW_RISK_ADJ_PATH, staged_low_adj),
                (FEATURE_PATH, staged_feature),
                (SELECTED_SCORE_PATH, staged_scores),
            ]
        )

    write_feature_metadata(list(pq.ParquetFile(FEATURE_PATH).schema_arrow.names))
    old_meta.update(
        {
            "feature_cache_signature": current_feature_cache_signature(),
            "low_risk_data_signature": low_risk_data_signature(),
            "score_data_end": new_end,
            "last_incremental_refresh": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    SELECTED_META_PATH.write_text(
        json.dumps(old_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "updated": True,
        "old_end": old_end,
        "new_end": new_end,
        "new_dates": len(dates),
        "replace_start": replace_start,
        "warmup_start": warmup_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="增量更新板块原型数据（不会运行GA或参数搜索）")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--token-file")
    args = parser.parse_args()
    result = refresh(args.end_date, args.token_file)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
