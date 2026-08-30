"""冻结产品协议并追加记录真正未见数据上的前向快照。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ...common.paths import CONFIG_ROOT, REPOSITORY_ROOT, ensure_output_dir


CONFIG_PATH = CONFIG_ROOT / "sector" / "forward_protocol.json"
DEFAULT_SOURCE_PATHS = [
    Path(__file__).with_name("run_experiments.py"),
    Path(__file__).with_name("rolling_validation.py"),
    Path(__file__).with_name("strategy.py"),
    Path(__file__).with_name("risk.py"),
    Path(__file__).with_name("low_risk.py"),
    Path(__file__).with_name("product_backtest.py"),
]


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol_payload(
    config: dict,
    model_meta: dict,
    policy_meta: dict,
    source_paths: list[Path] | None = None,
) -> dict:
    """协议只包含会改变预测或交易决策的冻结信息，不包含不断变化的数据哈希。"""
    paths = source_paths or DEFAULT_SOURCE_PATHS
    def source_key(path: Path) -> str:
        try:
            return str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
        except ValueError:
            return path.name

    return {
        "config": config,
        "model": {
            key: model_meta.get(key)
            for key in (
                "variant",
                "training_years",
                "half_life_years",
                "retrain_months",
                "score_column",
                "feature_protocol_version",
            )
        },
        "policy": {
            "strategy_policy_version": policy_meta.get("strategy_policy_version"),
            "policy_name": policy_meta.get("policy_name"),
            "policy": policy_meta.get("policy"),
            "boundary_mode": policy_meta.get("boundary_mode"),
            "low_risk_code": policy_meta.get("low_risk_code"),
        },
        "source_sha256": {
            source_key(path): _file_hash(path) for path in paths
        },
    }


def _performance(history: pd.DataFrame, freeze_data_end: str) -> dict:
    unseen = history[history["date"].astype(str).gt(freeze_data_end)].copy()
    if unseen.empty:
        return {
            "status": "awaiting_unseen_data",
            "freeze_data_end": freeze_data_end,
            "days": 0,
        }
    returns = unseen["net_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "status": "forward_evidence_available",
        "freeze_data_end": freeze_data_end,
        "start": str(unseen["date"].min()),
        "end": str(unseen["date"].max()),
        "days": int(len(unseen)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "average_daily_turnover": float(unseen["turnover"].mean()),
    }


def record_forward_snapshot(
    strategy_output_dir: Path,
    forward_dir: Path | None = None,
    config_path: Path = CONFIG_PATH,
    source_paths: list[Path] | None = None,
) -> dict:
    """同一数据截止日只能记录一个确定快照；协议变化时停止续接旧证据。"""
    forward_dir = forward_dir or ensure_output_dir("sector", "forward")
    forward_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_meta = json.loads(
        (REPOSITORY_ROOT / "outputs" / "sector" / "adaptation" / "SELECTED.json").read_text(
            encoding="utf-8"
        )
    )
    policy_meta = json.loads((strategy_output_dir / "POLICY.json").read_text(encoding="utf-8"))
    payload = build_protocol_payload(config, model_meta, policy_meta, source_paths)
    protocol_hash = _canonical_hash(payload)
    protocol_path = forward_dir / "PROTOCOL.json"
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
        if frozen["protocol_hash"] != protocol_hash:
            status = {
                "status": "protocol_mismatch",
                "recorded_at": now,
                "frozen_protocol_hash": frozen["protocol_hash"],
                "current_protocol_hash": protocol_hash,
                "message": "协议或决策代码已改变，不能把新结果静默续接到旧前向证据。",
            }
            (forward_dir / "STATUS.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return status
    else:
        frozen = {
            "created_at": now,
            "protocol_hash": protocol_hash,
            "payload": payload,
        }
        protocol_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    latest = pd.read_csv(strategy_output_dir / "LATEST_STATUS.csv").iloc[0]
    target = pd.read_csv(strategy_output_dir / "LATEST_TARGET_PORTFOLIO.csv").fillna("")
    target_records = target.sort_values(["板块代码", "目标权重"]).to_dict(orient="records")
    stable_snapshot = {
        "protocol_hash": protocol_hash,
        "data_end_date": str(latest["数据截止日"]),
        "strategy_date": str(latest["策略日期"]),
        "last_rebalance_signal_date": str(latest["最近调仓信号日"]),
        "strategy_action": str(latest["策略动作"]),
        "regime": str(latest["市场状态"]),
        "sector_exposure": str(latest["当前板块仓位"]),
        "low_risk_exposure": str(latest["当前低风险仓位"]),
        "position_count": int(latest["当前持仓数量"]),
        "target_hash": _canonical_hash({"target": target_records}),
    }
    row = {
        "recorded_at": now,
        "evidence_role": (
            "forward_unseen"
            if stable_snapshot["data_end_date"] > str(config["first_unseen_after"])
            else "freeze_baseline"
        ),
        **stable_snapshot,
        "snapshot_hash": _canonical_hash(stable_snapshot),
    }
    snapshots_path = forward_dir / "SNAPSHOTS.csv"
    if snapshots_path.exists():
        snapshots = pd.read_csv(snapshots_path, dtype=str)
        same_date = snapshots[snapshots["data_end_date"].eq(row["data_end_date"])]
        if not same_date.empty:
            if same_date.iloc[-1]["snapshot_hash"] != row["snapshot_hash"]:
                raise RuntimeError("同一数据截止日在冻结协议下产生了不同快照")
            status_path = forward_dir / "STATUS.json"
            if status_path.exists():
                return json.loads(status_path.read_text(encoding="utf-8"))
        else:
            if row["data_end_date"] <= str(snapshots["data_end_date"].max()):
                raise RuntimeError("前向快照日期必须严格递增")
            pd.DataFrame([row]).to_csv(
                snapshots_path, mode="a", header=False, index=False, encoding="utf-8-sig"
            )
    else:
        pd.DataFrame([row]).to_csv(snapshots_path, index=False, encoding="utf-8-sig")

    history = pd.read_parquet(strategy_output_dir / "HISTORY_DAILY.parquet")
    performance = {
        "protocol_hash": protocol_hash,
        **_performance(history, str(config["freeze_data_end"])),
    }
    (forward_dir / "PERFORMANCE.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = {
        "status": "recorded",
        "recorded_at": now,
        "protocol_hash": protocol_hash,
        "data_end_date": row["data_end_date"],
        "evidence_role": row["evidence_role"],
        "snapshot_count": int(len(pd.read_csv(snapshots_path))),
    }
    (forward_dir / "STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status
