"""检查板块策略在当前机器上是否具备可运行条件。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ..common.paths import REPOSITORY_ROOT


REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "joblib",
    "lightgbm",
    "tushare",
)

RUNTIME_FILES = {
    "data/sector/ths_daily.parquet": "板块日行情",
    "data/sector/ths_index.parquet": "板块目录",
    "data/sector/low_risk_fund_basic.parquet": "低风险基金目录",
    "data/sector/low_risk_fund_daily.parquet": "低风险基金日行情",
    "data/sector/low_risk_fund_adj.parquet": "低风险基金复权因子",
    "data/sector/low_risk_fund_nav.parquet": "低风险基金净值",
    "outputs/sector/rotation/sector_feature_panel.parquet": "特征缓存",
    "outputs/sector/rotation/sector_feature_panel.meta.json": "特征缓存元数据",
    "outputs/sector/adaptation/SELECTED_SCORES.parquet": "冻结模型评分",
    "outputs/sector/adaptation/SELECTED.json": "冻结模型元数据",
}


@dataclass(frozen=True)
class CheckItem:
    """一项可展示的运行前检查。"""

    name: str
    ok: bool
    detail: str


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return installed


def _token_available(repository_root: Path, token_file: str | None) -> tuple[bool, str]:
    if os.environ.get("TUSHARE_TOKEN", "").strip():
        return True, "环境变量 TUSHARE_TOKEN"

    candidates: list[Path] = []
    if token_file:
        candidates.append(Path(token_file))
    if env_file := os.environ.get("TUSHARE_TOKEN_FILE"):
        candidates.append(Path(env_file))
    candidates.append(repository_root / "tushare_token.txt")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return True, str(path)
    return False, "设置 TUSHARE_TOKEN、TUSHARE_TOKEN_FILE 或传入 --token-file"


def collect_static_checks(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    include_update: bool = False,
    token_file: str | None = None,
    installed: dict[str, str] | None = None,
) -> list[CheckItem]:
    """检查依赖、文件和更新凭据，不加载大型Parquet。"""
    installed = _installed_distributions() if installed is None else installed
    checks = [
        CheckItem(
            f"依赖 {name}",
            name in installed,
            installed.get(name, "未安装；运行 python -m pip install -e ."),
        )
        for name in REQUIRED_DISTRIBUTIONS
    ]

    for relative_path, role in RUNTIME_FILES.items():
        path = repository_root / relative_path
        checks.append(
            CheckItem(
                role,
                path.is_file() and path.stat().st_size > 0,
                relative_path,
            )
        )

    for relative_path in (
        "outputs/sector/rotation/sector_feature_panel.meta.json",
        "outputs/sector/adaptation/SELECTED.json",
    ):
        path = repository_root / relative_path
        if not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(CheckItem("JSON元数据", False, f"{relative_path}: {exc}"))

    if include_update:
        token_ok, token_detail = _token_available(repository_root, token_file)
        checks.append(CheckItem("Tushare更新凭据", token_ok, token_detail))
    return checks


def _deep_integrity_checks() -> list[CheckItem]:
    """在静态检查通过后核对特征、评分和低风险数据指纹。"""
    checks: list[CheckItem] = []
    try:
        from .rotation.run_experiments import current_feature_cache_signature

        feature_signature = current_feature_cache_signature()
        checks.append(CheckItem("特征缓存指纹", True, feature_signature[:12]))
    except Exception as exc:
        return [CheckItem("特征缓存指纹", False, str(exc))]

    try:
        selected_path = REPOSITORY_ROOT / "outputs" / "sector" / "adaptation" / "SELECTED.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        score_ok = selected.get("feature_cache_signature") == feature_signature
        checks.append(
            CheckItem(
                "冻结评分与特征一致",
                score_ok,
                "协议一致" if score_ok else "请重新生成或取得同版本 SELECTED_SCORES.parquet",
            )
        )
    except Exception as exc:
        checks.append(CheckItem("冻结评分与特征一致", False, str(exc)))

    try:
        from .rotation.low_risk import low_risk_data_signature

        checks.append(CheckItem("低风险数据指纹", True, low_risk_data_signature()[:12]))
    except Exception as exc:
        checks.append(CheckItem("低风险数据指纹", False, str(exc)))
    return checks


def run_preflight(*, include_update: bool = False, token_file: str | None = None) -> bool:
    """打印面向用户的检查结果，并返回是否可以执行请求的流程。"""
    checks = collect_static_checks(include_update=include_update, token_file=token_file)
    if all(item.ok for item in checks):
        checks.extend(_deep_integrity_checks())

    for item in checks:
        marker = "OK" if item.ok else "缺失"
        print(f"[{marker}] {item.name}: {item.detail}")

    ok = all(item.ok for item in checks)
    if ok:
        mode = "更新并回放" if include_update else "正式回放"
        print(f"[ready] 当前环境可以执行{mode}。")
    else:
        print("[blocked] 请按上面的缺失项补齐；数据文件说明见 docs/DATA_CONTRACT.md。")
    return ok
