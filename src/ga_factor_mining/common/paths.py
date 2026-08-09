"""统一管理仓库级数据、输出和报告目录。"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPOSITORY_ROOT / "data"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
REPORT_ROOT = REPOSITORY_ROOT / "reports"
CONFIG_ROOT = REPOSITORY_ROOT / "configs"


def ensure_project_dirs(project: str, component: str) -> tuple[Path, Path]:
    """创建并返回某条研究线隔离的机器产物和人工报告目录。"""
    output_dir = OUTPUT_ROOT / project / component
    report_dir = REPORT_ROOT / project / component
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, report_dir
