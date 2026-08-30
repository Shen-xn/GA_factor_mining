"""统一管理仓库级数据和结构化输出目录。"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPOSITORY_ROOT / "data"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
CONFIG_ROOT = REPOSITORY_ROOT / "configs"


def ensure_output_dir(project: str, component: str) -> Path:
    """创建并返回某条研究线隔离的结构化输出目录。"""
    output_dir = OUTPUT_ROOT / project / component
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
