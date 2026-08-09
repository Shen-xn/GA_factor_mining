"""将Top-50 V2的三份Markdown报告统一导出为PDF。"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import markdown


REPORTS = (
    "VALIDATION_REPORT.md",
    "TEST_2026_REPORT.md",
    "VALIDATION_TEST_GAP.md",
)

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm; }
* { box-sizing: border-box; }
body {
  margin: 0;
  color: #202124;
  font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.62;
}
h1 { margin: 0 0 15px; font-size: 22pt; color: #17324d; }
h2 { margin: 22px 0 9px; font-size: 15pt; color: #234f72; break-after: avoid; }
p { margin: 7px 0; }
blockquote {
  margin: 10px 0 16px;
  padding: 8px 12px;
  border-left: 4px solid #2d6f9f;
  background: #f2f6f9;
  color: #405466;
}
code {
  font-family: Consolas, "Microsoft YaHei", monospace;
  font-size: 9.5pt;
  color: #8b2d2d;
  background: #f5f5f5;
  padding: 1px 3px;
  border-radius: 2px;
}
table {
  width: 100%;
  margin: 8px 0 16px;
  border-collapse: collapse;
  font-size: 8.7pt;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #aeb9c2; padding: 5px 6px; vertical-align: top; }
th { background: #eaf0f4; color: #17324d; font-weight: 700; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
ul { margin: 6px 0 12px; padding-left: 22px; }
li { margin: 3px 0; }
"""


def render_html(source: Path) -> str:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=("tables", "fenced_code", "sane_lists"),
    )
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{source.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def export(source: Path) -> Path:
    target = source.with_suffix(".pdf").resolve()
    with tempfile.TemporaryDirectory(prefix="top50_v2_pdf_") as temp_dir:
        html_path = Path(temp_dir) / f"{source.stem}.html"
        html_path.write_text(render_html(source), encoding="utf-8")
        command = (
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={target}",
            html_path.resolve().as_uri(),
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not target.exists():
            raise RuntimeError(result.stderr.strip() or f"PDF导出失败: {source}")
    return target


def main() -> None:
    artifacts = Path(__file__).resolve().parent / "artifacts"
    if not EDGE.exists():
        raise FileNotFoundError(f"未找到Edge: {EDGE}")
    for name in REPORTS:
        target = export(artifacts / name)
        print(f"[pdf] {target.name} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
