"""Bind the current ARFTR report to its reviewed source, figures and renderer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "docs/ARFTR_REPORT.md",
    "tools/build_study_papers.py",
    "assets/arftr_figure_manifest.json",
    "output/pdf/arftr_report_v1.0.0.pdf",
)


def main() -> None:
    records = {}
    for name in FILES:
        raw = (ROOT / name).read_bytes()
        normalized = not name.endswith(".pdf")
        content = raw.replace(b"\r\n", b"\n") if normalized else raw
        records[name] = {"sha256": hashlib.sha256(content).hexdigest(), "normalized_lf": normalized}
    manifest = {"schema_version": 1, "project_version": "1.0.0", "artifacts": records}
    target = ROOT / "output/pdf/arftr_report_v1.0.0.manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Sealed {len(records)} report inputs/outputs")


if __name__ == "__main__":
    main()
