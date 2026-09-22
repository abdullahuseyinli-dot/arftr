import json
import shutil

import pytest

from tools.check_project import ROOT, report
from tools.seal_arftr_report import FILES

MANIFEST = "output/pdf/arftr_report_v1.0.0.manifest.json"


def copy_report(tmp_path):
    for name in (*FILES, MANIFEST, "pyproject.toml", "CITATION.cff", ".zenodo.json", ".gitignore"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


def test_current_report_integrity():
    assert report(ROOT) == 4


@pytest.mark.parametrize("name", FILES)
def test_report_source_and_output_changes_are_rejected(tmp_path, name):
    copy_report(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Current report source/artifact differs"):
        report(tmp_path)


@pytest.mark.parametrize("change", ["omission", "path_escape", "version", "normalization"])
def test_report_inventory_is_closed(tmp_path, change):
    copy_report(tmp_path)
    path = tmp_path / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "omission":
        manifest["artifacts"].pop(FILES[0])
    elif change == "path_escape":
        manifest["artifacts"]["../outside.md"] = manifest["artifacts"].pop(FILES[0])
    elif change == "version":
        manifest["project_version"] = "0.0.0"
    else:
        manifest["artifacts"][FILES[0]]["normalized_lf"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Current report"):
        report(tmp_path)
