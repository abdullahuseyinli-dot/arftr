import shutil
from pathlib import Path

import pytest

from tools.check_project import ROOT, figures
from tools.render_arftr_figures import SOURCES, load_data, render


def test_plotted_data_preserves_scores_fold_signs_and_confusion_orientation():
    data = load_data(ROOT)
    assert data["rows"] == 4977
    assert data["history"][1]["macro_f1_percent"] == pytest.approx(85.38364807373623)
    assert data["history"][1]["macro_f1_percent"] - data["history"][0][
        "macro_f1_percent"
    ] == pytest.approx(13.45987986169339)
    assert data["fold_nets"]["plain"] == [7, 2, -2, 4, -7]
    assert data["fold_nets"]["paired_null"] == [1, 1, -1, -2, -4]
    assert [sum(row) for row in data["confusion"]] == [734, 2118, 2125]
    assert data["upright_confusions"] == 457
    assert data["errors"] == 702


def test_figure_data_rejects_changed_evidence(tmp_path):
    source = ROOT / "results/arftr_development"
    target = tmp_path / "results/arftr_development"
    target.mkdir(parents=True)
    for name in ("metrics.json", "experiment_ledger.csv", "evidence_manifest.json"):
        shutil.copyfile(source / name, target / name)
    with (target / "experiment_ledger.csv").open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="Figure source differs"):
        load_data(tmp_path)


def test_figure_render_and_integrity_checks(tmp_path):
    output = tmp_path / "assets"
    result = render(ROOT, output)
    assert result["artifacts"] == 4
    for relative in (*SOURCES, "tools/render_arftr_figures.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assert figures(tmp_path) == 4
    png = output / "arftr_development_summary.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = output / "arftr_confusion_matrix.svg"
    assert "457 of 702 errors" in svg.read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
    png.write_bytes(b"changed chart")
    with pytest.raises(ValueError, match="Figure artifact changed"):
        figures(tmp_path)


def test_checked_in_figures_are_bound_to_public_evidence():
    assert figures(Path(ROOT)) == 4
