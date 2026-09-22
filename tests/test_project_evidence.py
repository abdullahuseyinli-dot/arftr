import copy
import json
import shutil

import pytest

from tools import check_project
from tools.check_project import (
    LOCKED_INVENTORIES,
    ROOT,
    architecture_evidence,
    check_scores,
    confusion_metrics,
    evidence,
    historical_evidence,
    locked_json,
    metadata,
    navigation,
)


def test_public_evidence_recomputes_without_private_files():
    result = evidence(ROOT)
    assert result["models_recomputed"] == 3
    assert result["metric_rows"] == 4977
    assert not result["continuation_pass"]


def test_current_navigation_and_metadata():
    assert navigation(ROOT) > 30
    assert metadata(ROOT) == "1.0.0"


def test_confusion_checker_rejects_changed_scores():
    score = confusion_metrics([[10, 1, 0], [0, 9, 1], [1, 0, 8]])
    score.update(confusion=[[10, 1, 0], [0, 9, 1], [1, 0, 8]], support=[11, 10, 9])
    check_scores(score, 30)
    changed = copy.deepcopy(score)
    changed["macro_f1"] += 0.01
    with pytest.raises(ValueError, match="macro_f1"):
        check_scores(changed, 30)


@pytest.mark.parametrize("matrix", [[], [[0, 0, 0]] * 3, [[-1, 0, 0]] * 3])
def test_invalid_confusion_matrices_fail(matrix):
    with pytest.raises(ValueError):
        confusion_metrics(matrix)


@pytest.mark.parametrize("field", ["support", "per_class_f1", "precision", "recall"])
def test_per_class_fields_cannot_disagree_with_confusion(field):
    score = confusion_metrics([[10, 1, 0], [0, 9, 1], [1, 0, 8]])
    score.update(confusion=[[10, 1, 0], [0, 9, 1], [1, 0, 8]], support=[11, 10, 9])
    # Support still sums to the correct total, but assigns it to the wrong classes.
    score[field] = [30, 0, 0] if field == "support" else [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="population|Per-class"):
        check_scores(score, 30)


@pytest.mark.parametrize("relative", LOCKED_INVENTORIES)
def test_frozen_inventory_cannot_be_replaced_by_an_empty_one(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"artifacts": {}, "files": [], "dependencies": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Locked evidence inventory or protocol changed"):
        locked_json(tmp_path, relative)


@pytest.mark.parametrize(
    ("relative", "checker"),
    [
        ("results/arftr_development/evidence_manifest.json", evidence),
        ("results/arftr_development/architecture_manifest.json", architecture_evidence),
        ("results/human_activity_study_v3.0.0_manifest.json", historical_evidence),
    ],
)
def test_public_checker_rejects_an_omitted_manifest_entry(tmp_path, relative, checker):
    manifest = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    manifest["artifacts"].pop(next(iter(manifest["artifacts"])))
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Locked evidence inventory"):
        checker(tmp_path)


def test_development_gates_are_bound_to_the_original_protocol(tmp_path):
    folder = "results/arftr_development"
    for name in (
        "evidence_manifest.json",
        "metrics.json",
        "experiment_ledger.csv",
        "knowledge_graph.json",
    ):
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / folder / name, target)
    relative = "experiments/okutama_motion_null_contrast_protocol.json"
    protocol = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    # This lower gain requirement leaves the rejection decision unchanged, but is
    # nevertheless not the preregistered protocol and must not validate.
    protocol["continuation_gates"]["macro_f1_gain_vs_arftr_min"] = 0.0
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="Locked evidence inventory or protocol changed"):
        evidence(tmp_path)


@pytest.mark.parametrize(
    "mutation", ["score", "transition", "fold", "class", "rows", "score_field"]
)
def test_development_schema_rejects_partial_exports(monkeypatch, mutation):
    original = check_project.read_json

    def changed(path):
        result = original(path)
        if path.name == "metrics.json":
            if mutation == "score":
                result["scores"].pop("plain")
            elif mutation == "transition":
                result["transitions"].pop("plain")
            elif mutation == "fold":
                result["transitions"]["plain"]["per_fold_net"].pop()
            elif mutation == "class":
                result["classes"].reverse()
            elif mutation == "rows":
                result["rows"] -= 1
            elif mutation == "score_field":
                result["scores"]["plain"].pop("precision")
        return result

    monkeypatch.setattr(check_project, "read_json", changed)
    with pytest.raises(ValueError, match="inventory|mapping|schema"):
        evidence(ROOT)


@pytest.mark.parametrize("mutation", ["arm", "primary", "folds", "score_field"])
def test_architecture_schema_rejects_partial_exports(monkeypatch, mutation):
    original = check_project.read_json

    def changed(path):
        result = original(path)
        if path.name == "architecture_study.json":
            if mutation == "arm":
                result["scores"].pop("r0_exact_m4")
            elif mutation == "primary":
                result["primary_arm"] = "r0_exact_m4"
            elif mutation == "folds":
                result["outer_folds"] -= 1
            elif mutation == "score_field":
                result["scores"]["r0_exact_m4"].pop("rows")
        return result

    monkeypatch.setattr(check_project, "read_json", changed)
    with pytest.raises(ValueError, match="contract|schema"):
        architecture_evidence(ROOT)
