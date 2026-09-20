import copy

import pytest

from tools.check_project import (
    ROOT,
    check_scores,
    confusion_metrics,
    evidence,
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
    assert metadata(ROOT) == "3.1.0.dev0"


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
