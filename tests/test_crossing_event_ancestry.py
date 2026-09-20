import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("crossing_ancestry", Path(__file__).resolve().parents[1] / "experiments/crossing_event_ancestry.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_selection_scope_rejects_inner_or_outer_label_contamination():
    scenes = np.array(["1.1", "1.10", "1.2", "1.3"])
    audit.check_scoped_fit([0], [1], [0, 1], scenes, "labels")
    audit.check_scoped_fit([0, 1], [2, 3], [0, 1], scenes, None)
    with pytest.raises(RuntimeError, match="training rows escape"):
        audit.check_scoped_fit([0, 2], [1], [0, 1], scenes, "labels")
    with pytest.raises(RuntimeError, match="selection labels escape"):
        audit.check_scoped_fit([0, 1], [2, 3], [0, 1], scenes, "labels")


def test_scenario_identity_is_not_coerced_to_float_and_repeated_scenes_fail():
    scenes = np.array(["1.1", "1.10", "1.1"])
    audit.check_scoped_fit([0], [1], [0, 1], scenes, "labels")
    with pytest.raises(RuntimeError, match="scenario overlap"):
        audit.check_scoped_fit([0], [2], [0, 2], scenes, "labels")


def test_base_selection_and_prediction_exclusion():
    ids = np.array(["a", "b", "c", "d"])
    labels = np.array([0, 1, 2, 0])
    scenes = np.array(["1.1", "1.10", "1.2", "1.3"])
    train = np.array([0, 1])
    request = {"train_rows": train.tolist(), "all_sample_ids_sha256": audit._canonical(ids.tolist()),
               "train_sample_ids_sha256": audit._canonical(ids[train].tolist()),
               "train_labels_sha256": audit._canonical(labels[train].tolist()),
               "train_scenarios_sha256": audit._canonical(scenes[train].tolist()),
               "train_scenarios": sorted(scenes[train].tolist())}
    splits = [{"train": [0], "held": [1]}, {"train": [1], "held": [0]}]
    audit.check_base_scope(request, splits, [2, 3], [0, 1], ids, labels, scenes)
    with pytest.raises(RuntimeError, match="predicted scenario"):
        audit.check_base_scope(request, splits, [1, 2], [0, 1], ids, labels, scenes)
    with pytest.raises(RuntimeError, match="selection labels escape"):
        audit.check_base_scope(request, [{"train": [0], "held": [2]}], [2, 3], [0, 1], ids, labels, scenes)
    with pytest.raises(RuntimeError, match="exact-once"):
        audit.check_base_scope(request, splits + splits, [2, 3], [0, 1], ids, labels, scenes)
