from __future__ import annotations

from pathlib import Path

import numpy as np

from hac.matr_artifacts import load_cached_study_data, selected_fold_artifacts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"


def test_selected_fold_loader_preserves_nested_identity_and_seed_contract():
    data = load_cached_study_data(SOURCE)
    selected = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, 0)
    assert np.intersect1d(selected.train_rows, selected.held_rows).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((selected.train_rows, selected.held_rows))), np.arange(4977)
    )
    assert selected.m4_inner["probabilities"].shape == (len(selected.train_rows), 3)
    assert selected.a3_inner["probabilities"].shape == (len(selected.train_rows), 3)
    assert selected.m4_outer["probabilities"].shape == (3, len(selected.held_rows), 3)
    assert selected.a3_outer["probabilities"].shape == (3, len(selected.held_rows), 3)
    assert not (set(data["scenarios"][selected.train_rows]) & set(data["scenarios"][selected.held_rows]))
