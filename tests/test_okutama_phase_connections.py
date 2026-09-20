from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_okutama_phase_connections import (  # noqa: E402
    checked_archive,
    error_connection,
    require_family_binding,
)

from hac.actor_memory_base import file_sha256  # noqa: E402


def test_error_connections_account_for_all_rows_and_shared_subset():
    labels = np.asarray([0, 1, 2, 0])
    left = np.eye(3)[[0, 1, 0, 1]]
    right = np.eye(3)[[0, 0, 2, 2]]
    measured = error_connection(labels, left, right, np.asarray([False, True, True, True]))
    assert measured["both_correct"] == 1
    assert measured["left_only_correct"] == 1
    assert measured["right_only_correct"] == 1
    assert measured["both_wrong"] == 1
    assert measured["historical_shared_errors_unresolved_by_either"] == 1


def test_execution_binding_rejects_an_unrelated_same_cohort_summary(tmp_path):
    path = tmp_path / "execution.json"
    path.write_text('{"run":"current"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="intended execution"):
        require_family_binding({"lock": "another_run_hash"}, "lock", path, {})
    provenance = {}
    require_family_binding({"lock": file_sha256(path)}, "lock", path, provenance)
    assert provenance[str(path)] == file_sha256(path)


@pytest.mark.parametrize("failure", ("hash", "identity", "probabilities"))
def test_archive_rejects_changed_hash_identities_or_invalid_probabilities(tmp_path, failure):
    data = {
        "sample_ids": np.asarray(["a", "b"]),
        "labels": np.asarray([0, 1]),
        "folds": np.asarray([0, 1]),
        "scenarios": np.asarray(["x", "y"]),
    }
    arrays = {**data, "candidate": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)}
    path = tmp_path / "oof.npz"
    if failure == "identity":
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
    if failure == "probabilities":
        arrays["candidate"][0, 0] = np.nan
    np.savez(path, **arrays)
    expected = "bad" if failure == "hash" else file_sha256(path)
    with pytest.raises(RuntimeError):
        checked_archive(path, expected, data, ("candidate",))
