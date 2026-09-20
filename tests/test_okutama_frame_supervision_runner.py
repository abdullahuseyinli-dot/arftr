from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from experiments.run_okutama_frame_supervision import (
    PROTOCOL,
    SOURCE,
    _save_or_validate_npz,
    _validate_fit_receipt_identity,
    _validate_protocol,
)
from hac.cached_frame_supervision import validate_cached_frame_supervision_receipt
from hac.matr_artifacts import load_cached_study_data
from hac.source_swap_data import read_json

ROOT = Path(__file__).resolve().parents[1]
FRAME_DATA = ROOT / ".runs/research_20260912/cached_frame_supervision_v2/data"


def _identity_fixture():
    data = {
        "sample_ids": np.array(["a", "b", "c", "d"]),
        "labels": np.array([0, 1, 2, 1]),
    }
    train, held = np.array([0, 1, 2]), np.array([3])
    from hac.actor_memory_base import canonical_hash

    receipt = {
        "status": "FSAR_FIT_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": "lock",
        "arm": "f2_center_only_supervision",
        "fold": 0,
        "seed": 42,
        "train_rows": 3,
        "held_rows": 1,
        "outer_held_labels_read": 0,
        "optimizer_steps": 8,
        "parameters": 100355,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
    }
    return data, train, held, receipt


def test_fit_receipt_rejects_wrong_arm_and_wrong_seed():
    data, train, held, receipt = _identity_fixture()
    for field, value in (("arm", "f3_all_cached_frame_supervision"), ("seed", 43)):
        changed = copy.deepcopy(receipt)
        changed[field] = value
        with pytest.raises(RuntimeError, match="request identity"):
            _validate_fit_receipt_identity(
                changed,
                arm="f2_center_only_supervision",
                fold=0,
                seed=42,
                lock_hash="lock",
                train=train,
                held=held,
                data=data,
                expected_steps=8,
                expected_parameters=100355,
            )


def test_existing_npz_must_exactly_replay_or_is_stale(tmp_path: Path):
    path = tmp_path / "artifact.npz"
    _save_or_validate_npz(path, values=np.array([1, 2, 3]))
    _save_or_validate_npz(path, values=np.array([1, 2, 3]))
    with pytest.raises(RuntimeError, match="differs"):
        _save_or_validate_npz(path, values=np.array([1, 2, 4]))
    assert not list(tmp_path.glob("*.tmp"))


def test_v2_protocol_and_metadata_receipt_validate_before_lock():
    data = load_cached_study_data(SOURCE)
    _validate_protocol(read_json(PROTOCOL), data)
    frames, receipt = validate_cached_frame_supervision_receipt(ROOT, FRAME_DATA)
    assert np.array_equal(frames["sample_ids"], data["sample_ids"])
    assert receipt["weight_audit"]["cross_partition_physical_frames"] == 0
    assert receipt["weight_audit"]["cross_partition_tracks"] == 0
    assert receipt["access_accounting"]["model_fits"] == 0
