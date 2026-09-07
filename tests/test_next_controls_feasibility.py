"""Tests for the no-data next-controls feasibility contract."""

import json

import numpy as np
import pytest

from experiments.audit_next_controls_feasibility import (
    DISTINCT_SHORT_INDICES,
    LEGACY_SHORT_INDICES,
    SHORT_CENTER_SLOT,
    AccessAudit,
    t1_all_valid_mean_pool,
    temporal_control_contract,
)


def test_t1_pool_is_permutation_invariant_and_preserves_duplicate_weighting():
    sequence = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    permutation = np.asarray([7, 2, 5, 0, 6, 3, 1, 4])
    expected = sequence.mean(axis=0)
    assert np.array_equal(t1_all_valid_mean_pool(sequence), expected)
    assert np.array_equal(t1_all_valid_mean_pool(sequence[permutation]), expected)

    source = np.square(np.arange(17, dtype=np.float32))[:, None]
    sampled = source[np.asarray(LEGACY_SHORT_INDICES)]
    assert float(t1_all_valid_mean_pool(sampled)[0]) == pytest.approx(70.0)
    assert float(t1_all_valid_mean_pool(sampled)[0]) != pytest.approx(
        float(np.unique(sampled).mean())
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((7, 3), dtype=np.float32),
        np.zeros((2, 9, 3), dtype=np.float32),
        np.asarray([[np.nan]] * 8),
    ],
)
def test_t1_pool_rejects_out_of_contract_inputs(bad):
    with pytest.raises(ValueError):
        t1_all_valid_mean_pool(bad)


def test_t2_contract_is_distinct_endpoint_and_center_preserving():
    contract = temporal_control_contract()
    distinct = contract["t2"]["distinct_indices"]
    assert tuple(contract["t1"]["input_indices"]) == LEGACY_SHORT_INDICES
    assert tuple(distinct) == DISTINCT_SHORT_INDICES
    assert len(distinct) == len(set(distinct)) == 8
    assert distinct[0] == 4 and distinct[-1] == 12
    assert distinct[SHORT_CENTER_SLOT] == 8
    assert 9 not in distinct
    assert contract["t2"]["omitted_relative_offset"] == 1


def test_access_audit_rejects_manifest_and_protected_role_content(tmp_path):
    forbidden = [
        tmp_path / ".runs/vcoco_v3/temporal/development_manifest.csv",
        tmp_path / ".runs/vcoco_v3/temporal/calibration/summary.json",
        tmp_path / ".runs/vcoco_v3/temporal/confirmation/summary.json",
        tmp_path / ".runs/polar_v2/locked_protocol/vcoco_test_clean.csv",
    ]
    audit = AccessAudit(tmp_path)
    for path in forbidden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"secret": True}), encoding="utf-8")
        with pytest.raises(PermissionError):
            audit.read_json(path, purpose="must fail")
    assert audit.reads == []


def test_access_audit_rejects_paths_outside_repository(tmp_path):
    audit = AccessAudit(tmp_path / "repository")
    outside = tmp_path / "summary.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        audit.read_json(outside, purpose="must fail")


def test_access_audit_accepts_only_declared_development_checkpoint_paths(tmp_path):
    directory = tmp_path / ".runs/cptr/crossfit/centre_short_parts/fold-0/seed-42"
    directory.mkdir(parents=True)
    summary = directory / "summary.json"
    checkpoint = directory / "checkpoint.pt"
    summary.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"development checkpoint")
    audit = AccessAudit(tmp_path)
    assert audit.read_json(summary, purpose="receipt") == {}
    assert audit.hash_binary(checkpoint, purpose="identity") == (
        "d8ee192ee3c8090f39832ac3433bf5e27950f7dc5d096943fd13cfd339e22cdc"
    )
