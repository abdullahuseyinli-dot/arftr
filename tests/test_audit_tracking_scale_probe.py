from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "independent_scale_audit",
    Path(__file__).resolve().parents[1] / "experiments/audit_okutama_tracking_scale_probe.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(n=64, t=61, center=30):
    local = np.broadcast_to(np.array([20.0, 25.0]), (t, n, 2)).copy()
    valid = np.ones((t, n), dtype=bool)
    cycle = np.full((9, n), 0.1)
    cycle[4] = np.nan
    return {
        "positions": local.copy(),
        "valid": valid.copy(),
        "local_positions": local.copy(),
        "local_valid": valid.copy(),
        "times_seconds": np.arange(t, dtype=float) - center,
        "point_ids": np.array([f"actor-{i:03d}" for i in range(n)]),
        "cycle_error": cycle,
        "cycle_valid": np.isfinite(cycle),
        "local_cycle_error": cycle.copy(),
        "local_cycle_valid": np.isfinite(cycle),
        "center_index": np.array(center),
        "sampled_indices": audit.SAMPLED.copy() if t == 61 else np.arange(9),
        "seeds_native": local[center].copy(),
        "camera_transforms": np.broadcast_to(np.eye(3), (t, 3, 3)).copy(),
        "crop": np.array([-1] * 4),
        "native_shape": np.array([2160, 3840]),
        "coordinate_frame": np.array("native"),
    }


def test_missing_return_cycles_never_receive_zero_error_credit():
    saved = fixture()
    metrics = audit.reconstruct_metrics(saved, 100.0)
    assert metrics["cycle_median_normalized"] == 0.001
    assert audit.sample_pass(metrics)
    saved["cycle_error"][:] = np.nan
    saved["cycle_valid"][:] = False
    saved["cycle_error"][0, 0] = 0.0
    saved["cycle_valid"][0, 0] = True
    metrics = audit.reconstruct_metrics(saved, 100.0)
    assert metrics["cycle_missing_pairs"] == 511
    assert metrics["actor_joint_forward_reverse_survival"] == 0.0
    assert not audit.sample_pass(metrics)


@pytest.mark.parametrize("count", [35, 39])
def test_variable_point_diagnostics_cannot_become_primary(count):
    assert not audit.check_primary_eligibility(count, False, count)
    with pytest.raises(RuntimeError, match="eligibility"):
        audit.check_primary_eligibility(count, True, count)
    with pytest.raises(RuntimeError, match="budget"):
        audit.check_primary_eligibility(64, True, count)


def test_center_cycles_and_false_masks_are_rejected():
    saved = fixture()
    saved["cycle_valid"][4, 0] = True
    saved["cycle_error"][4, 0] = 0.0
    with pytest.raises(RuntimeError, match="center contributes"):
        audit.reconstruct_metrics(saved, 10)
    saved = fixture()
    saved["cycle_valid"][0, 0] = False
    with pytest.raises(RuntimeError, match="missingness"):
        audit.reconstruct_metrics(saved, 10)


def test_camera_inverse_warp_and_native_bounds_are_independently_replayed():
    saved = fixture(n=3)
    saved["coordinate_frame"] = np.array("camera_crop")
    saved["crop"] = np.array([100, 150, 400, 300])
    saved["camera_transforms"][:, 0, 2] = np.arange(61) - 30
    saved["positions"] = saved["local_positions"] + np.array([100, 150])
    saved["positions"][:, :, 0] += (np.arange(61) - 30)[:, None]
    assert audit.reconstruct_native_mapping(saved) == "camera_crop"
    wrong = copy.deepcopy(saved)
    wrong["positions"][0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="inverse warp"):
        audit.reconstruct_native_mapping(wrong)
    wrong = copy.deepcopy(saved)
    wrong["local_positions"][0, 0, 0] = 400
    with pytest.raises(RuntimeError, match="validity/bounds"):
        audit.reconstruct_native_mapping(wrong)


def test_paired_crop_identity_cannot_silently_change():
    full = fixture()
    crop = copy.deepcopy(full)
    crop["crop"] = np.array([10, 20, 300, 200])
    crop["coordinate_frame"] = np.array("camera_crop")
    arrays = {
        audit.ARMS[0]: full,
        audit.ARMS[1]: copy.deepcopy(crop),
        audit.ARMS[2]: copy.deepcopy(crop),
    }
    audit.compare_paired_geometry(arrays)
    arrays[audit.ARMS[2]]["seeds_native"][0, 0] += 0.1
    with pytest.raises(RuntimeError, match="seeds_native"):
        audit.compare_paired_geometry(arrays)


def test_camera_fit_uses_only_even_background_identities():
    grid = np.column_stack((np.arange(64) % 8 * 10.0 + 20, np.arange(64) // 8 * 10.0 + 30))
    positions = np.broadcast_to(grid, (61, 64, 2)).copy()
    delta = np.column_stack((np.arange(61) - 30, (np.arange(61) - 30) * 0.2))
    positions += delta[:, None]
    transforms = np.broadcast_to(np.eye(3), (61, 3, 3)).copy()
    transforms[:, :2, 2] = delta * 3
    saved = {
        "background_positions720": positions,
        "background_valid": np.ones((61, 64), bool),
        "fit_ids": np.arange(0, 64, 2),
        "held_ids": np.arange(1, 64, 2),
        "camera_transforms": transforms,
    }
    assert audit.reconstruct_camera(saved)["held_median_error720"] < 1e-8
    saved["background_positions720"][:30, 1::2] += 20
    assert audit.reconstruct_camera(saved)["held_median_error720"] > 1
    saved["fit_ids"] = np.arange(1, 64, 2)
    with pytest.raises(RuntimeError, match="fit/held"):
        audit.reconstruct_camera(saved)


def test_synthetic_rigid_only_error_cannot_pass_nonrigid_recovery():
    from experiments.tracking_scale_synthetic import synthetic_scale_case

    truth = synthetic_scale_case("typical", "nonrigid", render=False)
    saved = fixture(t=9, center=4)
    saved["positions"] = truth["actor_truth"] - truth["articulation_truth"]
    saved["camera_transforms"] = truth["camera_truth"]
    saved["seeds_native"] = truth["actor_truth"][4].copy()
    saved["times_seconds"] = truth["times_seconds"]
    truth["point_ids"] = np.asarray(truth["point_ids"])
    result = audit.reconstruct_synthetic(saved, truth, True)
    assert result["nonrigid_known_truth_relative_rms"] == pytest.approx(1.0)
    assert not result["pass"]


@pytest.mark.parametrize("size", audit.SIZES)
@pytest.mark.parametrize("case", audit.CASES)
def test_all_synthetic_truth_score_fields_match_independent_algebra(size, case):
    from experiments.tracking_scale_synthetic import (
        score_synthetic_scale_tracks,
        synthetic_scale_case,
    )
    from hac.persistent_articulation_v2 import TrackSequence

    truth = synthetic_scale_case(size, case, render=False)
    saved = fixture(t=9, center=4)
    saved["positions"] = truth["actor_truth"].copy()
    saved["camera_transforms"] = truth["camera_truth"]
    saved["seeds_native"] = truth["actor_truth"][4].copy()
    saved["times_seconds"] = truth["times_seconds"]
    truth["point_ids"] = np.asarray(truth["point_ids"])
    expected = audit.reconstruct_synthetic(saved, truth, case == "nonrigid")
    sequence = TrackSequence(
        saved["positions"],
        saved["valid"],
        saved["times_seconds"],
        tuple(saved["point_ids"].tolist()),
    )
    produced = score_synthetic_scale_tracks(
        sequence, truth, saved["cycle_error"], saved["cycle_valid"]
    )
    assert expected["pass"]
    for key in expected:
        audit.compare(produced[key], expected[key], key)


def test_checksum_record_detects_changed_artifact(tmp_path):
    path = tmp_path / "array.npz"
    path.write_bytes(b"first")
    first = audit.record(path, root=tmp_path)
    path.write_bytes(b"second")
    assert audit.record(path, root=tmp_path)["sha256"] != first["sha256"]


def test_synthetic_stop_rejects_missing_cases_extra_outputs_and_real_inference():
    names = ["typical_camera_only", "typical_actor_only"]
    aggregate = {
        "pass": False,
        "stopped_arm": audit.ARMS[2],
        "stopped_case": names[-1],
        "cases": {name: {} for name in names},
    }
    files = [f"synthetic_truth_{name}.npz" for name in names]
    files += [f"synthetic_{name}_{arm}.npz" for name in names for arm in audit.ARMS]
    assert audit.validate_stop_prefix(aggregate, files, []) == names
    with pytest.raises(RuntimeError, match="ordered prefix"):
        audit.validate_stop_prefix(aggregate, files[:-1], [])
    with pytest.raises(RuntimeError, match="ordered prefix"):
        audit.validate_stop_prefix(aggregate, files + ["synthetic_truth_tiny_nonrigid.npz"], [])
    with pytest.raises(RuntimeError, match="real tracking"):
        audit.validate_stop_prefix(aggregate, files, ["clip_00_crop_native_detail.npz"])
    aggregate["cases"].pop(names[0])
    with pytest.raises(RuntimeError, match="omissions"):
        audit.validate_stop_prefix(aggregate, files, [])


def test_final_assessment_matches_producer_and_never_credits_low_point_centers():
    from experiments.run_okutama_tracking_scale_probe import assess

    population, reports, rows = [], [], []
    for i in range(16):
        eligible = i not in (3, 10)
        pop = {
            "selection_index": i,
            "scenario": "1.10" if i < 12 else "1.11",
            "primary_eligible": eligible,
        }
        population.append(pop)
        row = {**pop, "arms": {}}
        for arm, value in zip(audit.ARMS, (0.4, 0.4, 0.9), strict=True):
            metrics = {
                "actor_survival_all_nine": value,
                "actor_joint_forward_reverse_survival": value,
                "cycle_median_normalized": 0.01,
            }
            reports.append(
                {"selection_index": i, "arm": arm, "metrics": metrics, "primary_pass": True}
            )
            row["arms"][arm] = metrics
        rows.append(row)
    independent = audit.reconstruct_assessment(reports, population)
    audit.compare(assess(rows), independent, "final assessment")
    assert independent["primary_pass_counts_of16"][audit.ARMS[2]] == 14
    assert independent["mechanism_supported"]
    with pytest.raises(RuntimeError, match="complete48"):
        audit.reconstruct_assessment(reports[:-1], population)
