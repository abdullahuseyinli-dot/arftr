from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hac.persistent_articulation_v2 import TrackSequence, fixed_point_split, similarity_design

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tracking_scale_synthetic_test_module", ROOT / "experiments/tracking_scale_synthetic.py"
)
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)


def _sequence(fixture, positions=None, valid=None):
    return TrackSequence(
        fixture["actor_truth"] if positions is None else positions,
        np.ones((9, 64), bool) if valid is None else valid,
        fixture["times_seconds"],
        fixture["point_ids"],
    )


def _cycles():
    errors = np.zeros((9, 64), dtype=np.float64)
    errors[4] = np.nan
    valid = np.ones((9, 64), bool)
    valid[4] = False
    return errors, valid


@pytest.mark.parametrize("size", ["typical", "tiny"])
@pytest.mark.parametrize("case", synthetic.CASES)
def test_all_ten_fixed_fixtures_recover_known_physical_motion(size, case):
    fixture = synthetic.synthetic_scale_case(size, case, render=False)
    assert fixture["frames"] == []
    assert fixture["actor_truth"].shape == (9, 64, 2)
    assert len(set(fixture["point_ids"])) == 64
    width, height = synthetic.ACTOR_SIZES[size]
    x0, y0, x1, y1 = fixture["box_native"]
    assert (x1 - x0, y1 - y0) == (width, height)
    assert (
        np.max(np.linalg.norm(fixture["root_displacement_truth"], axis=-1))
        <= 0.15 * fixture["diagonal"]
    )
    report = synthetic.score_synthetic_scale_tracks(_sequence(fixture), fixture, *_cycles())
    assert report["pass"] and report["held_pairs_observed"] == 256
    assert report["actor_endpoint_median_diagonal"] < 1e-12
    assert report["root_endpoint_median_diagonal"] < 1e-8
    assert report["joint_forward_return_survival"] == 1.0
    assert not report["camera_estimator_evaluated"] and not report["real_point_gate_waived"]


def test_tiny_nonrigid_layout_is_identifiable_and_orthogonal_to_root():
    fixture = synthetic.synthetic_scale_case("tiny", "nonrigid", render=False)
    fit, _ = fixed_point_split(fixture["point_ids"]).indices(fixture["point_ids"])
    center = fixture["actor_truth"][4]
    design = similarity_design(center[fit])
    for t in range(9):
        np.testing.assert_allclose(
            design.T @ fixture["articulation_truth"][t, fit].reshape(-1), 0.0, atol=1e-7
        )
        distances = np.linalg.norm(
            fixture["actor_truth"][t, :, None] - fixture["actor_truth"][t, None, :], axis=-1
        )
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 4.0  # 3x3 sprites have <=4px subpixel support
    assert (
        np.max(np.linalg.norm(fixture["root_displacement_truth"], axis=-1))
        > 0.05 * fixture["diagonal"]
    )
    assert np.max(np.abs(fixture["articulation_truth"])) == pytest.approx(
        0.04 * fixture["diagonal"]
    )


@pytest.mark.parametrize("size", ["typical", "tiny"])
def test_frozen_center_and_rigid_only_negative_controls_fail(size):
    moving = synthetic.synthetic_scale_case(size, "actor_only", render=False)
    frozen = np.broadcast_to(moving["actor_truth"][4], (9, 64, 2)).copy()
    report = synthetic.score_synthetic_scale_tracks(_sequence(moving, frozen), moving, *_cycles())
    assert not report["pass"]
    assert not report["checks"]["relative_root_error"]
    assert report["relative_root_error_above_0_05_diagonal"] == pytest.approx(1.0)
    nonrigid = synthetic.synthetic_scale_case(size, "nonrigid", render=False)
    rigid = nonrigid["actor_truth"] - nonrigid["articulation_truth"]
    report = synthetic.score_synthetic_scale_tracks(
        _sequence(nonrigid, rigid), nonrigid, *_cycles()
    )
    assert not report["pass"] and not report["checks"]["nonrigid_known_truth_recovery"]
    assert report["nonrigid_known_truth_relative_rms"] == pytest.approx(1.0)


def test_sparse_zero_cycle_cannot_hide_missing_returns():
    fixture = synthetic.synthetic_scale_case("tiny", "actor_only", render=False)
    error = np.full((9, 64), np.nan)
    error[0, 0] = 0.0
    report = synthetic.score_synthetic_scale_tracks(
        _sequence(fixture), fixture, error, np.isfinite(error)
    )
    assert report["cycle_median_diagonal"] == 0.0
    assert report["cycle_pairs_observed"] == 1 and report["joint_forward_return_survival"] == 0.0
    assert not report["pass"]


def test_missing_high_motion_or_held_ids_cannot_select_easy_support():
    fixture = synthetic.synthetic_scale_case("tiny", "nonrigid", render=False)
    valid = np.ones((9, 64), bool)
    valid[[0, 1, 2, 6, 7, 8]] = False
    report = synthetic.score_synthetic_scale_tracks(
        _sequence(fixture, valid=valid), fixture, *_cycles()
    )
    assert report["high_motion_pairs_expected"] > 0 and report["high_motion_pairs_observed"] == 0
    assert not report["checks"]["relative_root_error"] and not report["pass"]
    valid[:] = True
    _, held = fixed_point_split(fixture["point_ids"]).indices(fixture["point_ids"])
    valid[:, held[:16]] = False
    report = synthetic.score_synthetic_scale_tracks(
        _sequence(fixture, valid=valid), fixture, *_cycles()
    )
    assert report["forward_all_time_survival"] == 0.75
    assert not report["checks"]["held_geometry_support"] and not report["pass"]


def test_identity_time_and_center_cycle_mismatch_are_rejected():
    fixture = synthetic.synthetic_scale_case("tiny", "camera_only", render=False)
    wrong = TrackSequence(
        fixture["actor_truth"],
        np.ones((9, 64), bool),
        fixture["times_seconds"],
        fixture["point_ids"][::-1],
    )
    with pytest.raises(ValueError, match="identities/timestamps"):
        synthetic.score_synthetic_scale_tracks(wrong, fixture, *_cycles())
    errors, mask = _cycles()
    errors[4] = 0.0
    with pytest.raises(ValueError, match="Center"):
        synthetic.score_synthetic_scale_tracks(_sequence(fixture), fixture, errors, mask)


def test_native_rgb_render_agrees_with_camera_truth_without_background_padding():
    fixture = synthetic.synthetic_scale_case("tiny", "nonrigid")
    assert len(fixture["frames"]) == 9
    assert all(
        frame.shape == (2160, 3840, 3) and frame.dtype == np.uint8 for frame in fixture["frames"]
    )
    center = fixture["frames"][4]
    for frame, camera, points in zip(
        fixture["frames"], fixture["camera_truth"], fixture["actor_truth"], strict=True
    ):
        dx, dy = camera[:2, 2].astype(int)
        np.testing.assert_array_equal(
            frame[600 + dy : 610 + dy, 500 + dx : 510 + dx], center[600:610, 500:510]
        )
        for x, y in points:
            patch = frame[
                int(np.floor(y)) - 2 : int(np.floor(y)) + 3,
                int(np.floor(x)) - 2 : int(np.floor(x)) + 3,
            ]
            assert patch.max() > 100  # every known point has its physical bright texture


def test_only_locked_sizes_and_cases_are_accepted():
    with pytest.raises(ValueError, match="fixed sizes"):
        synthetic.synthetic_scale_case("chosen_after_results", "actor_only", render=False)
    with pytest.raises(ValueError, match="five locked"):
        synthetic.synthetic_scale_case("tiny", "easier_motion", render=False)
