from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hac.learned_persistent_tracking import (
    CycledTracks,
    LocalCoTracker3,
    independently_cycled_tracks,
    point_survival_and_cycle,
)
from hac.persistent_articulation_v2 import TrackSequence

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "learned_smoke_test_module",
    ROOT / "experiments/pilot_okutama_learned_persistent_articulation.py",
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class TranslationPredictor:
    def __init__(self, count, *, reverse_bias=0.0, missing_return=False):
        self.count, self.reverse_bias, self.missing_return = count, reverse_bias, missing_return
        self.calls = []

    def __call__(self, queries, reverse):
        self.calls.append((queries.copy(), reverse))
        velocity = np.asarray([-2.0, 1.0] if reverse else [2.0, -1.0])
        points = (
            queries[None, :, 1:]
            + (np.arange(self.count)[:, None, None] - queries[None, :, :1]) * velocity
        )
        valid = np.ones(points.shape[:2], bool)
        if reverse:
            not_query = np.arange(self.count)[:, None] != queries[None, :, 0]
            points[..., 0] += not_query * self.reverse_bias
            if self.missing_return:
                valid[2] = False
        return points, valid


def test_real_cycle_contract_requeries_endpoints_not_coordinate_inverse():
    fake = TranslationPredictor(5, reverse_bias=1.0)
    seeds = np.asarray([[20.0, 30.0], [50.0, 70.0]])
    tracks = independently_cycled_tracks(
        fake,
        seeds,
        times_seconds=np.arange(5) / 30.0,
        center_index=2,
        sampled_indices=(0, 1, 2, 3, 4),
        point_ids=("a", "b"),
    )
    assert len(fake.calls) == 5 and not fake.calls[0][1]
    assert all(reverse for _, reverse in fake.calls[1:])
    assert np.isnan(tracks.cycle_error[2]).all()
    np.testing.assert_allclose(tracks.cycle_error[[0, 1, 3, 4]], 1.0)
    np.testing.assert_allclose(fake.calls[1][0][:, 1:], tracks.sequence.positions[0])
    assert np.all(fake.calls[1][0][:, 0] == 4)
    report = point_survival_and_cycle(tracks, 100.0, actor_count=2)
    assert report["cycle_median_normalized"] == pytest.approx(0.01)
    assert report["cycle_pairs_observed"] == report["cycle_pairs_expected"] == 8


def test_missing_independent_returns_are_not_zero_cycles():
    fake = TranslationPredictor(5, missing_return=True)
    tracks = independently_cycled_tracks(
        fake,
        np.asarray([[20.0, 30.0]]),
        times_seconds=np.arange(5),
        center_index=2,
        sampled_indices=(0, 2, 4),
        point_ids=("a",),
    )
    report = point_survival_and_cycle(tracks, 10.0, actor_count=1)
    assert report["cycle_median_normalized"] is None
    assert report["cycle_missing_pairs"] == 2
    assert report["actor_survival_all_nine"] == 1.0
    assert report["actor_joint_forward_reverse_survival"] == 0.0


def test_changed_center_identity_is_rejected():
    def bad(queries, reverse):
        positions, valid = TranslationPredictor(5)(queries, reverse)
        positions[2] += 1.0
        return positions, valid

    with pytest.raises(ValueError, match="center query"):
        independently_cycled_tracks(
            bad,
            np.asarray([[20.0, 30.0]]),
            times_seconds=np.arange(5),
            center_index=2,
            sampled_indices=(0, 2, 4),
            point_ids=("a",),
        )


def test_duplicate_sample_times_rejected_before_predictor():
    with pytest.raises(ValueError, match="invalid"):
        independently_cycled_tracks(
            None,
            np.asarray([[20.0, 30.0]]),
            times_seconds=np.arange(5),
            center_index=2,
            sampled_indices=(0, 2, 2, 4),
            point_ids=("a",),
        )


def test_framewise_resize_and_query_scaling_match_official_shape_contract():
    import torch

    backend = LocalCoTracker3.__new__(LocalCoTracker3)
    backend.torch, backend.device, backend.model_shape, backend.calls = torch, "cpu", (5, 7), 0
    frames = [
        np.random.default_rng(i).integers(0, 256, (13, 17, 3), dtype=np.uint8) for i in range(3)
    ]
    prepared = backend.prepare_video(frames)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float(),
        (5, 7),
        mode="bilinear",
        align_corners=True,
    )
    torch.testing.assert_close(prepared.tensor[0], expected, atol=0, rtol=0)
    calls = []

    def fake_model(video, queries, backward_tracking):
        calls.append((video.clone(), queries.clone(), backward_tracking))
        tracks = queries[:, None, :, 1:].expand(1, 3, -1, 2).clone()
        return tracks, torch.ones((1, 3, queries.shape[1]), dtype=torch.bool)

    backend.predictor = fake_model
    queries = np.asarray([[1.0, 8.0, 6.0], [2.0, 16.0, 12.0]])
    positions, valid = backend.predict(prepared, queries, reverse=True)
    np.testing.assert_allclose(positions, np.broadcast_to(queries[:, 1:], (3, 2, 2)))
    torch.testing.assert_close(calls[0][0], prepared.tensor.flip(1))
    np.testing.assert_allclose(calls[0][1][0].numpy(), [[1.0, 3.0, 2.0], [2.0, 6.0, 4.0]])
    assert valid.all() and calls[0][2] and backend.calls == 1


def test_coordinate_controls_do_not_claim_image_validation():
    result = pilot.coordinate_controls()
    assert result["pass"]
    assert "not image-tracker" in result["scope"]
    assert result["common_pairs"] == 8 * 32
    assert all(value >= 0.25 for value in result["specificity_penalties"].values())


def test_synthetic_rgb_correspondences_and_zero_motion_are_explicit():
    result = pilot.synthetic_case("camera_only")
    assert len(result["frames"]) == 9
    assert all(f.shape == (256, 384, 3) and f.dtype == np.uint8 for f in result["frames"])
    np.testing.assert_allclose(result["root"], np.broadcast_to(np.eye(3), (9, 3, 3)))
    assert not np.array_equal(result["frames"][0], result["frames"][-1])


def _fake_carriers():
    rng = np.random.default_rng(21)
    positions = rng.uniform(1.0, 10.0, (1, 128, 2)) + np.arange(61)[:, None, None] * np.asarray(
        [0.1, 0.02]
    )
    times = (np.arange(61) - 30) / 30.0
    masks = np.ones((61, 128), bool)
    cycles = np.zeros((9, 128))
    cycles[4] = np.nan
    return {
        name: CycledTracks(
            TrackSequence(positions.copy(), masks.copy(), times.copy(), pilot.IDS),
            cycles.copy(),
            np.isfinite(cycles),
            pilot.SAMPLED,
            0,
        )
        for name in pilot.NAMES
    }


def test_cross_reference_table_has_identical_held_support_and_excludes_center():
    carriers = _fake_carriers()
    carriers["learned_persistent"].sequence.valid[0, 1] = False
    report, sampled, _ = pilot.compare_carriers(carriers, 100.0)
    masks = [row["common_mask"] for row in report.values()]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    assert all(row["common_pairs"] == 255 for row in report.values())
    assert not masks[0][4].any()
    assert all(
        row["endpoint"] == "common_held_point_consistency_not_ground_truth"
        for row in report.values()
    )


@pytest.mark.parametrize("change", ["identity", "time"])
def test_carrier_axis_mismatch_rejected_before_scoring(change):
    carriers = _fake_carriers()
    original = carriers["lk_persistent"]
    ids = original.sequence.point_ids[::-1] if change == "identity" else original.sequence.point_ids
    times = original.sequence.times_seconds + (0.01 if change == "time" else 0.0)
    carriers["lk_persistent"] = CycledTracks(
        TrackSequence(original.sequence.positions, original.sequence.valid, times, ids),
        original.cycle_error,
        original.cycle_valid,
        original.sampled_indices,
        0,
    )
    with pytest.raises(ValueError, match="identity/time"):
        pilot.compare_carriers(carriers, 100.0)


def test_protocol_has_no_hidden_expansion_or_task_fit():
    protocol = json.loads(pilot.PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["centers"] == 16 and not protocol["task_fit_authorized"]
    assert protocol["geometry"]["real_retention_gate"] is None
    assert protocol["sampled_offsets"] == [-30, -22, -15, -8, 0, 8, 15, 22, 30]
    assert len(protocol["sensitivity_references"]) == 3
    assert not any("train" in carrier for carrier in protocol["carriers"])


def test_one_of_512_returned_points_cannot_pass_joint_cycle_coverage():
    original = _fake_carriers()["learned_persistent"]
    cycles = np.full((9, 128), np.nan)
    cycles[0, 0] = 0.0
    sparse = CycledTracks(original.sequence, cycles, np.isfinite(cycles), pilot.SAMPLED, 0)
    report = point_survival_and_cycle(sparse, 100.0)
    assert report["actor_survival_all_nine"] == 1.0
    assert report["cycle_median_normalized"] == 0.0
    assert report["cycle_missing_pairs"] == 511
    assert report["actor_joint_forward_reverse_survival"] == 0.0


def test_nonrigid_ground_truth_gate_rejects_rigid_only_tracker():
    truth = pilot.synthetic_case("nonrigid", render=False)
    true_positions = np.concatenate((truth["actor"], truth["background"]), axis=1)
    truth_seq = TrackSequence(true_positions, np.ones((9, 128), bool), truth["times"], pilot.IDS)
    assert pilot.score_synthetic_track(truth_seq, truth, nonrigid=True)["pass"]
    rigid = true_positions.copy()
    rigid[:, :64] = np.stack(
        [
            pilot.apply_similarity(c, pilot.apply_similarity(r, truth["actor"][4]))
            for c, r in zip(truth["camera"], truth["root"], strict=True)
        ]
    )
    rigid_seq = TrackSequence(rigid, truth_seq.valid, truth["times"], pilot.IDS)
    report = pilot.score_synthetic_track(rigid_seq, truth, nonrigid=True)
    assert report["nonrigid_known_truth_relative_rms"] == pytest.approx(1.0)
    assert not report["pass"]


def test_missing_actor_and_high_motion_support_cannot_hide_in_background_success():
    truth = pilot.synthetic_case("same_direction", render=False)
    positions = np.concatenate((truth["actor"], truth["background"]), axis=1)
    valid = np.ones((9, 128), bool)
    valid[:, :32] = False
    report = pilot.score_synthetic_track(
        TrackSequence(positions, valid, truth["times"], pilot.IDS), truth, nonrigid=False
    )
    assert (
        report["actor_survival_all_times"] == 0.5 and report["background_survival_all_times"] == 1.0
    )
    assert not report["pass"]
    valid[:] = True
    valid[[0, 1, 7, 8], :64] = False
    report = pilot.score_synthetic_track(
        TrackSequence(positions, valid, truth["times"], pilot.IDS), truth, nonrigid=False
    )
    assert report["high_true_motion_support_fraction"] == 0.0
    assert not report["pass"]


def test_exact_downsample_pixel_center_coordinate_map():
    np.testing.assert_allclose(
        pilot.scaled_pixel_centers([[1.0, 4.0], [7.0, 10.0]], 1 / 3), [[0.0, 1.0], [2.0, 3.0]]
    )
    point = np.asarray([[14.2, 40.3]])
    np.testing.assert_allclose(
        pilot.scaled_pixel_centers(pilot.scaled_pixel_centers(point, 1 / 3), 3), point
    )


def test_numerical_abort_is_not_primary_no_go_authorization(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"status": "NUMERICAL_ABORT", "promotion_passed": False, "gate_fits": 9})
    )
    with pytest.raises(ValueError, match="completed 45-fit"):
        pilot.validate_primary_no_go(path)


def test_primary_audit_hash_change_is_rejected(tmp_path):
    path = tmp_path / "summary.json"
    (tmp_path / "independent_audit.json").write_text("{}")
    (tmp_path / "execution_lock.json").write_text("{}")
    path.write_text(
        json.dumps(
            {
                "status": "CROSSING_EVENT_COMPLETE_NO_GO",
                "promotion_passed": False,
                "gate_fits": 45,
                "audit_sha256": "wrong",
                "execution_lock_sha256": "wrong",
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.validate_primary_no_go(path)
