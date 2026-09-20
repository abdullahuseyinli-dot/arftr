import copy
import json

import numpy as np
import pytest

from hac.correspondence_field_data import (
    P8_RUN_REQUEST_SHA256,
    canonical_digest,
    field_from_observations,
    file_sha256,
    index_workloads,
    load_correspondence_workload,
    load_frozen_context,
    motion_targets,
)


def observations(count=6):
    sample_id = "train__recording__track-1__frame-000100"
    frames = []
    for index, number in enumerate(range(68, 129, 4)):
        frames.append(
            {
                "source_frame": number,
                "time_index": index,
                "nominal_time_seconds": number / 30,
                "offset_seconds": (number - 100) / 30,
                "bbox": [10.0, 10.0, 30.0, 50.0],
                "image_present": True,
                "image_member": f"scene/{number}.jpg",
                "image_width": 1280,
                "image_height": 720,
                "valid_frame": True,
                "annotation_occluded": False,
            }
        )
    rng = np.random.default_rng(123)
    source = rng.uniform([11, 12], [29, 48], size=(count, 2))
    arrays, pairs = {}, []
    for index in range(15):
        prefix = f"pair_{index:02d}"
        arrays[f"{prefix}_actor_source"] = source.copy()
        arrays[f"{prefix}_actor_target"] = source + [4, 2]
        arrays[f"{prefix}_actor_fb_error"] = np.full(count, 0.4)
        arrays[f"{prefix}_camera_transform"] = np.eye(3)
        pairs.append(
            {
                "sample_id": sample_id,
                "pair_index": index,
                "source_frame": frames[index]["source_frame"],
                "target_frame": frames[index + 1]["source_frame"],
                "elapsed_seconds": 4 / 30,
                "pair_available": True,
                "camera_usable": True,
                "translation_usable": True,
                "articulation_usable": True,
                "camera_audit_median_pixels": 0.2,
                "camera_audit_p90_pixels": 0.5,
                "actor_primary_fb_median_pixels": 0.4,
                "actor_seed_points": count,
                "actor_primary_retained_points": count,
                "actor_vertical_regions": 3,
                "raw_translation_x_height_per_second": 0.75,
                "raw_translation_y_height_per_second": 0.375,
                "compensated_translation_x_height_per_second": 0.75,
                "compensated_translation_y_height_per_second": 0.375,
                "articulation_median_height_per_second": 0.0,
                "articulation_p90_height_per_second": 0.0,
            }
        )
    return sample_id, frames, pairs, arrays


def test_whole_actor_translation_survives_moving_boxes_and_has_correct_units():
    sample_id, frames, pairs, arrays = observations()
    frames[1]["bbox"] = [14.0, 12.0, 34.0, 52.0]
    values, audit = field_from_observations(sample_id, frames, pairs, arrays)
    tokens = values["point_features"][0, values["input_valid"][0]]
    np.testing.assert_allclose(tokens[:, 2:4], np.tile([0.75, 0.375], (6, 1)))
    np.testing.assert_allclose(tokens[:, 4:6], 0, atol=1e-7)
    np.testing.assert_allclose(tokens[:, 6:8], tokens[:, 2:4])
    np.testing.assert_allclose(tokens[:, 8:], np.tile([0.4, 0.01], (6, 1)))
    assert len(audit["selected_raw_point_indices"][0]) == 6
    assert values["point_features"].shape == (15, 32, 10)
    assert not values["input_valid"][:, 6:].any()


def test_annotation_flags_do_not_change_any_observation_array_or_digest():
    sample_id, frames, pairs, arrays = observations()
    before, before_audit = field_from_observations(sample_id, frames, pairs, arrays)
    for frame in frames:
        frame.update(
            valid_frame=False,
            annotation_occluded=True,
            annotation_generated=None,
            label=-1,
            label_valid=False,
            missing_reason="missing_action_annotation",
        )
    for pair in pairs:
        pair.update(long_valid=False, annotation_presence=False, support_purity="unknown")
    after, after_audit = field_from_observations(sample_id, frames, pairs, arrays)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before_audit == after_audit


def test_point_cap_is_deterministic_spatial_and_input_permutation_invariant():
    sample_id, frames, pairs, arrays = observations(count=73)
    # Include explicit corners: farthest-point coverage should span the actor box.
    source = arrays["pair_00_actor_source"]
    source[:4] = [[10, 10], [30, 10], [10, 50], [30, 50]]
    arrays["pair_00_actor_target"] = source + [4, 2]
    before, audit = field_from_observations(sample_id, frames, pairs, arrays, max_points=8)
    permutation = np.random.default_rng(7).permutation(73)
    for key in arrays:
        if "camera_transform" not in key:
            arrays[key] = arrays[key][permutation]
    after, _ = field_from_observations(sample_id, frames, pairs, arrays, max_points=8)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["input_valid"].all()
    selected = before["point_features"][0, :, :2]
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    nearest = np.linalg.norm(corners[:, None] - selected[None], axis=2).min(axis=1)
    assert np.all(nearest < 0.4)
    assert len(set(audit["selected_raw_point_indices"][0])) == 8


@pytest.mark.parametrize(
    "problem", ["nan_transform", "pole", "missing_image", "bad_box", "camera_unusable", "no_points"]
)
def test_invalid_measurements_zero_tokens_and_mask_before_normalization(problem):
    sample_id, frames, pairs, arrays = observations()
    if problem == "nan_transform":
        arrays["pair_00_camera_transform"][:] = np.nan
    elif problem == "pole":
        arrays["pair_00_camera_transform"][2] = [1, 0, -20]
    elif problem == "missing_image":
        frames[0]["image_present"] = False
    elif problem == "bad_box":
        frames[0]["bbox"][3] = 10
    elif problem == "camera_unusable":
        pairs[0]["camera_usable"] = False
    else:
        arrays["pair_00_actor_source"][:] = np.nan
    values, audit = field_from_observations(sample_id, frames, pairs, arrays)
    assert not values["input_valid"][0].any()
    assert not values["pair_input_valid"][0]
    assert not values["point_features"][0].any()
    assert not values["pair_features"][0].any()
    assert np.isfinite(values["point_features"]).all()
    assert audit["invalid_pair_reasons"][0]


def test_translation_and_camera_ablations_recompute_all_redundant_channels():
    sample_id, frames, pairs, arrays = observations()
    arrays["pair_00_actor_target"][0] += [3, -3]
    arrays["pair_00_camera_transform"][:2, 2] = [2, 1]
    full, _ = field_from_observations(sample_id, frames, pairs, arrays)
    tfree, _ = field_from_observations(
        sample_id, frames, pairs, arrays, motion_mode="translation_free"
    )
    rfree, _ = field_from_observations(
        sample_id, frames, pairs, arrays, motion_mode="residual_free"
    )
    identity, _ = field_from_observations(sample_id, frames, pairs, arrays, camera_mode="identity")
    for values in [full, tfree, rfree, identity]:
        p = values["point_features"]
        np.testing.assert_allclose(p[..., 2:4], p[..., 4:6] + p[..., 6:8], atol=1e-7)
    assert not tfree["point_features"][..., 6:8].any()
    assert not rfree["point_features"][..., 4:6].any()
    assert not np.array_equal(identity["point_features"], full["point_features"])
    np.testing.assert_array_equal(identity["input_valid"], full["input_valid"])


def test_loss_valid_does_not_hide_observations_or_label_held_rows():
    labels = np.array([-1, 0, 1, 2, 2])
    losses = motion_targets(labels, np.array([True, True, True, True, False]))
    np.testing.assert_array_equal(losses["loss_valid"], [False, False, True, True, False])
    np.testing.assert_array_equal(losses["targets"], [0, 0, 0, 1, 0])
    with pytest.raises(ValueError, match="training-only"):
        motion_targets(labels, np.array([0, 1]))


def write_workload(root, *, pilot=False):
    sample_id, frames, pairs, arrays = observations()
    location = (
        ".runs/research_20260907/okutama_ccac_pilot/results/workloads/000"
        if pilot
        else "results/workloads/000"
    )
    directory = root / location
    directory.mkdir(parents=True)
    request = {
        "sample_id": sample_id,
        "selection_rank": 0,
        "frames": frames,
        "run_request_sha256": P8_RUN_REQUEST_SHA256 if not pilot else "pilot-run",
    }
    clip = {
        "sample_id": sample_id,
        "center_frame": 100,
        "native_center_height": 40,
        "center_track_survival_fraction_75pct": 1.0,
        "center_seed_points": 6,
    }
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "clip_metrics.json").write_text(json.dumps(clip))
    (directory / "pair_metrics.json").write_text(
        json.dumps({"sample_id": sample_id, "pairs": pairs})
    )
    np.savez(directory / "correspondences.npz", **arrays)
    receipt = {
        "status": "OKUTAMA_CCAC_CLIP_WORKLOAD_COMPLETE",
        "request": request,
        "request_sha256": canonical_digest(request),
        "artifacts": {
            name: file_sha256(directory / name)
            for name in ["clip_metrics.json", "pair_metrics.json", "correspondences.npz"]
        },
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory, request, receipt


def test_workload_checks_hashes_ids_frames_and_preserves_all_rows(tmp_path):
    directory, request, _ = write_workload(tmp_path)
    sample_id = request["sample_id"]
    field = load_correspondence_workload(
        directory,
        repository_root=tmp_path,
        expected_sample_id=sample_id,
        expected_frames=request["frames"],
    )
    assert field.collapsed_features.shape == (31,)
    assert field.audit["source_artifacts"]["correspondences.npz"]
    assert index_workloads(tmp_path / "results", [sample_id]) == {sample_id: directory}
    with pytest.raises(ValueError, match="population"):
        index_workloads(tmp_path / "results", [sample_id, "missing"])
    with pytest.raises(ValueError, match="identity"):
        load_correspondence_workload(
            directory, repository_root=tmp_path, expected_sample_id="wrong"
        )
    changed = copy.deepcopy(request["frames"])
    changed[0]["bbox"][0] += 1
    with pytest.raises(ValueError, match="Physical boxes"):
        load_correspondence_workload(
            directory,
            repository_root=tmp_path,
            expected_sample_id=sample_id,
            expected_frames=changed,
        )
    with (directory / "correspondences.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_correspondence_workload(
            directory, repository_root=tmp_path, expected_sample_id=sample_id
        )


def test_pilot_reference_resolves_verified_source_and_rejects_escape(tmp_path):
    source, source_request, source_receipt = write_workload(tmp_path, pilot=True)
    directory = tmp_path / "results/workloads/000"
    directory.mkdir(parents=True)
    request = {**source_request, "run_request_sha256": P8_RUN_REQUEST_SHA256}
    receipt = {
        "status": "OKUTAMA_CCAC_PILOT_WORKLOAD_REUSED_BY_REFERENCE",
        "request": request,
        "request_sha256": canonical_digest(request),
        "source_directory": source.relative_to(tmp_path).as_posix(),
        "source_request_sha256": canonical_digest(source_request),
        "source_artifacts": source_receipt["artifacts"],
    }
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "receipt.json").write_text(json.dumps(receipt))
    field = load_correspondence_workload(
        directory, repository_root=tmp_path, expected_sample_id=request["sample_id"]
    )
    assert field.audit["source_directory"] == source.relative_to(tmp_path).as_posix()
    receipt["source_directory"] = "../outside"
    (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="escaped"):
        load_correspondence_workload(
            directory, repository_root=tmp_path, expected_sample_id=request["sample_id"]
        )


def test_context_reads_only_named_feature_members_and_exact_center(tmp_path):
    sample_ids = np.array(["a", "b"])
    features = np.arange(2 * 3072, dtype=np.float32).reshape(2, 3072)
    # Pickle-only poison labels prove allow_pickle=False never opens those members.
    np.savez(
        tmp_path / "memory.npz",
        sample_ids=sample_ids,
        features=features,
        labels=np.array([object(), object()]),
        support=np.array([object()]),
    )
    dino = np.zeros((2, 16, 1, 768), dtype=np.float32)
    dino[:, 8, 0] = 123
    np.save(tmp_path / "dino.npy", dino)
    (tmp_path / "summary.json").write_text(json.dumps({"sample_ids": sample_ids.tolist()}))
    paths = {"memory": "memory.npz", "dino": "dino.npy", "dino_summary": "summary.json"}
    hashes = {name: file_sha256(tmp_path / path) for name, path in paths.items()}
    context, audit = load_frozen_context(tmp_path, sample_ids, paths=paths, expected_hashes=hashes)
    np.testing.assert_array_equal(context[:, :1536], features[:, :1536])
    np.testing.assert_array_equal(context[:, 1536:], 123)
    assert audit["memory_npz_members_read"] == ["sample_ids", "features"]
    with pytest.raises(ValueError, match="population"):
        load_frozen_context(tmp_path, sample_ids[::-1], paths=paths, expected_hashes=hashes)
    with pytest.raises(ValueError, match="hashes"):
        load_frozen_context(tmp_path, sample_ids, paths=paths, expected_hashes={})


def test_pair_elapsed_time_and_center_identity_are_strict():
    sample_id, frames, pairs, arrays = observations()
    pairs[0]["elapsed_seconds"] = 1.0
    with pytest.raises(ValueError, match="elapsed-time"):
        field_from_observations(sample_id, frames, pairs, arrays)
    pairs[0]["elapsed_seconds"] = 4 / 30
    with pytest.raises(ValueError, match="center"):
        field_from_observations(sample_id, frames, pairs, arrays, center_frame=101)
