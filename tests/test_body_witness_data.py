from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from hac import body_witness_data as body


def observation(*, scenario="1.10", recording=None, track="0", frame=30, box=(10, 20, 110, 120)):
    recording = recording or f"1.{scenario}"
    sample_id = f"train__{recording}__track-{track}__frame-{frame:06d}"
    fold = int(body.EXPECTED_SCENARIOS[scenario][0].split("-")[1])
    return body.CenterObservation(
        sample_id, recording, track, scenario, fold, frame, body.frame_member(recording, frame), box
    )


@pytest.fixture(scope="module")
def full_fixture():
    rows, centers, ids, scenarios, folds = [], [], [], [], []
    labels = np.repeat(np.arange(3), [734, 2118, 2125]).astype(np.int64)
    names = tuple(body.LABEL_INDEX)
    for scenario, (fold_name, count) in sorted(body.EXPECTED_SCENARIOS.items()):
        for index in range(count):
            drone = 1 if scenario == "1.11" else 1 + index % 2
            track = str(index // (5 if scenario == "1.11" else 10))
            current = observation(
                scenario=scenario,
                recording=f"{drone}.{scenario}",
                track=track,
                frame=30 * (1 + index % 5),
            )
            label = int(labels[len(rows)])
            ids.append(current.sample_id)
            scenarios.append(scenario)
            folds.append(int(fold_name[-1]))
            rows.append(
                {"sample_id": current.sample_id, "label": names[label], "label_index": str(label)}
            )
            centers.append(
                {
                    "sample_id": current.sample_id,
                    "provider_recording_id": current.recording,
                    "provider_track_id": current.track,
                    "source_frame": str(current.center_frame),
                    "time_index": "8",
                    "image_member": current.image_member,
                    "image_width": "1280",
                    "image_height": "720",
                    "bbox_xmin": "10",
                    "bbox_ymin": "20",
                    "bbox_xmax": "110",
                    "bbox_ymax": "120",
                    "annotation_occluded": "0",
                    "valid_frame": "1",
                    "support": "pure",
                }
            )
    identity = {
        "sample_ids": np.asarray(ids),
        "labels": labels,
        "scenarios": np.asarray(scenarios),
        "folds": np.asarray(folds),
    }
    allowed = {row["image_member"] for row in centers}
    return identity, rows, centers, allowed


def join(fixture, *, identity=None, rows=None, centers=None):
    original, primary, frames, allowed = fixture
    return body.join_canonical_cohort(
        identity if identity is not None else original,
        rows if rows is not None else primary,
        centers if centers is not None else frames,
        allowed_members=allowed,
    )


def test_canonical_join_is_by_sample_id_and_labels_are_separate(full_fixture):
    original, labels = join(full_fixture)
    reordered, reordered_labels = join(
        full_fixture, rows=full_fixture[1][::-1], centers=full_fixture[2][::-1]
    )
    assert original == reordered
    np.testing.assert_array_equal(labels, reordered_labels)
    assert len(original) == 4977
    assert not labels.flags.writeable
    assert Counter(row.scenario for row in original)["1.10"] == 597
    assert {row.fold for row in original} == set(range(5))
    assert (
        not {"label", "labels", "support", "annotation_occluded", "valid_frame"}
        & original[0].observation_metadata().keys()
    )


def test_annotation_field_invariance(full_fixture):
    altered = [
        dict(
            row,
            annotation_occluded="1",
            annotation_generated="1",
            valid_frame="0",
            valid_geometry="0",
            image_present="0",
            support="unknown",
            label="unavailable",
            missing_reason="annotation_missing",
        )
        for row in full_fixture[2]
    ]
    original, _ = join(full_fixture)
    changed, _ = join(full_fixture, centers=altered)
    assert original == changed
    assert body.select_measurement_pilot(original) == body.select_measurement_pilot(changed)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "float_scenario", "wrong_scenario", "wrong_fold", "wrong_class", "wrong_rows"],
)
def test_rejects_noncanonical_identity(full_fixture, mutation):
    identity = {key: value.copy() for key, value in full_fixture[0].items()}
    if mutation == "duplicate":
        identity["sample_ids"][1] = identity["sample_ids"][0]
    elif mutation == "float_scenario":
        identity["scenarios"] = identity["scenarios"].astype(float)
    elif mutation == "wrong_scenario":
        identity["scenarios"][0] = "1.1"
    elif mutation == "wrong_fold":
        identity["folds"][0] = 3
    elif mutation == "wrong_class":
        identity["labels"][0] = 2
    else:
        identity = {key: value[:-1] for key, value in identity.items()}
    with pytest.raises(ValueError):
        join(full_fixture, identity=identity)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_frame", "wrong_track", "wrong_member"])
def test_rejects_bad_center_join(full_fixture, mutation):
    frames = [dict(row) for row in full_fixture[2]]
    if mutation == "duplicate":
        frames[1] = frames[0]
    elif mutation == "wrong_frame":
        frames[0]["source_frame"] = "31"
    elif mutation == "wrong_track":
        frames[0]["provider_track_id"] = "99"
    else:
        frames[0]["image_member"] = "TestSetFrames/30.jpg"
    with pytest.raises(ValueError):
        join(full_fixture, centers=frames)


def test_missing_geometry_does_not_drop_or_label_mask_center(full_fixture):
    frames = [dict(row) for row in full_fixture[2]]
    frames[0]["bbox_xmin"] = ""
    observed, _ = join(full_fixture, centers=frames)
    assert len(observed) == 4977
    assert observed[0].box_720 is None


def test_pilot_quota_shuffle_invariance_and_track_first(full_fixture):
    population, _ = join(full_fixture)
    selected = body.select_measurement_pilot(population)
    assert selected == body.select_measurement_pilot(population[::-1])
    assert len(selected) == len({row.sample_id for row in selected}) == 128
    expected = {
        scenario: 12 if i < 7 else 11 for i, scenario in enumerate(sorted(body.EXPECTED_SCENARIOS))
    }
    assert Counter(row.scenario for row in selected) == expected
    for scenario, quota in expected.items():
        group = [row for row in selected if row.scenario == scenario]
        available = {(row.recording, row.track) for row in population if row.scenario == scenario}
        keys = [(row.recording, row.track) for row in group]
        assert len(set(keys[: min(quota, len(available))])) == min(quota, len(available))


def test_pilot_second_pass_fills_in_hash_order():
    population = [
        observation(scenario=scenario, track="0", frame=30 * (index + 1))
        for scenario in body.EXPECTED_SCENARIOS
        for index in range(15)
    ]
    selected = body.select_measurement_pilot(population)
    for scenario in body.EXPECTED_SCENARIOS:
        current = [row for row in selected if row.scenario == scenario]
        ordered = sorted(
            [row for row in population if row.scenario == scenario],
            key=lambda row: hashlib.sha256(
                (body.PILOT_HASH_PREFIX + row.sample_id).encode()
            ).digest(),
        )
        assert current == ordered[: len(current)]
    with pytest.raises(ValueError, match="duplicate"):
        body.select_measurement_pilot(population + [population[0]])


def native_source(tmp_path, *, offset=-12, source_size=(3840, 2160)):
    path = tmp_path / "1.1.10.mp4"
    path.write_bytes(b"synthetic video bytes")
    return body.NativeVideoSource(
        "1.1.10",
        path,
        body.sha256_file(path),
        path.stat().st_size,
        source_size,
        Fraction(30000, 1001),
        Fraction(1, 30000),
        offset,
        0,
        3000,
        (),
        "a" * 64,
    )


def test_exact_center_offset_native_box_and_distinct_nominal_time(tmp_path):
    source = native_source(tmp_path)
    request = body.resolve_center_source(observation(frame=30), {source.recording: source})
    assert request.available
    assert request.native_index == 18
    assert request.target_pts == 18018
    assert request.nominal_seconds == 1.0
    assert request.native_seconds == pytest.approx(0.6006)
    assert request.native_box == (30, 60, 330, 360)


@pytest.mark.parametrize(
    "change", [{"offset": -40}, {"maximum_index": 1}, {"missing_indices": (18,)}]
)
def test_missing_native_center_never_chooses_a_neighbor(tmp_path, change):
    source = replace(native_source(tmp_path), **change)
    request = body.resolve_center_source(observation(), {source.recording: source})
    assert not request.available
    assert request.unavailable_reason == "exact_native_center_absent"
    image, receipt = body.decode_exact_center(request)
    assert image is None
    assert receipt["fallback"] == "retain_arftr_no_alternate_frame"


def test_nonintegral_pts_is_unavailable_not_rounded(tmp_path):
    source = replace(native_source(tmp_path), time_base=Fraction(1, 999))
    request = body.resolve_center_source(observation(), {source.recording: source})
    assert not request.available
    assert request.target_pts == -1


def test_decode_validation_requires_exact_pts_and_pixel_shape(tmp_path):
    source = native_source(tmp_path, source_size=(12, 9))
    request = body.resolve_center_source(observation(), {source.recording: source})
    pixels = np.zeros((9, 12, 3), dtype=np.uint8)
    correct = body.validate_decoded_center(
        request, pixels, pts=request.target_pts, time_base=source.time_base
    )
    assert correct["decode_valid"]
    assert correct["image_sha256"] == body.array_digest(pixels)
    later = body.validate_decoded_center(
        request, pixels, pts=request.target_pts + 1001, time_base=source.time_base
    )
    assert not later["decode_valid"]
    wrong_shape = body.validate_decoded_center(
        request, pixels[:8], pts=request.target_pts, time_base=source.time_base
    )
    assert not wrong_shape["decode_valid"]


def test_source_hash_and_stale_verification_rejected(tmp_path):
    source = native_source(tmp_path)
    request = body.resolve_center_source(observation(), {source.recording: source})
    receipt = body.verify_video_file(request)
    source.path.write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="hash changed"):
        body.verify_video_file(request)
    with pytest.raises(RuntimeError, match="verification receipt"):
        body.decode_exact_center(request, verified_video=receipt)


def test_lazy_decoder_rejects_later_frame_and_keeps_budget(tmp_path, monkeypatch):
    source = native_source(tmp_path, source_size=(12, 9))
    request = body.resolve_center_source(observation(), {source.recording: source})
    receipt = body.verify_video_file(request)
    returned = []

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(
                video=[
                    SimpleNamespace(
                        time_base=source.time_base,
                        average_rate=source.fps,
                        codec_context=SimpleNamespace(thread_count=0),
                    )
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def seek(self, pts, *, stream, backward):
            assert pts == request.target_pts and backward

        def decode(self, stream):
            for pts in [request.target_pts - 1001, request.target_pts + 1001]:
                returned.append(pts)
                yield SimpleNamespace(
                    pts=pts, to_ndarray=lambda **unused: np.zeros((9, 12, 3), dtype=np.uint8)
                )

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: Container()))
    image, audit = body.decode_exact_center(request, verified_video=receipt)
    assert image is None and not audit["decode_valid"]
    assert len(returned) == 2
    returned.clear()
    image, audit = body.decode_exact_center(
        request, verified_video=receipt, maximum_decoded_frames=1
    )
    assert image is None and len(returned) == 1


def test_native_crop_geometry_and_padding_without_resize():
    box = (10, 20, 110, 120)
    exact = body.make_body_crop_geometry(box, extent=1.0, source_size=(120, 125))
    context = body.make_body_crop_geometry(box, extent=1.25, source_size=(120, 125))
    assert exact.crop_box == (10, 20, 110, 120)
    assert context.crop_box == (-3, 7, 123, 133)
    np.testing.assert_array_equal(exact.image_to_raw @ [10, 20, 1], [0, 0, 1])
    rgb = np.full((125, 120, 3), 23, dtype=np.uint8)
    crop = body.crop_raw_body(rgb, context)
    assert crop.shape == (126, 126, 3)
    np.testing.assert_array_equal(crop[0, 0], body.PADDING_RGB)
    np.testing.assert_array_equal(crop[0, 3], [23, 23, 23])
    assert context.provenance()["interpolation"] == "none_raw_integer_crop"
    with pytest.raises(ValueError):
        body.make_body_crop_geometry(box, extent=1.5, source_size=(120, 125))
    with pytest.raises(ValueError):
        body.make_body_crop_geometry((1, 2, float("nan"), 10), extent=1.0, source_size=(120, 125))


@pytest.mark.parametrize("extent", body.CROP_EXTENTS)
@pytest.mark.parametrize("mask_id", body.MASK_IDS)
def test_hidden_pixels_cannot_change_predictor_payload(extent, mask_id):
    geometry = body.make_body_crop_geometry(
        (10, 20, 110, 120), extent=extent, source_size=(160, 160)
    )
    width, height = geometry.raw_size
    raw = np.random.default_rng(4).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    masks = body.raw_body_masks(geometry)
    assert not (masks["upper_visible"] & masks["lower_visible"]).any()
    assert np.all(masks["upper_visible"] | masks["lower_visible"] | masks["gap"])
    assert masks["gap"].sum() == 5 * width
    before = body.predictor_payload(raw, geometry, mask_id, decode_succeeded=True)
    corrupted = raw.copy()
    corrupted[~masks[mask_id]] = 255 - corrupted[~masks[mask_id]]
    after = body.predictor_payload(corrupted, geometry, mask_id, decode_succeeded=True)
    np.testing.assert_array_equal(before["masked_rgb"], after["masked_rgb"])
    assert before["input_valid"] == after["input_valid"]
    np.testing.assert_array_equal(before["crop_transform"], after["crop_transform"])
    assert set(before) == {"masked_rgb", "mask_id", "crop_transform", "crop_extent", "input_valid"}
    assert body.array_digest(raw) != body.array_digest(corrupted)


def test_normalized_transform_and_shared_reader_quality_contract():
    box = (10, 20, 110, 120)
    geometries = [
        body.make_body_crop_geometry(box, extent=extent, source_size=(120, 125))
        for extent in body.CROP_EXTENTS
    ]
    transform = body.normalized_crop_transform(geometries[0])
    np.testing.assert_allclose(transform, [0, 0, 1, 1, 100 / 120, 100 / 125])
    quality = body.reader_quality_features(
        geometries, np.array([True, True, True, False])
    )
    assert quality.shape == (36,)
    np.testing.assert_array_equal(quality[-4:], [1, 1, 1, 0])
    np.testing.assert_array_equal(quality[8:14], quality[14:20])
    with pytest.raises(ValueError, match="four fixed-slot"):
        body.reader_quality_features(geometries, np.ones(4, dtype=np.int64))


def test_teacher_corruption_uses_visible_mean_and_preserves_every_other_pixel():
    geometry = body.make_body_crop_geometry((0, 0, 100, 100), extent=1.0, source_size=(100, 100))
    masks = body.raw_body_masks(geometry)
    raw = np.full((100, 100, 3), 99, dtype=np.uint8)
    raw[masks["upper_visible"]] = [10, 20, 30]
    corrupted = body.corrupt_withheld_for_teacher(raw, geometry, "upper_visible")
    assert np.all(corrupted[masks["lower_visible"]] == [10, 20, 30])
    np.testing.assert_array_equal(corrupted[~masks["lower_visible"]], raw[~masks["lower_visible"]])


def test_predictor_validity_and_final_availability_are_separate():
    geometry = body.make_body_crop_geometry((0, 0, 20, 20), extent=1.0, source_size=(20, 20))
    assert body.predictor_input_valid(decode_succeeded=True, geometry=geometry)
    assert not body.predictor_input_valid(decode_succeeded=False, geometry=geometry)
    assert not body.predictor_input_valid(decode_succeeded=True, geometry=None)
    peaks = np.full((4, 2, 8), 0.25)
    decoded = np.ones((4, 2), dtype=bool)
    peaks[1, 1] = 0.24
    peaks[2, 0, 0] = np.nan
    decoded[3, 0] = False
    np.testing.assert_array_equal(
        body.common_acquisition_mask(decoded, peaks), [True, False, False, False]
    )
    assert body.predictor_input_valid(decode_succeeded=True, geometry=geometry)
    with pytest.raises(TypeError):
        body.predictor_input_valid(decode_succeeded=True, geometry=geometry, pose_confidence=0.1)
    with pytest.raises(ValueError):
        body.common_acquisition_mask(decoded.astype(int), peaks)


def test_hashes_bind_shape_dtype_and_payload():
    array = np.arange(6, dtype=np.int32)
    assert body.array_digest(array) != body.array_digest(array.reshape(2, 3))
    assert body.array_digest(array) != body.array_digest(array.astype(np.float32))
    assert body.canonical_digest({"b": 2, "a": 1}) == body.canonical_digest({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        body.array_digest(np.asarray([{}], dtype=object))
    with pytest.raises(ValueError):
        body.canonical_digest({"x": float("nan")})


def test_verified_source_map_binds_alignment_and_fidelity(tmp_path, monkeypatch):
    videos, packets, mappings = [], [], []
    for recording in sorted(body.EXPECTED_RECORDINGS):
        videos.append(
            {
                "recording": recording,
                "zip_member": f"Drone1/{recording}.mp4",
                "sha256": "b" * 64,
                "bytes": 10,
                "size": [3840, 2160],
                "fps": 30000 / 1001,
            }
        )
        packets.append(
            {
                "recording": recording,
                "sha256": "b" * 64,
                "default": {"min_index": 0, "max_index": 3000, "interior_missing_indices": []},
            }
        )
        mappings.append(
            {
                "recording": recording,
                "validated": True,
                "selected_model": {"scale": 1.0, "offset": -2},
                "validation_checks": [{"pyav_time_base": "1/30000"}],
            }
        )
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(json.dumps({"videos": videos}), encoding="utf-8")
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "all_recordings_validated": True,
                "fidelity_summary_sha256": body.sha256_file(fidelity),
                "mappings": mappings,
            }
        ),
        encoding="utf-8",
    )
    support = tmp_path / "support.json"
    support.write_text(
        json.dumps(
            {
                "alignment_sha256": body.sha256_file(alignment),
                "fidelity_sha256": body.sha256_file(fidelity),
                "videos": packets,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        body,
        "SOURCE_RECEIPTS",
        {
            name: (path.name, body.sha256_file(path))
            for name, path in [
                ("alignment", alignment),
                ("fidelity", fidelity),
                ("support", support),
            ]
        },
    )
    loaded = body.load_verified_source_map(tmp_path, source_directory=tmp_path / "videos")
    assert len(loaded) == 21
    assert loaded["1.1.10"].fps == Fraction(30000, 1001)
    assert loaded["1.1.10"].offset == -2
    alignment.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash changed"):
        body.load_verified_source_map(tmp_path)
