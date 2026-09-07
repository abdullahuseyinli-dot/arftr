from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hac.ccac_motion import (
    CameraEstimate,
    _homography_has_safe_image_domain,
    analyze_actor_correspondence,
    estimate_camera,
    native_size_band,
    select_pilot_candidates,
    track_points_forward_backward,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "experiments/okutama_ccac_pilot_protocol.json").read_text(encoding="utf-8")
)


def _runner_module():
    path = ROOT / "experiments/run_okutama_ccac_pilot.py"
    spec = importlib.util.spec_from_file_location("ccac_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_size_band_boundaries() -> None:
    assert native_size_band(31.999) == "small"
    assert native_size_band(32.0) == "medium"
    assert native_size_band(64.0) == "medium"
    assert native_size_band(64.001) == "large"
    with pytest.raises(ValueError):
        native_size_band(0)


def test_label_blind_selection_is_balanced_and_order_invariant() -> None:
    candidates = []
    bands = ("small", "medium", "large")
    for scenario in ("a", "b", "c"):
        for band in bands:
            for index in range(8):
                candidates.append(
                    {
                        "sample_id": f"{scenario}-{band}-{index}",
                        "scenario": scenario,
                        "size_band": band,
                        "long_valid": index >= 2,
                        "track_key": f"{scenario}-track-{index}",
                    }
                )
    selection = {
        "hash_domain": "unit-test|",
        "selected_clips": 12,
        "size_targets": {"small": 4, "medium": 4, "large": 4},
        "long_incomplete_target": 2,
    }
    first = select_pilot_candidates(candidates, selection)
    second = select_pilot_candidates(list(reversed(candidates)), selection)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == 12
    assert sum(not row["long_valid"] for row in first) == 2
    assert {band: sum(row["size_band"] == band for row in first) for band in bands} == {
        "small": 4,
        "medium": 4,
        "large": 4,
    }
    assert {scenario: sum(row["scenario"] == scenario for row in first) for scenario in "abc"} == {
        "a": 4,
        "b": 4,
        "c": 4,
    }


def test_forward_backward_lk_recovers_translation() -> None:
    rng = np.random.default_rng(4)
    previous = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (3, 3), 0)
    current = cv2.warpAffine(
        previous,
        np.asarray(((1.0, 0.0, 3.0), (0.0, 1.0, -2.0))),
        (128, 128),
    )
    points = cv2.goodFeaturesToTrack(
        previous, maxCorners=40, qualityLevel=0.01, minDistance=5, blockSize=3
    ).reshape(-1, 2)
    tracked = track_points_forward_backward(
        previous,
        current,
        points,
        PROTOCOL["background_correspondence"]["lk"],
        maximum_error=1.0,
    )
    assert tracked.valid.sum() >= 20
    displacement = tracked.target[tracked.valid] - tracked.source[tracked.valid]
    np.testing.assert_allclose(np.median(displacement, axis=0), (3.0, -2.0), atol=0.08)


def test_reverse_lk_never_receives_failed_or_nonfinite_forward_points(monkeypatch) -> None:
    calls = []

    def fake_lk(previous, current, query, _unused, **kwargs):
        del previous, current, kwargs
        calls.append(query.copy())
        if len(calls) == 1:
            forward = query.copy()
            forward[1, 0] = (np.nan, np.nan)
            return forward, np.asarray(((1,), (0,)), dtype=np.uint8), None
        assert len(query) == 1
        return np.asarray((([10.0, 10.0],),), dtype=np.float32), np.ones((1, 1), np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    image = np.zeros((32, 32), np.uint8)
    points = np.asarray(((10, 10), (12, 12), (np.nan, 4)), np.float32)
    result = track_points_forward_backward(
        image,
        image,
        points,
        PROTOCOL["actor_correspondence"]["primary_lk"],
        maximum_error=1.0,
    )
    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert result.valid.tolist() == [True, False, False]


def test_camera_uses_disjoint_audit_and_recovers_similarity() -> None:
    source = np.asarray(
        [(x, y) for y in np.linspace(20, 700, 8) for x in np.linspace(20, 1260, 12)],
        np.float64,
    )
    cells = np.asarray(
        [min(int(y * 3 / 720), 2) * 4 + min(int(x * 4 / 1280), 3) for x, y in source]
    )
    target = source + (5.0, -3.0)
    result = estimate_camera(
        source,
        target,
        cells,
        (720, 1280),
        seed=31,
        spec=PROTOCOL["camera_hypotheses"],
        gate=PROTOCOL["engineering_gates"],
    )
    assert result.usable
    assert result.selected == "H1"
    assert result.metrics["background_fit_points"] > 0
    assert result.metrics["background_selection_points"] >= 8
    assert result.metrics["background_audit_points"] >= 8
    assert result.metrics["camera_audit_median_pixels"] < 1e-4


def test_projective_pole_and_reflection_are_rejected() -> None:
    pole = np.asarray(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (-2 / 1280, 0.0, 1.0)))
    reflection = np.asarray(((-1.0, 0.0, 1280.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    assert not _homography_has_safe_image_domain(pole, (720, 1280))
    assert not _homography_has_safe_image_domain(reflection, (720, 1280))
    assert _homography_has_safe_image_domain(np.eye(3), (720, 1280))


def test_flat_actor_is_unavailable_not_zero_motion() -> None:
    image = np.zeros((120, 160), dtype=np.uint8)
    metrics, _ = analyze_actor_correspondence(
        image,
        image,
        (30.0, 20.0, 80.0, 100.0),
        (30.0, 20.0, 80.0, 100.0),
        CameraEstimate(np.eye(3), "H0", True, {}),
        elapsed_seconds=4 / 30,
        spec=PROTOCOL["actor_correspondence"],
        decomposition=PROTOCOL["motion_decomposition"],
    )
    assert not metrics["translation_usable"]
    assert metrics["translation_failure_reason"] == "insufficient_actor_correspondences"
    assert metrics["raw_translation_x_height_per_second"] is None
    assert metrics["compensated_translation_x_height_per_second"] is None
    assert metrics["articulation_median_height_per_second"] is None


def test_deterministic_npz_bytes_are_byte_exact() -> None:
    runner = _runner_module()
    arrays = {"z": np.arange(4), "a": np.eye(2, dtype=np.float32)}
    assert runner.deterministic_npz_bytes(arrays) == runner.deterministic_npz_bytes(arrays)
