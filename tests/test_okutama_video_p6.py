from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import run_okutama_video_p6 as runner

from hac.video_consensus import ARM_COMPONENTS, ARMS, derive_consensus, uniform_consensus
from tools import lock_okutama_video_p6 as lock

ROOT = Path(__file__).resolve().parents[1]


def test_uniform_consensus_is_exact_label_blind_arithmetic() -> None:
    left = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
    middle = np.asarray([[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]], dtype=np.float64)
    right = np.asarray([[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]], dtype=np.float64)
    retained = left.copy()
    actual = uniform_consensus((left, middle, right))
    expected = sum(np.asarray(value, dtype=np.float64) for value in (left, middle, right)) / 3
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(left, retained)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([[0.5, 0.5]]),
        np.asarray([[0.5, 0.4, 0.4]]),
        np.asarray([[0.5, np.nan, 0.5]]),
        np.asarray([[1.1, -0.1, 0.0]]),
    ],
)
def test_uniform_consensus_fails_closed_on_invalid_components(bad: np.ndarray) -> None:
    valid = np.asarray([[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError):
        uniform_consensus((valid, bad))


def test_locked_arm_wiring_and_protocol_are_exact() -> None:
    sources = {
        "p3": {
            "long_vjepa_mean": np.asarray([[0.6, 0.3, 0.1]]),
            "dual_scale_vjepa_dino": np.asarray([[0.3, 0.6, 0.1]]),
        },
        "p5": {
            "orthogonal_moments_factorized": np.asarray([[0.3, 0.2, 0.5]]),
            "spatial_contrast_factorized": np.asarray([[0.1, 0.2, 0.7]]),
        },
    }
    outputs = derive_consensus(sources)
    assert tuple(outputs) == ARMS
    np.testing.assert_allclose(outputs[ARMS[0]], [[0.4, 11 / 30, 7 / 30]])
    np.testing.assert_allclose(outputs[ARMS[1]], [[1 / 3, 11 / 30, 0.3]])
    spec = lock.load_protocol(ROOT)
    assert spec["architecture"]["arms"] == {
        arm: [f"{phase}.{name}" for phase, name in ARM_COMPONENTS[arm]] for arm in ARMS
    }
    assert spec["statistics"]["comparisons"] == lock.expected_comparisons()
    assert spec["adaptation_disclosure"]["already_observed_primary_macro_f1"] > 0.82
    runner._validate_protocol(spec)


def test_protocol_drift_and_lock_overwrite_fail_closed(tmp_path: Path) -> None:
    spec = json.loads((ROOT / lock.PROTOCOL_PATH).read_text(encoding="utf-8"))
    protocol_path = tmp_path / lock.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    spec["architecture"]["learned_parameters"] = 1
    protocol_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        lock.load_protocol(tmp_path)
    output = tmp_path / "lock.json"
    lock.write_lock(tmp_path, output, {"status": lock.STATUS})
    with pytest.raises(FileExistsError, match="overwrite"):
        lock.write_lock(tmp_path, output, {"status": "changed"})


def test_summary_records_zero_fit_target_crossing_and_adaptation() -> None:
    labels = np.tile(np.arange(3), 10)
    rows = len(labels)
    scenarios = np.repeat([f"scene-{index}" for index in range(5)], 6)
    folds = np.repeat(np.arange(5), 6)
    strong = np.full((rows, 3), 0.05)
    strong[np.arange(rows), labels] = 0.9
    weak = np.roll(strong, 1, axis=0)
    sources = {
        "p3": {"long_vjepa_mean": strong, "dual_scale_vjepa_dino": strong},
        "p5": {
            "orthogonal_moments_factorized": strong,
            "spatial_contrast_factorized": strong,
        },
    }
    evidence = runner.Evidence(
        sample_ids=np.asarray([str(index) for index in range(rows)]),
        scenarios=scenarios,
        labels=labels,
        folds=folds,
        long_valid=np.arange(rows) % 2 == 0,
        baseline=weak,
        p3_best=weak,
        p5_best=weak,
        sources=sources,
    )
    spec = lock.load_protocol(ROOT)
    spec["statistics"]["bootstrap_resamples"] = 20
    result = runner.summarize(evidence, derive_consensus(sources), spec)
    assert result["target_crossed_by"] == list(ARMS)
    assert result["primary_macro_f1"] == 1.0
    assert result["model_fits"] == 0
    assert not result["labels_used_in_fusion"]
    assert "already_observed_primary_macro_f1" in result["adaptation_disclosure"]
