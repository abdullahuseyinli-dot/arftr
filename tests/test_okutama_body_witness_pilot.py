from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.audit_okutama_body_witness import (
    audit_dependency_arrays,
    audit_pilot_target_sensitivity,
    audit_target_sensitivity,
    categorical_total_variation,
)
from experiments.build_okutama_body_witness_review import build_interactive
from experiments.pilot_okutama_body_witness import exact_fov_pose_tensor
from experiments.run_okutama_body_witness import architecture_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_exact_fov_pose_adapter_preserves_full_crop_and_shape():
    raw = np.zeros((100, 50, 3), dtype=np.uint8)
    raw[:, :25, 0] = 255
    raw[:, 25:, 2] = 255
    tensor, receipt = exact_fov_pose_tensor(raw)
    assert tensor.shape == (3, 256, 192)
    assert tensor.dtype == np.float32
    assert receipt["processor_bbox_padding_disabled"] is True
    assert receipt["source_size"] == [50, 100]
    # Tall input letterboxes horizontally; neither colored half is cropped.
    assert receipt["resized_size"] == [128, 256]
    assert receipt["padding_left_top"] == [32, 0]


def test_total_variation_scale_can_cross_locked_point_zero_five_gate():
    original = np.zeros((2, 4, 8, 193), dtype=np.float64)
    changed = original.copy()
    original[..., 0] = 1.0
    changed[..., 1] = 1.0
    tv = categorical_total_variation(original, changed)
    assert np.array_equal(tv, np.ones((2, 4)))
    result = audit_target_sensitivity(
        original, changed, review_pass=np.array([True, False])
    )
    assert result["status"] == "PASS"
    assert result["eligible_witnesses"] == 4
    assert result["fraction"] == 1.0


def test_dependency_audit_requires_inputs_and_outputs_to_be_bit_exact():
    values = {
        "visible_pixels_original": np.zeros((2, 3), np.uint8),
        "visible_pixels_corrupted": np.zeros((2, 3), np.uint8),
        "visible_features_original": np.zeros((2, 4), np.float32),
        "visible_features_corrupted": np.zeros((2, 4), np.float32),
        "crop_transforms_original": np.eye(3, dtype=np.float64)[None],
        "crop_transforms_corrupted": np.eye(3, dtype=np.float64)[None],
        "predictor_validity_original": np.ones((1,), bool),
        "predictor_validity_corrupted": np.ones((1,), bool),
        "mask_ids_original": np.array([0], np.int64),
        "mask_ids_corrupted": np.array([0], np.int64),
        "predictor_logits_original": np.zeros((1, 2), np.float32),
        "predictor_logits_corrupted": np.zeros((1, 2), np.float32),
    }
    assert audit_dependency_arrays(values)["status"] == "PASS"
    values["visible_features_corrupted"][0, 0] = 1
    assert audit_dependency_arrays(values)["status"] == "FAIL"


def test_protocol_freezes_sensitivity_metric_and_no_training_authorization():
    protocol = json.loads(
        (ROOT / "experiments/okutama_body_witness_protocol.json").read_text()
    )
    assert protocol["pilot"]["rows"] == 128
    assert protocol["pilot"]["selection"]["salt"] == "hac-body-witness-v1|"
    assert protocol["observations"]["crop_extents"] == [1.0, 1.25]
    sensitivity = protocol["pilot"]["target_sensitivity_gate"]
    assert sensitivity["total_variation_strict_min"] == 0.05
    assert sensitivity["spatial_witness_fraction_min"] == 0.5
    assert "total_variation" in sensitivity["metric_name"]
    assert "task_model_fitting" not in protocol["execution_authorization"]["currently_authorized"]


def test_full_active_capacity_and_equal_direct_inputs_are_locked():
    receipt = architecture_receipt()
    assert receipt["active_trainable_parameters"] == {
        "V0": 152321,
        "V1": 790153,
        "V2": 796361,
        "V3": 821287,
    }
    assert receipt["V1_V2_V3_active_relative_parameter_range"] < 0.10
    assert receipt["capacity_includes_every_task_trained_module"] is True
    assert receipt["identical_direct_RGB_bank_V1_V2_V3"] is True


def test_per_bin_mean_l1_is_rejected_as_gate_interpretation():
    # The maximum mean-per-bin L1 of two 193-bin categorical distributions is
    # 2/193, so it can never exceed 0.05.
    assert 2 / 193 < 0.05
    malformed = np.zeros((1, 8, 193))
    with pytest.raises(ValueError, match="normalized"):
        categorical_total_variation(malformed, malformed)


def test_production_sensitivity_rejects_toy_shapes():
    arrays = {
        "sample_ids": np.array(["one"]),
        "original_witness": np.array([[[[1.0, 0.0]]]]),
        "corrupted_witness": np.array([[[[0.0, 1.0]]]]),
        "review_pass": np.array([True]),
    }
    with pytest.raises(RuntimeError, match="128x4x8x193"):
        audit_pilot_target_sensitivity(arrays, [{"sample_id": "one"}])


def test_interactive_review_is_label_blind_and_exports_assigned_schema(tmp_path):
    run = tmp_path / "pilot"
    run.mkdir()
    rows = []
    for index in range(128):
        sample_id = f"sample-{index:03d}"
        rows.append(
            {
                "selection_index": index,
                "sample_id": sample_id,
                "crops": {
                    "1.0": {"file": f"blinded_review/images/{sample_id}__extent-1p0.png"},
                    "1.25": {"file": f"blinded_review/images/{sample_id}__extent-1p25.png"},
                },
            }
        )
    (run / "crop_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    page = build_interactive(run)
    assert "click-to-annotate" not in page  # no implementation prose leaks into UI
    assert "action_labels" not in page
    assert "ARFTR_outputs" not in page
    assert "left_shoulder" in page
    assert "n+'_visible'" in page
    assert "Export assigned CSV" in page
    assert "function fitPrimary()" in page
    assert "Image/canvas aspect ratio mismatch" in page
    assert "hac-body-witness-review-v2-" in page
    assert "object-fit:contain" not in page
    assert page.count("sample-") == 384  # sample id plus two image paths per row
