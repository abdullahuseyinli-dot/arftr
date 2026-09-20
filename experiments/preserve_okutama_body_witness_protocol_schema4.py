"""Materialize the exact schema-4 protocol bytes referenced by pose receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "experiments/okutama_body_witness_protocol.json"
OUTPUT = (
    ROOT
    / ".runs/research_20260913/body_witness_review_geometry_v1/"
    "okutama_body_witness_protocol_schema4.json"
)
RECEIPT = OUTPUT.with_name("protocol_schema4_preservation_receipt.json")
EXPECTED_SHA256 = "4f4b6cd285ea2877092751500a941d059cd5ebc55be8a29ea918ed2a32ed4be8"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reconstruct_schema4(current: str) -> str:
    start_marker = (
        "      },\n      {\n        \"date\": \"2026-09-13\",\n"
        "        \"reason\": \"Post-extraction overlay inspection exposed horizontal"
    )
    start = current.index(start_marker)
    end = current.index("\n      }\n    ]", start) + len("\n      }")
    value = current[:start] + "      }" + current[end:]
    replacements = (
        ('  "schema_version": 5,', '  "schema_version": 4,'),
        (
            '  "status": "task_screen_not_executed; provisional pose cache retained; v5 human-coordinate UI defect discovered and final gate blocked",',
            '  "status": "prospective_task_screen_not_executed; native_crop_pilot_completed",',
        ),
        (
            '      "status": "computational_mismatch_corrected; pilot preprocessing and flip replay pass; publisher byte identity remains unresolved",',
            '      "status": "computational_mismatch_corrected; publisher_byte_identity_and_pilot_preprocessing_lock_remain",',
        ),
        (
            '      "pilot_preprocessing_and_flip_lock_passed": true,',
            '      "pilot_preprocessing_and_flip_lock_passed": false,',
        ),
        (
            '    "human_reference": {"reviewers": 2, "blinded_to": ["pose_estimates", "action_labels", "ARFTR_outputs", "error_membership"], "visibility_disagreements_resolved_before_final_gate": true, "adjudication_remains_blinded_to_pose_outputs": true, "use": "feasibility_only_never_training_or_policy", "coordinate_status": "reviewer_a visibility decisions retained; v5 coordinates invalid for final scoring due to UI geometry defect; CSS inverse is diagnostic only", "valid_reviewer": ".runs/research_20260913/body_witness_pilot_v1/blinded_review/index_v6_interactive.html", "interim_single_review": {"authorized_scope": ["frozen_pose_cache_preservation", "provisional_target_sensitivity_diagnostic", "pose_blind_css_geometry_diagnostic"], "forbidden_scope": ["final_availability_pass", "task_model_fitting", "route_promotion", "threshold_revision", "treating_css_inverse_as_final_reference"], "second_review_still_required": true}},',
            '    "human_reference": {"reviewers": 2, "blinded_to": ["pose_estimates", "action_labels", "ARFTR_outputs", "error_membership"], "visibility_disagreements_resolved_before_predictions": true, "use": "feasibility_only_never_training_or_policy", "interim_single_review": {"authorized_scope": ["frozen_pose_extraction", "provisional_availability_diagnostic", "target_sensitivity_diagnostic"], "forbidden_scope": ["final_availability_pass", "task_model_fitting", "route_promotion", "threshold_revision"], "second_review_still_required": true}},',
        ),
        (
            '    {"if": "computational_pose_compatibility_or_pilot_preprocessing_lock_missing", "then": "native_crop_and_blinded_review_materials_allowed; no_pose_quality_or_target_sensitivity_outputs"},',
            '    {"if": "pose_compatibility_or_exact_pose_model_lock_missing", "then": "native_crop_and_blinded_review_materials_allowed; no_pose_quality_or_target_sensitivity_outputs"},',
        ),
        (
            '    "currently_authorized": ["protocol_preparation", "label_blind_dependency_planning", "synthetic_focused_tests", "corrected_v6_blinded_review", "preservation_and_audit_of_existing_frozen_pose_cache", "pose_blind_review_geometry_diagnostics"],',
            '    "currently_authorized": ["protocol_preparation", "label_blind_dependency_planning", "synthetic_focused_tests", "label_blind_native_crop_extraction_and_review_materials"],',
        ),
        (
            '    "currently_blocked": ["final_pose_availability_result_until_two_valid_reviews_and_adjudication", "task_training_fits_until_all_prior_gates_and_execution_authority", "nested_integration_due_to_insufficient_scenario_ancestors"],',
            '    "currently_blocked": ["pretrained_pose_quality_outputs_until_source_model_compatibility_lock", "task_training_fits_until_all_prior_gates_and_execution_authority", "nested_integration_due_to_insufficient_scenario_ancestors"],',
        ),
    )
    for new, old in replacements:
        if value.count(new) != 1:
            raise RuntimeError(f"Schema-5 source does not contain exactly one expected field: {new}")
        value = value.replace(new, old)
    return value


def main() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    reconstructed = reconstruct_schema4(current).encode("utf-8")
    observed = sha256_bytes(reconstructed)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"Reconstructed schema-4 hash differs: {observed}")
    with OUTPUT.open("xb") as stream:
        stream.write(reconstructed)
    receipt = {
        "status": "EXACT_SCHEMA4_PROTOCOL_BYTES_PRESERVED",
        "artifact": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": observed,
        "current_schema5_sha256": sha256_bytes(current.encode("utf-8")),
        "method": "deterministic reversal of the sole schema-5 UI-defect amendment",
    }
    with RECEIPT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
