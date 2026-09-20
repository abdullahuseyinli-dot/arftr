"""Label-free invariants for PCAR forensic analysis and protocol design."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from experiments.analyze_okutama_pcar_design import (
    P2_LOCKED_MEAN,
    PRIMARY_ERROR_CEILING,
    PRIMARY_RELATIVE_GATE,
    binary_auc,
    build_protocol,
    convex_candidate,
    oracle_reduction,
    validate_protocol,
)


def minimal_findings() -> dict[str, object]:
    return {
        "scope": "synthetic label-blind test",
        "zero_access_counters": {
            "task_labels": 0,
            "ARFTR_probability_arrays": 0,
            "pose_outputs": 0,
            "annotation_derived_support_categories": 0,
            "clip_index": 0,
            "optimizer_updates": 0,
            "model_fits": 0,
            "router_fits": 0,
        },
    }


def test_oracle_math_distinguishes_center_and_token_routing() -> None:
    p2_tokens = np.array([[0.2, 0.8], [0.5, 0.5]])
    p3_tokens = np.array([[0.8, 0.2], [0.4, 0.6]])
    p2_centers = p2_tokens.mean(axis=1)
    p3_centers = p3_tokens.mean(axis=1)
    center_oracle = oracle_reduction(p2_centers, p3_centers)
    token_oracle_error = np.minimum(p2_tokens, p3_tokens).mean(axis=1)
    token_oracle = float(
        (p2_centers.mean() - token_oracle_error.mean()) / p2_centers.mean()
    )
    assert center_oracle == pytest.approx(0.0)
    assert token_oracle == pytest.approx(0.35)
    assert token_oracle > center_oracle


def test_binary_auc_is_tie_aware_and_has_no_fit() -> None:
    labels = np.array([False, False, True, True])
    assert binary_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert binary_auc(labels, np.ones(4)) == 0.5
    assert binary_auc(np.ones(4, dtype=bool), np.arange(4.0)) is None
    with pytest.raises(ValueError):
        binary_auc(labels, np.array([0.1, np.nan, 0.8, 0.9]))


def test_protocol_preserves_p2_and_the_failed_five_percent_gate() -> None:
    protocol = build_protocol({"synthetic": "hash"}, minimal_findings())
    validate_protocol(protocol)
    assert protocol["architecture"]["incumbent"]["action"] == "exact_P2_prediction"
    assert (
        protocol["architecture"]["competitive_gate"]["actions"][0]
        == "alpha_0_retain_P2_exact"
    )
    assert protocol["architecture"]["competitive_gate"]["inference_threshold"] is None
    assert protocol["architecture"]["competitive_gate"]["tie_action"] == (
        "alpha_0_retain_P2_exact"
    )
    assert protocol["architecture"]["evidence_construction"]["feature_width"] == 768
    assert protocol["architecture"]["evidence_construction"]["projection_before_fusion"] is False
    assert protocol["architecture"]["bounded_residual_actions"]["interior_equation"] == (
        "prediction(alpha)=P2 + alpha*(affine_uniform_768-P2)"
    )
    primary = protocol["go_no_go_gates"]["primary"]
    assert primary["relative_reduction_vs_P2_min"] == PRIMARY_RELATIVE_GATE
    assert primary["mean_center_cosine_error_max"] == pytest.approx(
        P2_LOCKED_MEAN * 0.95
    )
    assert PRIMARY_ERROR_CEILING == pytest.approx(0.36466429613064975)


def test_protocol_has_nested_leakage_barrier_and_no_authorization() -> None:
    protocol = build_protocol({"synthetic": "hash"}, minimal_findings())
    assert protocol["splits"]["outer"]["folds"] == 5
    assert protocol["splits"]["inner"]["folds_per_outer_train"] == 4
    assert protocol["splits"]["inner"]["outer_held_teacher_access"] == 0
    assert protocol["splits"]["inner"]["nested_base_pairs"]["unique_unordered_pairs"] == 10
    assert protocol["compute"]["nested_P0_fits"] == 10
    assert protocol["compute"]["nested_P2_fits"] == 10
    assert protocol["compute"]["total_new_fits_max"] == 30
    assert protocol["splits"]["inner"]["policy_selection"] == "none"
    assert protocol["authorization"]["pcar_training_authorized"] is False
    assert protocol["authorization"]["task_training_authorized"] is False
    assert "task_labels" in protocol["prohibited_inputs"]
    assert "ARFTR_probability_arrays" in protocol["prohibited_inputs"]
    assert "annotation_derived_support_categories" in protocol["prohibited_inputs"]
    assert protocol["compute"]["estimated_runtime_minutes_same_GPU"][1] < 20


def test_protocol_validation_rejects_post_hoc_relaxation() -> None:
    protocol = build_protocol({"synthetic": "hash"}, minimal_findings())
    protocol["architecture"]["competitive_gate"]["inference_threshold"] = 0.5
    with pytest.raises(ValueError, match="may not use a fitted threshold"):
        validate_protocol(protocol)
    protocol = build_protocol({"synthetic": "hash"}, minimal_findings())
    protocol["go_no_go_gates"]["primary"]["relative_reduction_vs_P2_min"] = 0.04
    with pytest.raises(ValueError, match="may not be relaxed"):
        validate_protocol(protocol)


def test_convex_endpoints_are_byte_and_hash_exact() -> None:
    p2 = np.array([[1.0, -0.0, 3.25]], dtype=np.float32)
    affine = np.array([[9.0, 2.0, -7.5]], dtype=np.float32)
    retained = convex_candidate(p2, affine, 0.0)
    direct = convex_candidate(p2, affine, 1.0)
    assert retained is not p2 and direct is not affine
    assert retained.tobytes() == p2.tobytes()
    assert direct.tobytes() == affine.tobytes()
    assert hashlib.sha256(retained.tobytes()).digest() == hashlib.sha256(p2.tobytes()).digest()
    assert hashlib.sha256(direct.tobytes()).digest() == hashlib.sha256(affine.tobytes()).digest()
    interior = convex_candidate(p2, affine, 0.5)
    np.testing.assert_array_equal(interior, np.array([[5.0, 1.0, -2.125]], dtype=np.float32))
