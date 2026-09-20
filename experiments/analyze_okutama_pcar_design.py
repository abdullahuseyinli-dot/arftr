"""Materialize tokenwise PCAR forensics and a no-training protocol lock.

This entry point is intentionally read-only with respect to the scientific inputs.
It replays the already-fitted P2/P3 checkpoints on the independently audited,
label-blind completion caches.  It never opens task labels, ARFTR probabilities,
pose outputs, support categories, images, or annotation manifests, and it performs
no optimization.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from hac.center_completion_factorial import cosine_errors, uniform_consensus
from hac.center_completion_screen import new_screen_model
from hac.center_evidence_completion import (
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / ".runs/research_20260913/center_evidence_completion_screen_v1"
CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
CACHE_AUDIT = (
    ROOT
    / ".runs/research_20260913/center_evidence_completion_cache_v1_audit/audit_receipt.json"
)
TRANSPORT = ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1"
TRANSPORT_AUDIT = (
    ROOT
    / ".runs/research_20260913/center_evidence_completion_transport_v1_audit/audit_receipt.json"
)
FACTORIAL = ROOT / ".runs/research_20260913/center_completion_matched_factorial_v1"
FACTORIAL_AUDIT = (
    ROOT
    / ".runs/research_20260913/center_completion_matched_factorial_v1_audit/audit_receipt.json"
)
SCREEN_LOADER = ROOT / "experiments/screen_okutama_center_evidence_completion.py"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/center_completion_pcar_design_v5"

EXPECTED_SCREEN_HASHES = {
    "summary.json": "002ad504f36de7e59ac5d345abacbaff548eca9fac718e1661dca95dad0306cd",
    "oof_metrics.npz": "6e6bf70e31d011ce69a5e359ee9beb572e758a41e3c5190b22ef9498cdb54b40",
    "per_center_oof.csv": "dce86e36d3186df3a415e124f8bbc76e728b17d8a17a41b1b3762a933069c355",
    "request.json": "7212655af4d7c0d0ad0fd9f73c6fce96dffb61e9708f177adc2c73672a9459ad",
}
EXPECTED_SOURCE_HASHES = {
    "experiments/screen_okutama_center_evidence_completion.py": (
        "6e6544234612123129f88d609c9a81eb7a61f72280ed22a1b58baf2fed47f274"
    ),
    "src/hac/center_completion_screen.py": (
        "582a3f81b56dae749fb98691dc07eea65f0e409bd25f95158ce9afd870a37db7"
    ),
    "src/hac/center_completion_factorial.py": (
        "5d9220b6fe8adac5fb66a16a67c5e67831118cfcbc8227753711aec6d361d1b9"
    ),
    "experiments/run_okutama_center_completion_matched_factorial.py": (
        "2ef8f3d27f2a319e4347a7376968a715fb011d9e4734745b98fb7b340eaa8a87"
    ),
}
EXPECTED_FACTORIAL_HASHES = {
    "summary.json": "70b8fa965f24058c8617c651f8132d89d416cba1495dd675633792e41c6bb919",
    "request.json": "20bd0b4adf357a212dd38c369ee650b04e2e79b00690eaa3e1e6c10dcf162d05",
    "per_center.csv": "9f47f29c967650e49a67558c0c0f2cbae261030fd3d6888865c1948a10747beb",
    "per_center_cosine_error.npy": (
        "2b219f9745ade27bb7d4d463bc5c6959315146933d3d7a389bb8e47d218438d2"
    ),
    "per_center_token_error_sum.npy": (
        "a2f7ba92a7974ffd6aedfee5042d568da6e69daa3b5f68a105c3d0c403547dd9"
    ),
    "target_count.npy": "7462fde0442ec3ccadc070e08563fe93968dd7a6885e122a1196897dfb4193f2",
}
FORBIDDEN_INPUT_TERMS = (
    "arftr",
    "task_label",
    "support_categor",
    "pose_output",
    "clip_index",
    "annotation_manifest",
)
OUTER_FOLDS = 5
CENTERS = 128
TARGET_TOKENS = 27_407
PRIMARY_RELATIVE_GATE = 0.05
P2_LOCKED_MEAN = 0.3838571538217366
P3_LOCKED_MEAN = 0.3691946086473763
PRIMARY_ERROR_CEILING = P2_LOCKED_MEAN * (1.0 - PRIMARY_RELATIVE_GATE)
AFFINE_UNIFORM_LOCKED_MEAN = 0.21304081300483318
AFFINE_UNIFORM_INCREMENTAL_GATE = 0.02
AFFINE_UNIFORM_INCREMENTAL_CEILING = (
    AFFINE_UNIFORM_LOCKED_MEAN * (1.0 - AFFINE_UNIFORM_INCREMENTAL_GATE)
)
CONVEX_ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 20_260_913


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return tie-aware binary ROC AUC without fitting any model."""

    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("binary AUC inputs must be finite, aligned vectors")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = stats.rankdata(scores, method="average")
    value = (ranks[labels].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return float(value)


def oracle_reduction(reference: np.ndarray, alternative: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    alternative = np.asarray(alternative, dtype=np.float64)
    if (
        reference.shape != alternative.shape
        or reference.size == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(alternative).all()
        or float(reference.mean()) <= 0
    ):
        raise ValueError("oracle inputs must be finite, aligned, and nonempty")
    return float((reference.mean() - np.minimum(reference, alternative).mean()) / reference.mean())


def convex_candidate(p2: np.ndarray, affine_uniform: np.ndarray, alpha: float) -> np.ndarray:
    """Construct a bounded action with byte-exact endpoint branches."""

    p2 = np.asarray(p2)
    affine_uniform = np.asarray(affine_uniform)
    if (
        p2.shape != affine_uniform.shape
        or p2.dtype != affine_uniform.dtype
        or not np.issubdtype(p2.dtype, np.floating)
        or not np.isfinite(p2).all()
        or not np.isfinite(affine_uniform).all()
        or alpha not in CONVEX_ACTIONS
    ):
        raise ValueError("convex PCAR action violates the locked aligned floating contract")
    if alpha == 0.0:
        return p2.copy()
    if alpha == 1.0:
        return affine_uniform.copy()
    return ((1.0 - alpha) * p2 + alpha * affine_uniform).astype(p2.dtype)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed if a core PCAR scientific invariant is absent or relaxed."""

    if protocol["status"] != "PCAR_LABEL_BLIND_PROTOCOL_LOCKED_NO_TRAINING_RUN":
        raise ValueError("PCAR protocol status changed")
    if protocol["authorization"]["task_training_authorized"] is not False:
        raise ValueError("task training must remain unauthorized")
    if protocol["authorization"]["pcar_training_authorized"] is not False:
        raise ValueError("this design artifact cannot authorize a PCAR fit")
    if protocol["architecture"]["incumbent"]["action"] != "exact_P2_prediction":
        raise ValueError("P2 must be an explicit exact-retain action")
    evidence = protocol["architecture"]["evidence_construction"]
    if (
        evidence["feature_width"] != 768
        or evidence["aggregation"] != "arithmetic mean over all donors in matched support"
        or evidence["projection_before_fusion"] is not False
        or evidence["componentwise_median"] is not False
    ):
        raise ValueError("the primary path must remain full-768 affine-uniform")
    if protocol["architecture"]["bounded_residual_actions"]["alphas"] != list(
        CONVEX_ACTIONS
    ):
        raise ValueError("the preregistered bounded convex action set changed")
    endpoints = protocol["architecture"]["bounded_residual_actions"]["endpoint_branches"]
    if endpoints != {
        "alpha_0": "return P2.clone() without arithmetic",
        "alpha_1": "return affine_uniform_768.clone() without arithmetic",
    }:
        raise ValueError("PCAR endpoints must bypass floating-point arithmetic")
    gate = protocol["architecture"]["competitive_gate"]
    if gate["inference_threshold"] is not None or gate["post_hoc_threshold_selection"] is not False:
        raise ValueError("the hard action decision may not use a fitted threshold")
    if gate["tie_action"] != "alpha_0_retain_P2_exact":
        raise ValueError("a policy tie must retain P2 exactly")
    outer = protocol["splits"]["outer"]
    inner = protocol["splits"]["inner"]
    if outer["folds"] != 5 or inner["folds_per_outer_train"] != 4:
        raise ValueError("the locked five-by-four nested split changed")
    if inner["outer_held_teacher_access"] != 0:
        raise ValueError("an outer held target entered nested gate fitting")
    primary = protocol["go_no_go_gates"]["primary"]
    if primary["relative_reduction_vs_P2_min"] != PRIMARY_RELATIVE_GATE:
        raise ValueError("the failed five-percent gate may not be relaxed")
    if not np.isclose(primary["mean_center_cosine_error_max"], PRIMARY_ERROR_CEILING):
        raise ValueError("the absolute PCAR gate no longer matches the locked P2 result")
    invention = protocol["go_no_go_gates"]["incremental_vs_affine_uniform"]
    if invention["relative_reduction_min"] != AFFINE_UNIFORM_INCREMENTAL_GATE:
        raise ValueError("the incremental affine-uniform gate changed")
    if not np.isclose(
        invention["mean_center_cosine_error_max"], AFFINE_UNIFORM_INCREMENTAL_CEILING
    ):
        raise ValueError("the incremental affine-uniform ceiling changed")
    if primary["positive_net_correction_required_in_every_outer_fold"] is not True:
        raise ValueError("every outer fold must remain positive")
    weighting = protocol["losses"]["policy_fit"]
    if "mean_over_centers" not in weighting["equation"] or "action harm magnitude" not in weighting["weighting"]:
        raise ValueError("policy loss must align with equal-center cosine risk")
    controls = protocol["controls_and_ablations"]["decision_ablations"]
    if "identically_trained_same_grid_uniform_policy_capacity_control" not in controls:
        raise ValueError("the matched-capacity same-grid control is mandatory")
    if protocol["compute"]["total_new_fits_max"] != 30:
        raise ValueError("compute must include nested bases and both policy families")
    nested = protocol["splits"]["inner"]["nested_base_pairs"]
    if nested["unique_unordered_pairs"] != 10 or "Never use" not in nested[
        "forbidden_substitute"
    ]:
        raise ValueError("inner base-level cross-fitting is not locked")
    if protocol["splits"]["inner"]["policy_selection"] != "none":
        raise ValueError("inner learned-policy selection would contaminate its comparator")
    estimands = protocol["controls_and_ablations"]["exact_control_estimands"]
    if estimands["gain_fraction"] != "F_k=G_k(S_k)/G_nom(S_k)":
        raise ValueError("causal-control gain fraction changed")
    if "intersect" not in estimands["wrong_track_support"]:
        raise ValueError("wrong-track common support is not exact")
    if "same_grid_trained_capacity_control_gain_fraction_lt" in protocol["go_no_go_gates"][
        "causal_controls"
    ]:
        raise ValueError("capacity attribution may not reject a useful promoted policy")
    if set(protocol["prohibited_inputs"]) != {
        "task_labels",
        "ARFTR_probability_arrays",
        "pose_outputs",
        "annotation_derived_support_categories",
        "clip_index",
        "held_fold_teacher_tokens_during_fit_or_selection",
    }:
        raise ValueError("the prohibited-input contract changed")
    command = protocol["execution"]["exact_training_command_placeholder"]
    if "--protocol" not in command or "--output-dir" not in command:
        raise ValueError("the future command is not exact enough to audit")


def build_protocol(parent_hashes: dict[str, Any], findings: dict[str, Any]) -> dict[str, Any]:
    oracle_precondition = (
        findings.get("primary_affine_uniform_full768", {})
        .get("P2_to_C_full768_convex_action_grid", {})
        .get("oracle_can_reach_locked_two_percent_incremental_gate")
    )
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "okutama_p2_protected_competitive_aligned_residual_v1",
        "status": "PCAR_LABEL_BLIND_PROTOCOL_LOCKED_NO_TRAINING_RUN",
            "scope": (
                "A new label-blind reconstruction experiment only. Passing it may authorize a "
                "separately versioned task protocol; it cannot change or claim ARFTR performance."
            ),
        "retain_semantics": (
            "PCAR retain means exact retention of the reconstruction-screen P2 token. It is not "
            "the later mandatory retain-ARFTR-class action, and cannot authorize or substitute "
            "for that task-level safeguard."
        ),
        "authorization": {
            "analysis_only_run_complete": True,
            "pcar_training_authorized": False,
            "task_training_authorized": False,
            "requires_new_versioned_execution_lock_before_fit": True,
            "retained_ARFTR_changed": False,
        },
        "superseded_design_artifacts": {
            "center_completion_pcar_design_v1": "rejected_obsolete_robust_median_primary_basis",
            "center_completion_pcar_design_v2": "rejected_inner_base_prediction_leakage",
            "center_completion_pcar_design_v3": "rejected_inner_base_prediction_leakage",
            "center_completion_pcar_design_v4": (
                "rejected_inner_learned_vs_fixed_selection_contamination"
            ),
            "reason": (
                "Those directories are immutable historical drafts. None authorized fitting. "
                "Only this schema fixes base-level inner cross-fitting."
            ),
        },
        "diagnostic_precondition": {
            "requirement": (
                "The no-fit full-768 P2-to-C convex action-grid token oracle must reach the "
                "locked two-percent improvement over direct C before a policy runner is implemented."
            ),
            "evaluated": oracle_precondition is not None,
            "passed": oracle_precondition,
            "effect": (
                "If false, stop PCAR policy work and retain direct affine-uniform C as the "
                "surviving representation."
            ),
        },
        "parent_evidence": parent_hashes,
        "locked_findings": findings,
        "hypothesis": {
            "statement": (
                "Actor-specific affine transport is strongly useful when its matched donors are "
                "averaged uniformly in the original 768-dimensional DINO space. P3 discarded "
                "much of that signal through componentwise robust median aggregation, a fixed "
                "128-dimensional projection, decoder reconstruction from P0, and a saturated gate."
            ),
            "falsifiable_prediction": (
                "A hard policy trained on pair-cross-fitted meta rows over full-768 convex corrections from P2 toward "
                "affine-uniform transport improves at least two percent over direct affine-uniform "
                "evidence, remains at least five percent better than P2, and is positive in every "
                "outer fold."
            ),
            "already_falsified_alternative": (
                "Routing only between the existing P2 and P3 predictions at center granularity; "
                "its diagnostic oracle reaches only 4.032611%, below the unchanged 5% gate."
            ),
            "discarded_mechanisms": [
                "componentwise confidence-weighted donor median",
                "fixed 768-to-128 evidence bottleneck",
                "reconstruction of the transport candidate from P0",
                "the saturated P3 intervention gate",
            ],
        },
        "prohibited_inputs": [
            "task_labels",
            "ARFTR_probability_arrays",
            "pose_outputs",
            "annotation_derived_support_categories",
            "clip_index",
            "held_fold_teacher_tokens_during_fit_or_selection",
        ],
        "permitted_inference_inputs": [
            "frozen_P0_prediction",
            "frozen_P2_prediction",
            "masked_center_DINO_patch_token",
            "same_grid_neighbor_pool_token",
            "true_track_affine_uniform_transport_token_on_matched_support",
            "fixed_21_transport_quality_features",
            "target_patch_position",
            "upper_or_lower_mask_identity",
            "transport_availability",
        ],
        "architecture": {
            "name": "P2-Protected Competitive Affine-Uniform Residual (PCAR)",
            "incumbent": {
                "source": "same outer-fold P2_same_grid_pool checkpoint",
                "frozen": True,
                "action": "exact_P2_prediction",
                "behavior_when_transport_unavailable": "exact_P2_prediction",
            },
            "evidence_construction": {
                "feature_width": 768,
                "geometry": "visible-anchor robust affine transport",
                "support": "intersection of affine-observed and same-grid-valid donor support",
                "aggregation": "arithmetic mean over all donors in matched support",
                "confidence_weighting": False,
                "componentwise_median": False,
                "projection_before_fusion": False,
            },
            "bounded_residual_actions": {
                "alphas": list(CONVEX_ACTIONS),
                "interior_equation": "prediction(alpha)=P2 + alpha*(affine_uniform_768-P2)",
                "endpoint_branches": {
                    "alpha_0": "return P2.clone() without arithmetic",
                    "alpha_1": "return affine_uniform_768.clone() without arithmetic",
                },
                "alpha_zero_action": "P2 bit-exact",
                "alpha_one_action": "affine_uniform_768 bit-exact",
                "extrapolation": False,
                "learned_768_direction": False,
                "behavior_when_transport_unavailable": "alpha=0; P2 bit-exact",
            },
            "competitive_gate": {
                "actions": [
                    "alpha_0_retain_P2_exact",
                    "alpha_0.25",
                    "alpha_0.5",
                    "alpha_0.75",
                    "alpha_1_affine_uniform_exact",
                ],
                "features_in_order": [
                    "existing_transport_features_0_through_20",
                    "cosine_distance(P2,P0)",
                    "normalized_L2(P2,P0)",
                    "cosine_distance(affine_uniform,P2)",
                    "normalized_L2(affine_uniform,P2)",
                    "cosine_distance(affine_uniform,same_grid_uniform)",
                    "normalized_L2(affine_uniform,same_grid_uniform)",
                    "cosine_distance(P2,masked_center)",
                    "cosine_distance(affine_uniform,masked_center)",
                    "upper_lower_mask_bit",
                    "normalized_patch_y",
                    "normalized_patch_x",
                ],
                "feature_count": 32,
                "network": [
                    "LayerNorm(32)",
                    "Linear(32,32)",
                    "GELU",
                    "Linear(32,5)",
                    "softmax_for_center-balanced expected-cost training only",
                ],
                "inference_threshold": None,
                "inference_decision": "hard_argmax_over_five_actions",
                "tie_action": "alpha_0_retain_P2_exact",
                "post_hoc_threshold_selection": False,
                "hard_inference": True,
            },
        },
        "splits": {
            "outer": {
                "folds": 5,
                "identity": "unchanged scenario-held folds saved in oof_metrics.npz",
                "held_teacher_use": "evaluation_only_after_checkpoint_and_threshold_are_frozen",
                "positive_net_correction_required_each_fold": True,
            },
            "inner": {
                "folds_per_outer_train": 4,
                "identity": "leave one remaining scenario-fold group out in each rotation",
                "nested_base_pairs": {
                    "unique_unordered_pairs": 10,
                    "pair_definition": "one outer-held group O and one inner-held group J",
                    "training_groups": "the other three of five scenario-fold groups",
                    "fits_per_pair": ["P0_center_only", "P2_same_grid_pool"],
                    "reuse_rule": (
                        "The base checkpoint for unordered pair {O,J} may serve both orientations, "
                        "but predictions are made only for a group excluded from that checkpoint."
                    ),
                    "forbidden_substitute": (
                        "Never use the existing global OOF P2 prediction to form inner meta rows; "
                        "its checkpoint was trained on the current outer-held group."
                    ),
                },
                "cross_fitted_meta_rows": (
                    "For outer O and each training group J!=O, predict J with P0/P2 trained on "
                    "groups excluding {O,J}; compute all five convex action costs on J."
                ),
                "policy_fits_per_outer": 1,
                "purpose": "construct leakage-safe cross-fitted meta rows for one fixed policy fit",
                "outer_held_teacher_access": 0,
                "policy_selection": "none",
                "policy_fit": (
                    "Fit the single preregistered five-action policy on all four cross-fitted "
                    "outer-training meta groups, then freeze it before evaluating outer-held O."
                ),
                "fixed_comparator": (
                    "alpha=1 direct affine-uniform C, fixed before outer evaluation; no learned-vs-fixed "
                    "inner selection and no fallback choice"
                ),
                "threshold_selection": "none; hard argmax with any score tie forced to alpha=0",
                "base_leakage_invariant": (
                    "Every policy-fit or policy-selection row is predicted by P0/P2 that excluded "
                    "both its own scenario group and the current outer-held scenario group."
                ),
            },
        },
        "losses": {
            "nested_base_fits": {
                "P0_architecture_and_loss": "bit-exact locked screen P0_center_only",
                "P2_architecture_and_loss": "bit-exact locked screen P2_same_grid_pool",
                "training_source": (
                    "Only centers from the three scenario groups outside unordered pair {O,J}; "
                    "teacher tokens from O or J are never read by that base fit."
                ),
                "updates_each": 400,
                "optimizer": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "batch_size": 64,
                "seed": 42,
                "early_stopping": False,
            },
            "policy_fit": {
                "target_cost": "raw full-768 token cosine error for each of the five actions",
                "equation": (
                    "mean_over_centers(mean_over_center_tokens(sum_actions(" 
                    "softmax(logits)*action_cosine_error)))"
                ),
                "weighting": (
                    "every center has equal total weight; action harm magnitude is retained; no "
                    "class balancing, binary correctness target, Huber term, or 0.5 threshold"
                ),
                "updates": 200,
                "early_stopping": False,
                "invalid_condition": "nonfinite action cost or an empty center",
            },
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "batch_size": 64,
            "seed": 42,
        },
        "controls_and_ablations": {
            "unchanged_checkpoint_controls": [
                "wrong_track_affine_uniform_on_true_wrong_common_donor_support",
                "repeated_masked_center_uniform_on_nominal_control_common_support",
                "spatial_reassignment_of_affine_uniform_evidence_within_same_mask_row",
                "time_reversal_descriptive_only",
            ],
            "decision_ablations": [
                "always_retain_P2",
                "always_use_affine_uniform_alpha_1",
                "fixed_alpha_0.25",
                "fixed_alpha_0.5",
                "fixed_alpha_0.75",
                "legacy_P2_P3_gate_secondary_only",
                "identically_trained_same_grid_uniform_policy_capacity_control",
            ],
            "causal_interpretation": (
                "Wrong/repeated/spatial controls are evaluated through unchanged PCAR policy "
                "checkpoints. Nominal and control scores use their exact common donor/target "
                "support. The capacity control alone is trained identically with C replaced by A."
            ),
            "exact_control_estimands": {
                "equal_center_error": (
                    "E_X(S)=mean_centers(mean_target_tokens_in_S_for_center(cosine_error_X))"
                ),
                "nominal_gain": "G_nom(S)=E_P2(S)-E_PCAR_true(S)",
                "control_gain": "G_k(S)=E_P2(S)-E_PCAR_control_k(S)",
                "gain_fraction": "F_k=G_k(S_k)/G_nom(S_k)",
                "spatial_removed_fraction": (
                    "R_spatial=(E_spatial(S)-E_true(S))/G_nom(S)=1-F_spatial"
                ),
                "wrong_track_support": (
                    "For each donor and target, intersect true-affine-observed, "
                    "wrong-affine-observed, and same-grid-valid; average true and wrong over the "
                    "identical surviving donor set; include targets with at least one donor."
                ),
                "repeated_support": (
                    "Use the primary matched true-affine/same-grid target and donor support for "
                    "both nominal and repeated masked-center evidence."
                ),
                "spatial_support": (
                    "Use the primary matched support and the locked deterministic within-mask-row "
                    "spatial permutation; availability and target denominators remain identical."
                ),
                "undefined_rule": "Fail attribution if G_nom(S_k)<=0 or any compared center is empty.",
            },
        },
        "metrics": {
            "primary": "mean of per-center mean target-token cosine error",
            "secondary": [
                "LayerNorm Huber error",
                "token rescue mass",
                "token harm mass",
                "hard retain fraction",
                "per-fold and per-scenario error",
                "paired center bootstrap difference versus direct affine-uniform C",
                "hard action fractions and regret versus the action-grid oracle",
            ],
            "bootstrap": {
                "unit": "center",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAPS,
                "interval": "percentile_95",
            },
        },
        "go_no_go_gates": {
            "primary": {
                "baseline": "P2_same_grid_pool",
                "baseline_mean_center_cosine_error": P2_LOCKED_MEAN,
                "relative_reduction_vs_P2_min": PRIMARY_RELATIVE_GATE,
                "mean_center_cosine_error_max": PRIMARY_ERROR_CEILING,
                "positive_net_correction_required_in_every_outer_fold": True,
                "scenarios_strictly_better_than_P2_min": 10,
            },
            "incremental_vs_affine_uniform": {
                "baseline": "C_affine_uniform_full768_matched_support",
                "baseline_mean_center_cosine_error": AFFINE_UNIFORM_LOCKED_MEAN,
                "relative_reduction_min": AFFINE_UNIFORM_INCREMENTAL_GATE,
                "mean_center_cosine_error_max": AFFINE_UNIFORM_INCREMENTAL_CEILING,
                "positive_net_correction_required_in_every_outer_fold": True,
                "paired_bootstrap_95_percent_lower_bound_for_C_minus_PCAR": ">0",
                "token_rescue_mass_divided_by_harm_mass_min": 3.0,
                "hard_policy_no_worse_than_direct_affine_uniform": True,
                "policy_training_and_outer_comparison_use_equal_center_mean_cosine_only": True,
            },
            "learned_policy_promotion": {
                "comparator": "fixed alpha=1 direct affine-uniform C",
                "selection": "none",
                "failure_effect": (
                    "Reject the learned PCAR policy. Retain direct affine-uniform C only as the "
                    "surviving representation; do not substitute a post-hoc fixed alpha."
                ),
            },
            "causal_controls": {
                "wrong_track_gain_fraction_lt": 0.5,
                "repeated_masked_center_gain_fraction_lt": 0.5,
                "spatial_reassignment_removed_gain_fraction_min": 0.5,
                "time_reversal": "descriptive; no directionality claim or gate",
            },
            "affine_specific_attribution_not_promotion": {
                "same_grid_trained_capacity_control_gain_fraction_lt": 0.5,
                "failure_effect": (
                    "Do not reject a policy that passes every promotion gate. Retain it as a "
                    "generic full-768 convex-fusion result, but forbid an affine-specific mechanism claim."
                ),
            },
            "integrity": {
                "exact_source_and_input_hashes": True,
                "same_code_isolated_exact_replay": True,
                "independent_algorithm_replication_claimed": False,
                "held_teacher_access_during_fit_or_selection": 0,
                "all_prohibited_input_access_counters": 0,
                "post_hoc_gate_or_threshold_change": 0,
                "inner_meta_rows_from_in_sample_base_predictions": 0,
                "existing_global_OOF_P2_used_for_inner_meta_rows": 0,
            },
            "all_required": True,
        },
        "compute": {
            "new_DINO_forwards": 0,
            "reused_final_outer_P0_P2_checkpoint_pairs_after_selection_only": 5,
            "unique_nested_base_pairs": 10,
            "nested_P0_fits": 10,
            "nested_P2_fits": 10,
            "nominal_policy_fits": 5,
            "identical_same_grid_capacity_control_fits": 5,
            "total_new_fits_max": 30,
            "total_optimizer_updates_max": 10_000,
            "estimated_runtime_minutes_same_GPU": [10, 13],
            "hard_runtime_launch_limit_minutes": 20,
            "runtime_basis": (
                "20 nested base fits projected from the audited 20-fit/561-second screen (~9.35 "
                "minutes) plus 10 much smaller 32-input, five-logit policy fits."
            ),
        },
        "stop_conditions": [
            "Do not launch if the measured dry-run projection exceeds 20 minutes.",
            "Reject any implementation that forms inner policy rows from a base trained on that row's group or on the outer-held group.",
            "Do not implement or launch the policy if the locked convex action oracle cannot clear two percent versus C.",
            "Abort immediately on prohibited-input access, split overlap, nonfinite action cost, or an empty center.",
            "Do not authorize task fitting unless every reconstruction and integrity gate passes unchanged.",
            "Abandon learned PCAR fusion if mean error exceeds 0.208780 or any outer fold fails to improve direct affine-uniform evidence.",
            "If PCAR fails, retain direct affine-uniform full-768 evidence as the surviving representation; do not revive robust median P3.",
            "Do not relax the five-percent P2 floor, two-percent C gate, fold rule, action set, or controls after results.",
        ],
        "execution": {
            "runner_status": "NOT_IMPLEMENTED_AND_NOT_AUTHORIZED_BY_THIS_ARTIFACT",
            "exact_training_command_placeholder": (
                ".\\.venv\\Scripts\\python.exe experiments\\run_okutama_pcar.py "
                "--protocol .runs\\research_20260913\\center_completion_pcar_design_v5\\pcar_protocol.json "
                "--output-dir .runs\\research_20260913\\center_completion_pcar_screen_v1 "
                "--device cuda --stage all"
            ),
            "required_timing_smoke_command_placeholder": (
                ".\\.venv\\Scripts\\python.exe experiments\\run_okutama_pcar.py "
                "--protocol .runs\\research_20260913\\center_completion_pcar_design_v5\\pcar_protocol.json "
                "--output-dir .runs\\research_20260913\\center_completion_pcar_smoke_v1 "
                "--device cuda --stage timing-smoke"
            ),
            "required_before_command": (
                "Implement and test the runner, create a new execution lock pinning its source "
                "hash and this protocol hash, run a bounded timing smoke, and confirm <=20 minutes."
            ),
        },
    }
    validate_protocol(protocol)
    return protocol


def _load_screen_module() -> Any:
    if sha256_file(SCREEN_LOADER) != EXPECTED_SOURCE_HASHES[
        "experiments/screen_okutama_center_evidence_completion.py"
    ]:
        raise RuntimeError("the locked scientific screen loader changed")
    spec = importlib.util.spec_from_file_location("pcar_locked_screen_loader", SCREEN_LOADER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the locked screen entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_center_rows() -> list[dict[str, str]]:
    with (SCREEN / "per_center_oof.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != CENTERS or [int(row["center_index"]) for row in rows] != list(range(CENTERS)):
        raise RuntimeError("center result order changed")
    return rows


def _checkpoint_hashes() -> dict[str, str]:
    result = {}
    for fold in range(OUTER_FOLDS):
        for arm in ("P2_same_grid_pool", "P3_partial_transport"):
            name = f"fold{fold}_{arm}.pt"
            result[name] = sha256_file(SCREEN / "checkpoints" / name)
    return result


def _verify_array_bundle(directory: Path, summary_path: Path) -> dict[str, str]:
    summary = read_json(summary_path)
    observed = {}
    for name, receipt in sorted(summary["artifacts"].items()):
        digest = sha256_file(directory / name)
        if digest != receipt["sha256"]:
            raise RuntimeError(f"locked array changed: {directory.name}/{name}")
        observed[name] = digest
    return observed


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean requires finite values")
    return float(array.mean())


def _recompute_affine_uniform_tokenwise(
    p2_predictions: list[np.ndarray | None], center_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay matched A/C tokens and the no-fit P2-to-C convex action oracle."""

    teacher = np.load(CACHE / "teacher_tokens.npy", mmap_mode="r", allow_pickle=False)
    masked = np.load(CACHE / "masked_tokens.npy", mmap_mode="r", allow_pickle=False)
    neighbors = np.load(CACHE / "neighbor_tokens.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(CACHE / "target_masks.npy", mmap_mode="r", allow_pickle=False)
    visible = np.load(CACHE / "visible_masks.npy", mmap_mode="r", allow_pickle=False)
    neighbor_valid = np.load(CACHE / "neighbor_valid.npy", mmap_mode="r", allow_pickle=False)
    token_rows: list[dict[str, Any]] = []
    center_output: list[dict[str, Any]] = []
    factorial_rows: dict[int, dict[str, str]] = {}
    with (FACTORIAL / "per_center.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            factorial_rows[int(row["center_index"])] = row
    if set(factorial_rows) != set(range(CENTERS)):
        raise RuntimeError("matched-factorial center order changed")

    for center in range(CENTERS):
        p2_center = p2_predictions[center]
        if p2_center is None:
            raise RuntimeError(f"missing frozen P2 predictions for center {center}")
        cursor = 0
        center_a: list[np.ndarray] = []
        center_c: list[np.ndarray] = []
        center_p2: list[np.ndarray] = []
        center_oracle: list[np.ndarray] = []
        center_best: list[np.ndarray] = []
        center_action_errors: dict[float, list[np.ndarray]] = {
            alpha: [] for alpha in CONVEX_ACTIONS
        }
        for mask_id in range(2):
            target_mask = np.asarray(targets[center, mask_id], dtype=bool)
            positions = np.argwhere(target_mask)
            count = len(positions)
            donor_affine = np.zeros((4, count, 768), dtype=np.float32)
            donor_same = np.zeros_like(donor_affine)
            affine_observed = np.zeros((4, count), dtype=bool)
            same_valid = np.zeros_like(affine_observed)
            donor_logits = np.full((4, count), -np.inf, dtype=np.float64)
            center_tokens = np.asarray(masked[center, mask_id], dtype=np.float32)
            center_valid = np.asarray(visible[center, mask_id], dtype=bool)
            for donor in range(4):
                donor_tokens = np.asarray(neighbors[center, donor], dtype=np.float32)
                valid = np.asarray(neighbor_valid[center, donor], dtype=bool)
                transform, matches = fit_visible_anchor_transport(
                    center_tokens,
                    donor_tokens,
                    target_mask,
                    center_valid=center_valid,
                    neighbor_valid=valid,
                )
                values, observed = bilinear_transport(
                    donor_tokens, target_mask, transform, neighbor_valid=valid
                )
                y, x = positions[:, 0], positions[:, 1]
                donor_affine[donor] = values
                donor_same[donor] = donor_tokens[y, x]
                affine_observed[donor] = observed
                same_valid[donor] = valid[y, x]
                donor_logits[donor] = donor_confidence_logits(
                    matches, transform, target_mask, observed
                )
            common = affine_observed & same_valid
            if not np.all(common.any(axis=0)):
                raise RuntimeError("a matched factorial target lost all donors")
            cell_a = uniform_consensus(donor_same, common, donor_logits).features
            cell_c = uniform_consensus(donor_affine, common, donor_logits).features
            truth = np.asarray(teacher[center][target_mask], dtype=np.float32)
            p2 = np.asarray(p2_center[cursor : cursor + count], dtype=np.float32)
            if p2.shape != truth.shape:
                raise RuntimeError("P2 and matched-factorial token order diverged")
            error_a = cosine_errors(cell_a, truth)
            error_c = cosine_errors(cell_c, truth)
            error_p2 = cosine_errors(p2, truth)
            action_errors = np.stack(
                [
                    cosine_errors(convex_candidate(p2, cell_c, alpha), truth)
                    for alpha in CONVEX_ACTIONS
                ],
                axis=1,
            )
            best_action = np.argmin(action_errors, axis=1)
            oracle = np.take_along_axis(action_errors, best_action[:, None], axis=1)[:, 0]
            center_a.append(error_a)
            center_c.append(error_c)
            center_p2.append(error_p2)
            center_oracle.append(oracle)
            center_best.append(best_action)
            for action_index, alpha in enumerate(CONVEX_ACTIONS):
                center_action_errors[alpha].append(action_errors[:, action_index])
            for token in range(count):
                token_rows.append(
                    {
                        "center_index": center,
                        "token_index_within_center": cursor + token,
                        "mask_id": mask_id,
                        "patch_y": int(positions[token, 0]),
                        "patch_x": int(positions[token, 1]),
                        "A_same_grid_uniform_cosine_error": float(error_a[token]),
                        "C_affine_uniform_cosine_error": float(error_c[token]),
                        "P2_cosine_error": float(error_p2[token]),
                        "convex_action_oracle_cosine_error": float(oracle[token]),
                        "convex_action_oracle_alpha": CONVEX_ACTIONS[int(best_action[token])],
                        "C_better_than_A": int(error_c[token] < error_a[token]),
                        "C_better_than_P2": int(error_c[token] < error_p2[token]),
                        **{
                            f"alpha_{alpha}_cosine_error": float(
                                action_errors[token, action_index]
                            )
                            for action_index, alpha in enumerate(CONVEX_ACTIONS)
                        },
                    }
                )
            cursor += count
        a = np.concatenate(center_a)
        c = np.concatenate(center_c)
        p2 = np.concatenate(center_p2)
        oracle = np.concatenate(center_oracle)
        actions = np.concatenate(center_best)
        saved = factorial_rows[center]
        if (
            abs(a.mean() - float(saved["A_same_grid_uniform_equal_center_cosine"])) > 1e-10
            or abs(c.mean() - float(saved["C_affine_uniform_equal_center_cosine"])) > 1e-10
        ):
            raise RuntimeError(f"factorial token replay disagrees at center {center}")
        center_output.append(
            {
                "center_index": center,
                "sample_id": center_rows[center]["sample_id"],
                "scenario": center_rows[center]["scenario"],
                "held_fold": int(center_rows[center]["held_fold"]),
                "target_tokens": len(a),
                "A_same_grid_uniform_mean_cosine_error": float(a.mean()),
                "C_affine_uniform_mean_cosine_error": float(c.mean()),
                "P2_mean_cosine_error": float(p2.mean()),
                "convex_action_oracle_mean_cosine_error": float(oracle.mean()),
                "C_token_win_fraction_vs_A": float((c < a).mean()),
                "C_token_win_fraction_vs_P2": float((c < p2).mean()),
                "oracle_retain_P2_fraction": float((actions == 0).mean()),
                "oracle_direct_C_fraction": float((actions == len(CONVEX_ACTIONS) - 1).mean()),
                **{
                    f"alpha_{alpha}_mean_cosine_error": float(
                        np.concatenate(center_action_errors[alpha]).mean()
                    )
                    for alpha in CONVEX_ACTIONS
                },
            }
        )
    return token_rows, center_output


def main() -> None:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {DEFAULT_OUTPUT}")
    observed_screen = {
        name: sha256_file(SCREEN / name) for name in EXPECTED_SCREEN_HASHES
    }
    if observed_screen != EXPECTED_SCREEN_HASHES:
        raise RuntimeError("locked completion-screen evidence changed")
    observed_source = {
        name: sha256_file(ROOT / name) for name in EXPECTED_SOURCE_HASHES
    }
    if observed_source != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("locked reconstruction source changed")
    observed_factorial = {
        name: sha256_file(FACTORIAL / name) for name in EXPECTED_FACTORIAL_HASHES
    }
    if observed_factorial != EXPECTED_FACTORIAL_HASHES:
        raise RuntimeError("locked matched-factorial evidence changed")
    screen_summary = read_json(SCREEN / "summary.json")
    if (
        screen_summary["status"] != "CENTER_COMPLETION_RECONSTRUCTION_SCREEN_COMPLETE"
        or screen_summary["scientific"] is not True
        or screen_summary["gates"]["all_pass"] is not False
        or screen_summary["task_training_authorized"] is not False
        or screen_summary["retained_arftr_changed"] is not False
        or any(screen_summary["zero_access_counters"].values())
    ):
        raise RuntimeError("scientific result or zero-access boundary changed")
    cache_audit = read_json(CACHE_AUDIT)
    transport_audit = read_json(TRANSPORT_AUDIT)
    factorial_summary = read_json(FACTORIAL / "summary.json")
    factorial_audit = read_json(FACTORIAL_AUDIT)
    if (
        cache_audit["status"] != "INDEPENDENT_FULL_CACHE_REPLAY_AUDIT_PASS"
        or transport_audit["status"] != "INDEPENDENT_TRANSPORT_CACHE_REPLAY_AUDIT_PASS"
        or factorial_audit["status"] != "CENTER_COMPLETION_MATCHED_FACTORIAL_EXACT_REPLAY_PASS"
    ):
        raise RuntimeError("independent input replay is not locked PASS")
    if (
        factorial_summary["status"] != "CENTER_COMPLETION_MATCHED_FACTORIAL_COMPLETE"
        or factorial_summary["locked_decision"]["verdict"]["transport_survives"] is not True
        or factorial_summary["locked_decision"]["verdict"]["aggregation_survives"] is not False
        or any(factorial_summary["zero_access_counters"].values())
    ):
        raise RuntimeError("matched-factorial scientific boundary changed")

    input_hashes: dict[str, Any] = {
        "screen": observed_screen,
        "matched_factorial": observed_factorial,
        "source": observed_source,
        "checkpoints": _checkpoint_hashes(),
        "cache_arrays": _verify_array_bundle(CACHE, CACHE / "summary.json"),
        "transport_arrays": _verify_array_bundle(TRANSPORT, TRANSPORT / "summary.json"),
        "audit_receipts": {
            str(CACHE_AUDIT.relative_to(ROOT)): sha256_file(CACHE_AUDIT),
            str(TRANSPORT_AUDIT.relative_to(ROOT)): sha256_file(TRANSPORT_AUDIT),
            str(FACTORIAL_AUDIT.relative_to(ROOT)): sha256_file(FACTORIAL_AUDIT),
        },
    }
    serialized_inputs = json.dumps(input_hashes, sort_keys=True).lower()
    if any(term in serialized_inputs for term in FORBIDDEN_INPUT_TERMS):
        # "arftr" occurs only in the explicitly forbidden contract, never in this allowlist.
        matched = [term for term in FORBIDDEN_INPUT_TERMS if term in serialized_inputs]
        raise RuntimeError(f"forbidden term entered the file allowlist: {matched}")

    screen = _load_screen_module()
    cache = screen.load_cache()
    true, wrong, repeated = screen.load_transport_cache(TRANSPORT)
    packed = screen.pack_all(cache, true, wrong, repeated)
    center_rows = _load_center_rows()

    token_rows: list[dict[str, Any]] = []
    derived_centers: list[dict[str, Any]] = []
    p2_predictions: list[np.ndarray | None] = [None] * CENTERS
    fold_stats: dict[str, Any] = {}
    for fold in range(OUTER_FOLDS):
        held = np.asarray(
            [index for index, row in enumerate(center_rows) if int(row["held_fold"]) == fold],
            dtype=np.int64,
        )
        if held.size == 0:
            raise RuntimeError(f"outer fold {fold} is empty")
        outputs = {}
        for arm in ("P2_same_grid_pool", "P3_partial_transport"):
            checkpoint = torch.load(
                SCREEN / "checkpoints" / f"fold{fold}_{arm}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                checkpoint["arm"] != arm
                or checkpoint["held_fold"] != fold
                or checkpoint["updates"] != 400
                or not np.array_equal(np.asarray(checkpoint["held_rows"]), held)
            ):
                raise RuntimeError(f"checkpoint split lock changed: fold {fold}/{arm}")
            dummy_base = new_screen_model("P0_center_only", device="cpu")
            model = new_screen_model(arm, base=dummy_base, device="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            with torch.inference_mode():
                outputs[arm] = model(packed[arm].select(held))

        data = packed["P2_same_grid_pool"].select(held)
        teacher = data.teacher_tokens.float()
        p2_error = 1.0 - F.cosine_similarity(
            outputs["P2_same_grid_pool"].prediction, teacher, dim=-1, eps=1e-12
        )
        p3_error = 1.0 - F.cosine_similarity(
            outputs["P3_partial_transport"].prediction, teacher, dim=-1, eps=1e-12
        )
        gate = outputs["P3_partial_transport"].gate
        valid = data.target_valid
        flat_p2 = p2_error[valid].cpu().numpy().astype(np.float64)
        flat_p3 = p3_error[valid].cpu().numpy().astype(np.float64)
        flat_gate = gate[valid].cpu().numpy().astype(np.float64)
        wins = flat_p3 < flat_p2
        fold_stats[str(fold)] = {
            "centers": int(held.size),
            "tokens": int(valid.sum()),
            "P2_mean_token_cosine_error": float(flat_p2.mean()),
            "P3_mean_token_cosine_error": float(flat_p3.mean()),
            "token_oracle_mean_cosine_error": float(np.minimum(flat_p2, flat_p3).mean()),
            "P3_token_win_fraction": float(wins.mean()),
            "P3_gate_token_AUC_for_P3_win": binary_auc(wins, flat_gate),
        }
        for local_index, center_index in enumerate(held.tolist()):
            center_valid = valid[local_index]
            e2 = p2_error[local_index, center_valid].cpu().numpy().astype(np.float64)
            e3 = p3_error[local_index, center_valid].cpu().numpy().astype(np.float64)
            gates = gate[local_index, center_valid].cpu().numpy().astype(np.float64)
            positions = data.positions_yx[local_index, center_valid].cpu().numpy()
            masks = data.mask_ids[local_index, center_valid].cpu().numpy()
            center_wins = e3 < e2
            p2_predictions[center_index] = (
                outputs["P2_same_grid_pool"]
                .prediction[local_index, center_valid]
                .cpu()
                .numpy()
                .astype(np.float32)
                .copy()
            )
            stored_p2 = float(center_rows[center_index]["P2_same_grid_pool_cosine"])
            stored_p3 = float(center_rows[center_index]["P3_partial_transport_cosine"])
            if abs(e2.mean() - stored_p2) > 2e-6 or abs(e3.mean() - stored_p3) > 2e-6:
                raise RuntimeError(f"CPU token replay disagrees with saved center {center_index}")
            derived_centers.append(
                {
                    "center_index": center_index,
                    "sample_id": center_rows[center_index]["sample_id"],
                    "scenario": center_rows[center_index]["scenario"],
                    "held_fold": fold,
                    "target_tokens": int(center_valid.sum()),
                    "P2_mean_cosine_error": float(e2.mean()),
                    "P3_mean_cosine_error": float(e3.mean()),
                    "token_oracle_mean_cosine_error": float(np.minimum(e2, e3).mean()),
                    "P3_token_win_fraction": float(center_wins.mean()),
                    "P3_mean_gate": float(gates.mean()),
                    "P3_gate_token_AUC_for_P3_win": binary_auc(center_wins, gates),
                }
            )
            for token_index in range(len(e2)):
                token_rows.append(
                    {
                        "center_index": center_index,
                        "token_index_within_center": token_index,
                        "mask_id": int(masks[token_index]),
                        "patch_y": int(positions[token_index, 0]),
                        "patch_x": int(positions[token_index, 1]),
                        "P2_cosine_error": float(e2[token_index]),
                        "P3_cosine_error": float(e3[token_index]),
                        "token_oracle_cosine_error": float(min(e2[token_index], e3[token_index])),
                        "P3_gate": float(gates[token_index]),
                        "P3_wins": int(center_wins[token_index]),
                    }
                )

    derived_centers.sort(key=lambda row: row["center_index"])
    if len(derived_centers) != CENTERS or len(token_rows) != TARGET_TOKENS:
        raise RuntimeError("the locked diagnostic population changed")
    primary_token_rows, primary_center_rows = _recompute_affine_uniform_tokenwise(
        p2_predictions, center_rows
    )
    if len(primary_token_rows) != TARGET_TOKENS or len(primary_center_rows) != CENTERS:
        raise RuntimeError("the affine-uniform diagnostic population changed")
    center_p2 = np.asarray([row["P2_mean_cosine_error"] for row in derived_centers])
    center_p3 = np.asarray([row["P3_mean_cosine_error"] for row in derived_centers])
    center_oracle = np.asarray(
        [row["token_oracle_mean_cosine_error"] for row in derived_centers]
    )
    token_p2 = np.asarray([row["P2_cosine_error"] for row in token_rows])
    token_p3 = np.asarray([row["P3_cosine_error"] for row in token_rows])
    token_gate = np.asarray([row["P3_gate"] for row in token_rows])
    token_wins = token_p3 < token_p2
    center_level_oracle = np.minimum(center_p2, center_p3)
    token_oracle_relative = float(
        (center_p2.mean() - center_oracle.mean()) / center_p2.mean()
    )
    extra_absolute = float(center_p3.mean() - center_oracle.mean())
    required_absolute = float(center_p3.mean() - PRIMARY_ERROR_CEILING)
    primary_a = np.asarray(
        [row["A_same_grid_uniform_mean_cosine_error"] for row in primary_center_rows]
    )
    primary_c = np.asarray(
        [row["C_affine_uniform_mean_cosine_error"] for row in primary_center_rows]
    )
    primary_p2 = np.asarray(
        [row["P2_mean_cosine_error"] for row in primary_center_rows]
    )
    primary_oracle = np.asarray(
        [row["convex_action_oracle_mean_cosine_error"] for row in primary_center_rows]
    )
    primary_token_best_alpha = np.asarray(
        [row["convex_action_oracle_alpha"] for row in primary_token_rows]
    )
    fixed_action_center_errors = {
        alpha: np.asarray(
            [row[f"alpha_{alpha}_mean_cosine_error"] for row in primary_center_rows]
        )
        for alpha in CONVEX_ACTIONS
    }
    if (
        abs(primary_a.mean() - factorial_summary["mean_equal_center_cosine_error"]["A"])
        > 1e-10
        or abs(primary_c.mean() - AFFINE_UNIFORM_LOCKED_MEAN) > 1e-10
        or np.max(np.abs(primary_p2 - center_p2)) > 2e-6
    ):
        raise RuntimeError("primary affine-uniform/P2 forensic replay changed")
    convex_oracle_incremental = float(
        (primary_c.mean() - primary_oracle.mean()) / primary_c.mean()
    )
    direct_transport_reduction = float(
        (primary_a.mean() - primary_c.mean()) / primary_a.mean()
    )
    fixed_action_means = {
        alpha: float(fixed_action_center_errors[alpha].mean()) for alpha in CONVEX_ACTIONS
    }
    best_fixed_alpha = min(CONVEX_ACTIONS, key=lambda alpha: (fixed_action_means[alpha], alpha))
    outer_selected_fixed = np.full(CENTERS, np.nan, dtype=np.float64)
    selected_alpha_by_fold: dict[str, float] = {}
    for fold in range(OUTER_FOLDS):
        training = np.asarray(
            [int(row["held_fold"]) != fold for row in primary_center_rows], dtype=bool
        )
        held = ~training
        selected = min(
            CONVEX_ACTIONS,
            key=lambda alpha: (float(fixed_action_center_errors[alpha][training].mean()), alpha),
        )
        selected_alpha_by_fold[str(fold)] = selected
        outer_selected_fixed[held] = fixed_action_center_errors[selected][held]
    if not np.isfinite(outer_selected_fixed).all():
        raise RuntimeError("outer-selected fixed-alpha diagnostic is incomplete")
    primary_fold_stats = {}
    for fold in range(OUTER_FOLDS):
        rows = np.asarray(
            [int(row["held_fold"]) == fold for row in primary_center_rows], dtype=bool
        )
        primary_fold_stats[str(fold)] = {
            "centers": int(rows.sum()),
            "A_same_grid_uniform": float(primary_a[rows].mean()),
            "C_affine_uniform": float(primary_c[rows].mean()),
            "convex_action_oracle": float(primary_oracle[rows].mean()),
            "C_relative_reduction_vs_A": float(
                (primary_a[rows].mean() - primary_c[rows].mean())
                / primary_a[rows].mean()
            ),
            "convex_oracle_relative_reduction_vs_C": float(
                (primary_c[rows].mean() - primary_oracle[rows].mean())
                / primary_c[rows].mean()
            ),
            "fixed_alpha_mean_center_errors": {
                str(alpha): float(fixed_action_center_errors[alpha][rows].mean())
                for alpha in CONVEX_ACTIONS
            },
        }
    findings = {
        "scope": "diagnostic label-blind reconstruction upper bounds; not deployable results",
        "population": {
            "centers": CENTERS,
            "target_tokens": TARGET_TOKENS,
            "outer_folds": OUTER_FOLDS,
            "scenarios": len({row["scenario"] for row in center_rows}),
        },
        "primary_affine_uniform_full768": {
            "status": "PRIMARY_DESIGN_BASIS",
            "aggregation_decision": {
                "retained": "affine_uniform_C",
                "discarded": "affine_robust_D_and_componentwise_weighted_median",
                "A_same_grid_uniform_mean_center_error": float(primary_a.mean()),
                "C_affine_uniform_mean_center_error": float(primary_c.mean()),
                "C_relative_reduction_vs_A": direct_transport_reduction,
                "C_vs_A_bootstrap_95_percent": factorial_summary["paired_bootstrap"]["C_vs_A"]["paired_center_bootstrap_95_percent"],
                "C_better_than_A_folds": factorial_summary["locked_decision"]["measurements"]["C_better_than_A_folds"],
                "D_relative_reduction_vs_C": factorial_summary["locked_decision"]["measurements"]["aggregation_affine_relative_reduction_D_vs_C"],
            },
            "P2_to_C_full768_convex_action_grid": {
                "alphas": list(CONVEX_ACTIONS),
                "P2_mean_center_error": float(primary_p2.mean()),
                "direct_C_mean_center_error": float(primary_c.mean()),
                "no_fit_token_action_oracle_mean_center_error": float(primary_oracle.mean()),
                "oracle_relative_reduction_vs_direct_C": convex_oracle_incremental,
                "oracle_can_reach_locked_two_percent_incremental_gate": bool(
                    convex_oracle_incremental >= AFFINE_UNIFORM_INCREMENTAL_GATE
                ),
                "direct_C_relative_reduction_vs_P2": float(
                    (primary_p2.mean() - primary_c.mean()) / primary_p2.mean()
                ),
                "oracle_action_token_fractions": {
                    str(alpha): float((primary_token_best_alpha == alpha).mean())
                    for alpha in CONVEX_ACTIONS
                },
                "fixed_alpha_mean_center_errors": {
                    str(alpha): fixed_action_means[alpha] for alpha in CONVEX_ACTIONS
                },
                "best_fixed_alpha_posthoc_diagnostic_only": best_fixed_alpha,
                "best_fixed_alpha_relative_reduction_vs_C": float(
                    (primary_c.mean() - fixed_action_means[best_fixed_alpha])
                    / primary_c.mean()
                ),
                "outer_train_selected_fixed_alpha_by_held_fold": selected_alpha_by_fold,
                "outer_train_selected_fixed_policy_mean_center_error": float(
                    outer_selected_fixed.mean()
                ),
                "outer_train_selected_fixed_policy_relative_reduction_vs_C": float(
                    (primary_c.mean() - outer_selected_fixed.mean()) / primary_c.mean()
                ),
                "fraction_of_convex_oracle_headroom_required_for_two_percent_gate": float(
                    (primary_c.mean() - AFFINE_UNIFORM_INCREMENTAL_CEILING)
                    / (primary_c.mean() - primary_oracle.mean())
                ),
                "interpretation": (
                    "This is the correct no-fit upper bound for the locked full-768 convex action "
                    "family. It must clear the two-percent C gate before any policy fit can be "
                    "authorized."
                ),
            },
            "folds": primary_fold_stats,
        },
        "legacy_P2_P3_secondary": {
            "status": "SECONDARY_OBSOLETE_AS_PRIMARY_DESIGN_BASIS",
            "reason": (
                "The matched factorial later established affine-uniform full-768 evidence as the "
                "surviving mechanism and robust median as harmful."
            ),
            "mean_center_cosine_error": {
                "P2_same_grid_pool": float(center_p2.mean()),
                "P3_partial_transport": float(center_p3.mean()),
                "center_level_P2_P3_oracle": float(center_level_oracle.mean()),
                "token_level_P2_P3_oracle": float(center_oracle.mean()),
            },
            "relative_reduction_vs_P2": {
                "P3": float((center_p2.mean() - center_p3.mean()) / center_p2.mean()),
                "center_level_oracle": oracle_reduction(center_p2, center_p3),
                "token_level_oracle": token_oracle_relative,
                "five_percent_gate": PRIMARY_RELATIVE_GATE,
            },
            "center_router_can_reach_five_percent": bool(
                oracle_reduction(center_p2, center_p3) >= PRIMARY_RELATIVE_GATE
            ),
            "token_opportunity": {
                "P3_win_tokens": int(token_wins.sum()),
                "P3_loss_or_tie_tokens": int((~token_wins).sum()),
                "P3_win_fraction": float(token_wins.mean()),
                "P3_gate_AUC_for_token_P3_win": binary_auc(token_wins, token_gate),
                "P3_mean_gate": float(token_gate.mean()),
                "P3_to_token_oracle_extra_absolute_error_reduction": extra_absolute,
                "additional_absolute_reduction_needed_to_clear_gate": required_absolute,
                "fraction_of_P3_to_token_oracle_headroom_required": float(
                    required_absolute / extra_absolute
                ),
            },
            "folds": fold_stats,
        },
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
        "interpretive_limits": [
            "Both oracle calculations inspect saved reconstruction targets and are diagnostic only.",
            "The convex action oracle is not a fitted or deployable policy.",
            "No threshold, model, residual, or router was fitted in this analysis.",
            "The existing five-percent P3-versus-P2 gate remains failed and unchanged.",
            "The later matched factorial supersedes robust-median P3 as the primary mechanism evidence.",
            "Passing a future PCAR reconstruction screen would not itself establish classification gain.",
        ],
    }

    request = {
        "status": "PCAR_TOKENWISE_ANALYSIS_REQUEST_LOCKED",
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": input_hashes,
        "allowed_roots": [
            str(SCREEN.relative_to(ROOT)),
            str(CACHE.relative_to(ROOT)),
            str(TRANSPORT.relative_to(ROOT)),
            str(FACTORIAL.relative_to(ROOT)),
        ],
        "operations": [
            "checkpoint_forward_replay_on_CPU",
            "matched_affine_uniform_replay",
            "full768_convex_action_grid",
            "cosine_error",
            "oracle_min",
            "ROC_AUC",
        ],
        "zero_access_contract": findings["zero_access_counters"],
        "optimization_authorized": False,
        "task_training_authorized": False,
    }
    protocol = build_protocol(input_hashes, findings)

    DEFAULT_OUTPUT.mkdir(parents=True)
    write_json_exclusive(DEFAULT_OUTPUT / "request.json", request)
    write_json_exclusive(DEFAULT_OUTPUT / "tokenwise_forensics.json", findings)
    write_json_exclusive(DEFAULT_OUTPUT / "pcar_protocol.json", protocol)
    with (DEFAULT_OUTPUT / "per_center_affine_uniform_convex.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary_center_rows[0]))
        writer.writeheader()
        writer.writerows(primary_center_rows)
    with (DEFAULT_OUTPUT / "per_token_affine_uniform_convex.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary_token_rows[0]))
        writer.writeheader()
        writer.writerows(primary_token_rows)
    with (DEFAULT_OUTPUT / "per_center_legacy_P2_P3.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(derived_centers[0]))
        writer.writeheader()
        writer.writerows(derived_centers)
    with (DEFAULT_OUTPUT / "per_token_legacy_P2_P3.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(token_rows[0]))
        writer.writeheader()
        writer.writerows(token_rows)
    snapshot = DEFAULT_OUTPUT / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(Path(__file__).resolve(), snapshot / Path(__file__).name)

    artifacts = {}
    for path in sorted(item for item in DEFAULT_OUTPUT.rglob("*") if item.is_file()):
        artifacts[str(path.relative_to(DEFAULT_OUTPUT)).replace("\\", "/")] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "status": "PCAR_DESIGN_EVIDENCE_COMPLETE_NO_TRAINING",
        "artifacts": artifacts,
        "input_bundle_sha256": hashlib.sha256(
            json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "output_directory": str(DEFAULT_OUTPUT.relative_to(ROOT)),
        "task_training_authorized": False,
        "pcar_training_authorized": False,
        "retained_ARFTR_changed": False,
        "zero_access_counters": findings["zero_access_counters"],
    }
    write_json_exclusive(DEFAULT_OUTPUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
