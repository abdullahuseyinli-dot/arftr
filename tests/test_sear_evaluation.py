"""Evaluation conventions must not confuse batch balance, collapse or rescues."""

import json

import numpy as np
import pytest

from hac.actor_memory_base import probability_metrics
from hac.sear_evaluation import classification_metrics, reference_strata_summary, slot_diagnostics


def test_balanced_batch_can_still_collapse_each_image_to_one_slot():
    mass = np.eye(3)[None]
    report = slot_diagnostics(mass, seed_ids=[42])["per_seed"]["42"]
    assert report["within_row"]["entropy_nats"]["mean"] == 0
    assert report["within_row"]["inverse_concentration_effective_slots"]["mean"] == 1
    assert report["batch_utilization"]["entropy_effective_slots"] == pytest.approx(3)
    assert report["positive_rows_with_one_active_slot"] == 3


def test_uniform_rows_are_distinct_from_single_slot_and_null_collapse():
    mass = np.array([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    report = slot_diagnostics(mass)["per_seed"]["0"]
    assert report["positive_evidence_rows"] == 2
    assert report["zero_or_negligible_total_mass_rows"] == 1
    assert report["within_row"]["entropy_effective_slots"]["mean"] == pytest.approx(2)
    assert report["within_row"]["herfindahl_concentration"]["mean"] == pytest.approx(2 / 3)
    assert report["active_slot_count"]["mean"] == pytest.approx(4 / 3)


def test_descriptor_and_position_redundancy_are_separate():
    masses = np.full((1, 1, 3), 1 / 3)
    descriptors = np.array([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]])
    positions = np.array([[[[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]]]])
    report = slot_diagnostics(masses, seed_positions=positions, seed_descriptors=descriptors)[
        "per_seed"
    ]["0"]
    assert report["descriptor_distinction"]["near_duplicate_pairs"] == 1
    assert report["descriptor_distinction"]["pairs"]["mean"] == pytest.approx(1 / 3)
    assert report["position_distinction"]["nearby_pairs"] == 1
    assert report["joint_descriptor_position_redundancy"]["near_duplicate_pairs"] == 1
    positions[0, 0, 1] = [2, 0]
    changed = slot_diagnostics(masses, seed_positions=positions, seed_descriptors=descriptors)[
        "per_seed"
    ]["0"]
    assert changed["descriptor_distinction"]["near_duplicate_pairs"] == 1
    assert changed["joint_descriptor_position_redundancy"]["near_duplicate_pairs"] == 0


def test_zero_norm_descriptor_is_undefined_not_distinct_evidence():
    report = slot_diagnostics(
        np.ones((1, 1, 2)), seed_descriptors=np.array([[[[0.0, 0.0], [1.0, 0.0]]]])
    )["per_seed"]["0"]
    assert report["descriptor_distinction"]["active_zero_norm_slots"] == 1
    assert report["descriptor_distinction"]["pairs"]["count"] == 0
    assert report["descriptor_distinction"]["mass_product_weighted_mean"] is None


def test_missing_rows_and_zero_mass_slots_can_nan_without_contamination():
    masses = np.array([[[1.0, 0.0], [np.nan, np.nan], [0.0, 0.0]]])
    descriptors = np.full((1, 3, 2, 4), np.nan)
    positions = np.full((1, 3, 2, 2), np.nan)
    descriptors[0, 0, 0] = 1
    positions[0, 0, 0] = 0
    before = masses.copy()
    result = slot_diagnostics(
        masses,
        seed_descriptors=descriptors,
        seed_positions=positions,
        local_valid=np.array([True, False, True]),
    )
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(masses, before)
    row = result["per_seed"]["0"]
    assert row["masked_local_rows"] == 1 and row["valid_local_rows"] == 2
    assert row["zero_or_negligible_total_mass_rows"] == 1


def test_empty_rows_single_slot_and_no_valid_local_rows_are_json_safe():
    for masses, valid in (
        (np.zeros((2, 0, 1)), None),
        (np.full((2, 3, 1), np.nan), np.zeros(3, bool)),
    ):
        result = slot_diagnostics(masses, local_valid=valid)
        json.dumps(result, allow_nan=False)
        assert result["per_seed"]["0"]["batch_utilization"]["entropy_effective_slots"] is None
    result = slot_diagnostics(np.ones((1, 2, 1)), seed_positions=np.zeros((1, 2, 1, 2)))
    assert result["per_seed"]["0"]["position_distinction"]["pairs"]["count"] == 0


def test_slot_permutation_preserves_scalar_diagnostics_without_seed_averaging():
    masses = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    result = slot_diagnostics(masses, seed_ids=[42, 43])
    left, right = result["per_seed"]["42"], result["per_seed"]["43"]
    assert left["within_row"] == right["within_row"]
    assert left["batch_utilization"]["aggregate_mass_share_by_slot"] == [1, 0, 0]
    assert right["batch_utilization"]["aggregate_mass_share_by_slot"] == [0, 0, 1]
    assert left["batch_utilization"]["entropy_effective_slots"] == 1
    assert right["batch_utilization"]["entropy_effective_slots"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_masses": np.ones((2, 3))},
        {"seed_masses": -np.ones((1, 2, 3))},
        {"seed_masses": np.ones((1, 2, 3)), "local_valid": np.ones(2, int)},
        {"seed_masses": np.ones((2, 2, 3)), "seed_ids": [42, 42]},
        {"seed_masses": np.ones((1, 2, 3)), "seed_positions": np.ones((1, 2, 3, 3))},
        {"seed_masses": np.ones((1, 2, 3)), "seed_descriptors": np.full((1, 2, 3, 4), np.nan)},
        {"seed_masses": np.ones((1, 2, 3)), "active_share_threshold": 0},
    ],
)
def test_invalid_slot_contracts_refused(kwargs):
    with pytest.raises(ValueError):
        slot_diagnostics(**kwargs)


def test_metrics_match_existing_three_class_definition_exactly():
    rng = np.random.default_rng(8)
    p = rng.dirichlet([1, 1, 1], size=60)
    labels = np.arange(60) % 3
    assert classification_metrics(labels, p) == probability_metrics(labels, p)


def test_empty_metrics_undefined_and_binary_class_order_supported():
    result = classification_metrics(np.array([], int), np.empty((0, 3)))
    assert result["macro_f1"] is None and result["per_class_f1"] == [None] * 3
    assert classification_metrics(np.array([0, 1]), np.eye(2))["macro_f1"] == 1


def test_metric_class_index_arithmetic_does_not_overflow_small_integer_labels():
    labels = np.arange(100, dtype=np.int8)
    assert classification_metrics(labels, np.eye(100))["accuracy"] == 1
    with pytest.raises(ValueError, match="simplex"):
        classification_metrics(np.array([0, 1]), np.eye(2, dtype=np.complex128))


def test_named_comparison_distinguishes_wrong_to_wrong_flips_and_shared_repairs():
    labels = np.array([0, 1, 2, 0, 1, 2])
    candidate = np.eye(3)[[0, 2, 1, 0, 1, 0]]
    old = np.eye(3)[[1, 1, 0, 0, 1, 0]]
    already_repaired = candidate.copy()
    shared = np.array([True, False, True, True, False, False])
    result = reference_strata_summary(
        labels,
        candidate,
        {"old_m4": old, "other": already_repaired},
        {"all": np.ones(6, bool), "empty": np.zeros(6, bool)},
        shared,
        candidate_name="sear",
    )
    row = result["references"]["old_m4"]["strata"]["all"]
    assert row["rescues"] == 1 and row["harms"] == 1
    assert row["correctness_flips"] == 2 and row["prediction_flips"] == 3
    assert row["wrong_to_different_wrong"] == 1 and row["both_wrong"] == 2
    assert row["original_shared_rows"] == 3 and row["original_shared_repairs"] == 2
    assert row["original_shared_rescues_vs_reference"] == 1
    other = result["references"]["other"]["strata"]["all"]
    assert (
        other["original_shared_repairs"] == 2 and other["original_shared_rescues_vs_reference"] == 0
    )
    assert "p6_metrics" not in row
    assert (
        result["references"]["old_m4"]["strata"]["empty"]["candidate_metrics"]["accuracy"] is None
    )
    json.dumps(result, allow_nan=False)


def test_strata_may_overlap_and_inputs_are_not_modified():
    labels = np.array([0, 1, 2])
    p, reference = np.eye(3), np.eye(3)[[1, 0, 2]]
    original = p.copy()
    result = reference_strata_summary(
        labels,
        p,
        {"baseline": reference},
        {"first": np.array([True, True, False]), "second": np.array([True, False, True])},
        np.ones(3, bool),
    )
    assert result["references"]["baseline"]["strata"]["first"]["rescues"] == 2
    assert result["references"]["baseline"]["strata"]["second"]["rescues"] == 1
    np.testing.assert_array_equal(p, original)


@pytest.mark.parametrize("mutation", ["mask", "shared", "negative", "nan", "shape", "labels"])
def test_comparison_alignment_and_simplex_failures_are_refused(mutation):
    labels = np.array([0, 1, 2])
    p = np.eye(3)
    references = {"baseline": p.copy()}
    masks, shared = {"all": np.ones(3, bool)}, np.ones(3, bool)
    if mutation == "mask":
        masks["all"] = np.ones(3, int)
    elif mutation == "shared":
        shared = np.ones(2, bool)
    elif mutation == "negative":
        references["baseline"][0] = [-1, 1, 1]
    elif mutation == "nan":
        references["baseline"][0, 0] = np.nan
    elif mutation == "shape":
        references["baseline"] = np.ones((3, 2)) / 2
    else:
        labels[2] = 3
    with pytest.raises(ValueError):
        reference_strata_summary(labels, p, references, masks, shared)
