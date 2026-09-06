from __future__ import annotations

import json
from io import BytesIO
from itertools import product
from pathlib import Path

import analyze_okutama_cptr_baseline_preservation as analyzer
import numpy as np
import pandas as pd
import pytest
from analyze_okutama_cptr_baseline_preservation import (
    ATTESTED_DEVELOPMENT_MANIFEST_SHA256,
    CLASS_NAMES,
    EXPECTED_SOURCE_SHA256,
    STATUS,
    ArraySpec,
    InputPaths,
    ScopeData,
    align_scope,
    build_paired_rows,
    build_slice_metrics,
    exact_recording_swap,
    fold_level_sensitivity,
    model_metrics,
    oracle_occlusion_fallback,
    pair_outcomes,
    recording_cluster_bootstrap,
    recording_resample_indices,
    recording_resample_plan_sha256,
    run_analysis,
    sha256_file,
    static_direction_diagnostics,
    validate_development_summary,
    validate_npz_arrays,
    validate_probabilities,
    verify_source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]


def probabilities(predictions: list[int], confidence: float = 0.8) -> np.ndarray:
    remainder = (1.0 - confidence) / (len(CLASS_NAMES) - 1)
    values = np.full((len(predictions), len(CLASS_NAMES)), remainder, dtype=np.float64)
    values[np.arange(len(predictions)), predictions] = confidence
    return values


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def independent_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for class_index in range(len(CLASS_NAMES)):
        true_positive = int(np.sum((labels == class_index) & (predictions == class_index)))
        denominator = int(np.sum(labels == class_index) + np.sum(predictions == class_index))
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(sum(scores) / len(scores))


def independent_nll(labels: np.ndarray, values: np.ndarray) -> float:
    return float(np.mean(-np.log(np.clip(values[np.arange(len(labels)), labels], 1e-12, 1.0))))


def synthetic_scope() -> ScopeData:
    labels = np.asarray([0, 1, 2, 0], dtype=np.int64)
    baseline = probabilities([0, 2, 2, 1])
    candidate = probabilities([0, 1, 1, 2])
    static = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    return ScopeData(
        scope="grouped_crossfit_oof",
        development_role="train",
        sample_ids=np.asarray(["s0", "s1", "s2", "s3"]),
        recording_ids=np.asarray(["g0", "g0", "g1", "g1"]),
        track_ids=np.asarray(["t0", "t1", "t2", "t3"]),
        folds=np.asarray(["fold-0", "fold-0", "fold-1", "fold-1"]),
        labels=labels,
        transition=np.asarray([False, False, True, True]),
        occluded=np.asarray([False, True, True, False]),
        baseline=baseline,
        candidate=candidate,
        static=static,
    )


def synthetic_contract() -> dict:
    return {
        "development_role": "train",
        "rows": 4,
        "recordings": ("g0", "g1"),
        "tracks": 4,
        "label_counts": {0: 2, 1: 1, 2: 1},
        "transition_rows": 2,
        "occluded_rows": 2,
    }


def synthetic_prediction_and_anchor() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = synthetic_scope()
    prediction = {
        "sample_ids": data.sample_ids.copy(),
        "recording_ids": data.recording_ids.copy(),
        "track_ids": data.track_ids.copy(),
        "labels": data.labels.copy(),
        "transition_targets": data.transition.copy(),
        "occlusion_targets": data.occluded.copy(),
        "probabilities": data.candidate.copy(),
        "baseline_probabilities": data.baseline.copy(),
    }
    anchor = {
        "sample_ids": data.sample_ids.copy(),
        "recording_ids": data.recording_ids.copy(),
        "track_ids": data.track_ids.copy(),
        "labels": data.labels.copy(),
        "static_probabilities": data.static.copy(),
        "teacher_probabilities": data.baseline.copy(),
    }
    return prediction, anchor


def test_pair_outcomes_losses_and_oracle_fallback_are_row_paired() -> None:
    data = synthetic_scope()
    outcomes, correctness_changed = pair_outcomes(data.labels, data.baseline, data.candidate)

    assert outcomes.tolist() == ["both_correct", "rescued", "harmed", "both_wrong"]
    assert correctness_changed.tolist() == [False, True, True, False]

    fallback = oracle_occlusion_fallback(data)
    assert np.array_equal(fallback[data.occluded], data.baseline[data.occluded])
    assert np.array_equal(fallback[~data.occluded], data.candidate[~data.occluded])

    rows = build_paired_rows(data)
    assert rows.loc[1, "candidate_minus_baseline_nll"] < 0.0
    assert rows.loc[2, "candidate_minus_baseline_nll"] > 0.0
    assert rows.loc[1, "candidate_vs_baseline_outcome"] == "rescued"
    assert rows.loc[2, "candidate_vs_baseline_outcome"] == "harmed"
    assert rows.loc[1, "oracle_occlusion_fallback_prediction_index"] == 2
    assert rows.loc[3, "oracle_occlusion_fallback_prediction_index"] == 2


def test_static_direction_projection_and_degenerate_rows() -> None:
    baseline = np.asarray([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.5, 0.3, 0.2]])
    static = np.asarray([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    candidate = np.asarray([[0.4, 0.5, 0.1], [0.6, 0.3, 0.1], [0.5, 0.3, 0.2]])

    result = static_direction_diagnostics(baseline, candidate, static)

    assert result["projection"][0] == pytest.approx(0.5)
    assert result["cosine"][0] == pytest.approx(1.0)
    assert bool(result["toward_static"][0])
    assert bool(result["closer_to_static"][0])
    assert not bool(result["defined"][1])
    assert np.isnan(result["projection"][1])
    assert result["projection"][2] == pytest.approx(0.0)
    assert np.isnan(result["cosine"][2])
    assert not bool(result["toward_static"][2])


def test_cluster_bootstrap_is_deterministic_and_keeps_recording_pairs() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 3)
    groups = np.repeat(np.asarray(["a", "b", "c"]), 3)
    candidate = probabilities(labels.tolist(), confidence=0.85)
    baseline = probabilities([1, 1, 2, 0, 2, 2, 0, 1, 0], confidence=0.70)
    recording_order, sampled = recording_resample_indices(groups, resamples=200, seed=17)

    first = recording_cluster_bootstrap(
        labels,
        candidate,
        baseline,
        groups,
        recording_order=recording_order,
        sampled_recording_indices=sampled,
    )
    second = recording_cluster_bootstrap(
        labels,
        candidate,
        baseline,
        groups,
        recording_order=recording_order,
        sampled_recording_indices=sampled,
    )

    assert first == second
    assert first["clusters"] == 3
    assert first["resamples"] == 200
    assert first["delta_definition"] == "candidate_minus_baseline"
    assert "two_sided_p" not in first["macro_f1"]
    assert first["resample_plan_sha256"] == recording_resample_plan_sha256(recording_order, sampled)
    assert first["macro_f1"]["point_estimate"] > 0.0
    assert first["negative_log_likelihood"]["point_estimate"] < 0.0
    assert set(first["recording_estimates"]) == {"a", "b", "c"}


def test_exact_recording_swap_enumerates_every_assignment() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 3)
    groups = np.repeat(np.asarray(["a", "b", "c"]), 3)
    candidate = probabilities(labels.tolist(), confidence=0.85)
    baseline = probabilities([1, 1, 2, 0, 2, 2, 0, 1, 0], confidence=0.70)

    result = exact_recording_swap(labels, candidate, baseline, groups)

    assert result["available"] is True
    assert result["recordings"] == 3
    assert result["permutations"] == 8
    assert result["delta_definition"] == "candidate_minus_baseline"
    assert result["macro_f1"]["point_estimate"] == pytest.approx(
        model_metrics(labels, candidate)["macro_f1"] - model_metrics(labels, baseline)["macro_f1"]
    )
    assert result["macro_f1"]["null_min"] == pytest.approx(-result["macro_f1"]["null_max"])
    assert 0.0 < result["macro_f1"]["two_sided_p"] <= 1.0


def test_tiny_exact_swap_and_resample_match_independent_brute_force() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 2)
    groups = np.repeat(np.asarray(["a", "b"]), 3)
    candidate = probabilities([0, 1, 2, 0, 2, 2], confidence=0.8)
    baseline = probabilities([0, 2, 2, 1, 1, 2], confidence=0.7)

    point_macro = independent_macro_f1(labels, candidate.argmax(axis=1)) - independent_macro_f1(
        labels, baseline.argmax(axis=1)
    )
    point_nll = independent_nll(labels, candidate) - independent_nll(labels, baseline)
    exact_macro_values = []
    exact_nll_values = []
    for assignment in product((False, True), repeat=2):
        choose_candidate = np.asarray([assignment[0]] * 3 + [assignment[1]] * 3)
        permuted_candidate = np.where(choose_candidate[:, None], candidate, baseline)
        permuted_baseline = np.where(choose_candidate[:, None], baseline, candidate)
        exact_macro_values.append(
            independent_macro_f1(labels, permuted_candidate.argmax(axis=1))
            - independent_macro_f1(labels, permuted_baseline.argmax(axis=1))
        )
        exact_nll_values.append(
            independent_nll(labels, permuted_candidate) - independent_nll(labels, permuted_baseline)
        )
    exact_macro = np.asarray(exact_macro_values)
    exact_nll = np.asarray(exact_nll_values)
    exact = exact_recording_swap(labels, candidate, baseline, groups)
    assert exact["macro_f1"]["point_estimate"] == pytest.approx(point_macro)
    assert exact["macro_f1"]["null_min"] == pytest.approx(exact_macro.min())
    assert exact["macro_f1"]["null_max"] == pytest.approx(exact_macro.max())
    assert exact["macro_f1"]["two_sided_p"] == pytest.approx(
        np.mean(np.abs(exact_macro) >= abs(point_macro) - 1e-15)
    )
    assert exact["negative_log_likelihood"]["point_estimate"] == pytest.approx(point_nll)
    assert exact["negative_log_likelihood"]["null_min"] == pytest.approx(exact_nll.min())
    assert exact["negative_log_likelihood"]["null_max"] == pytest.approx(exact_nll.max())
    assert exact["negative_log_likelihood"]["two_sided_p"] == pytest.approx(
        np.mean(np.abs(exact_nll) >= abs(point_nll) - 1e-15)
    )

    recording_order = np.asarray(["a", "b"])
    sampled = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    bootstrap_macro = []
    bootstrap_nll = []
    for draw in sampled:
        row_indices = np.concatenate(
            [np.flatnonzero(groups == recording_order[index]) for index in draw]
        )
        draw_labels = labels[row_indices]
        draw_candidate = candidate[row_indices]
        draw_baseline = baseline[row_indices]
        bootstrap_macro.append(
            independent_macro_f1(draw_labels, draw_candidate.argmax(axis=1))
            - independent_macro_f1(draw_labels, draw_baseline.argmax(axis=1))
        )
        bootstrap_nll.append(
            independent_nll(draw_labels, draw_candidate)
            - independent_nll(draw_labels, draw_baseline)
        )
    bootstrap = recording_cluster_bootstrap(
        labels,
        candidate,
        baseline,
        groups,
        recording_order=recording_order,
        sampled_recording_indices=sampled,
    )
    for metric, expected in (
        ("macro_f1", np.asarray(bootstrap_macro)),
        ("negative_log_likelihood", np.asarray(bootstrap_nll)),
    ):
        assert bootstrap[metric]["valid_resamples"] == len(sampled)
        assert bootstrap[metric]["resample_min"] == pytest.approx(expected.min())
        assert bootstrap[metric]["resample_max"] == pytest.approx(expected.max())
        assert bootstrap[metric]["ci_95_low"] == pytest.approx(np.quantile(expected, 0.025))
        assert bootstrap[metric]["ci_95_high"] == pytest.approx(np.quantile(expected, 0.975))


@pytest.mark.parametrize(
    ("arrays", "schema", "message"),
    [
        ({"x": np.zeros(2, dtype=np.int64)}, {"x": ArraySpec((3,), "int64")}, "shape"),
        ({"x": np.zeros(2, dtype=np.float32)}, {"x": ArraySpec((2,), "float64")}, "dtype"),
        (
            {"x": np.zeros(2, dtype=np.int64), "extra": np.zeros(1)},
            {"x": ArraySpec((2,), "int64")},
            "keys",
        ),
    ],
)
def test_npz_schema_validation_rejects_shape_dtype_and_key_drift(
    arrays: dict[str, np.ndarray], schema: dict[str, ArraySpec], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_npz_arrays(arrays, schema, name="fixture")


def test_npz_loader_checks_exact_members_before_materializing_arrays() -> None:
    schema = {
        "ids": ArraySpec((2,), "int64"),
        "scores": ArraySpec((2,), "float64"),
    }
    arrays = {
        "ids": np.asarray([7, 9], dtype=np.int64),
        "scores": np.asarray([0.25, 0.75], dtype=np.float64),
    }
    loaded = analyzer.load_npz(npz_bytes(arrays), schema, name="tiny")
    assert set(loaded) == set(schema)
    assert np.array_equal(loaded["ids"], arrays["ids"])
    assert np.array_equal(loaded["scores"], arrays["scores"])

    # The unexpected object member would require pickle to materialize.  Exact archive-member
    # validation must reject it before NumPy attempts to load any array.
    with_extra_object = {**arrays, "unexpected": np.asarray([object()], dtype=object)}
    with pytest.raises(RuntimeError, match="NPZ member names changed"):
        analyzer.load_npz(npz_bytes(with_extra_object), schema, name="tiny")


def test_resample_npz_is_deterministic_with_exact_keys_and_dtypes(tmp_path: Path) -> None:
    arrays = {
        "grouped_crossfit_oof__recording_order": np.asarray(["a", "b"], dtype="<U1"),
        "grouped_crossfit_oof__sampled_recording_indices": np.asarray(
            [[0, 1], [1, 1]], dtype=np.int64
        ),
        "fixed_development_validation__recording_order": np.asarray(
            ["v0", "v1", "v2"], dtype="<U2"
        ),
        "fixed_development_validation__sampled_recording_indices": np.asarray(
            [[0, 2, 1], [2, 2, 0]], dtype=np.int64
        ),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    analyzer._atomic_npz(first, arrays)
    analyzer._atomic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as payload:
        assert set(payload.files) == set(arrays)
        for key, expected in arrays.items():
            assert payload[key].dtype == expected.dtype
            assert np.array_equal(payload[key], expected)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[np.nan, 0.5, 0.5]]),
        np.asarray([[-0.1, 0.5, 0.6]]),
        np.asarray([[0.2, 0.3, 0.4]]),
        np.asarray([[0.5, 0.5]]),
    ],
)
def test_probability_validation_rejects_invalid_matrices(values: np.ndarray) -> None:
    with pytest.raises(RuntimeError):
        validate_probabilities(values, name="fixture")


def test_sample_id_join_rejects_anchor_drift() -> None:
    prediction, anchor = synthetic_prediction_and_anchor()
    anchor["sample_ids"][0] = "not-s0"

    with pytest.raises(RuntimeError, match="anchor/prediction ID sets differ"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=prediction,
            anchor=anchor,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )


def test_shuffled_anchor_joins_by_id_but_metadata_mismatch_is_rejected() -> None:
    prediction, anchor = synthetic_prediction_and_anchor()
    order = np.asarray([2, 0, 3, 1])
    shuffled = {name: values[order] for name, values in anchor.items()}
    aligned = align_scope(
        scope="grouped_crossfit_oof",
        prediction=prediction,
        anchor=shuffled,
        expectation=synthetic_contract(),
        fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
    )
    assert np.array_equal(aligned.sample_ids, prediction["sample_ids"])
    assert np.array_equal(aligned.static, anchor["static_probabilities"])

    mismatched = {name: values.copy() for name, values in anchor.items()}
    mismatched["recording_ids"] = np.roll(mismatched["recording_ids"], 1)
    with pytest.raises(RuntimeError, match="recording IDs disagree"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=prediction,
            anchor=mismatched,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )

    ids_only_shuffled = {name: values.copy() for name, values in prediction.items()}
    ids_only_shuffled["sample_ids"] = ids_only_shuffled["sample_ids"][[1, 0, 2, 3]]
    with pytest.raises(RuntimeError, match="anchor labels disagree"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=ids_only_shuffled,
            anchor=anchor,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )


def test_prediction_scope_contract_rejects_group_and_count_drift() -> None:
    prediction, anchor = synthetic_prediction_and_anchor()
    crossing = {name: values.copy() for name, values in prediction.items()}
    crossing["track_ids"][2] = crossing["track_ids"][0]
    relaxed_track_count = {**synthetic_contract(), "tracks": 3}
    with pytest.raises(RuntimeError, match="cross recording boundaries"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=crossing,
            anchor=anchor,
            expectation=relaxed_track_count,
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )

    wrong_recording = {name: values.copy() for name, values in prediction.items()}
    wrong_recording["recording_ids"][0] = "unexpected"
    with pytest.raises(RuntimeError, match="recording IDs changed"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=wrong_recording,
            anchor=anchor,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )

    wrong_transition = {name: values.copy() for name, values in prediction.items()}
    wrong_transition["transition_targets"][0] = True
    with pytest.raises(RuntimeError, match="transition count changed"):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=wrong_transition,
            anchor=anchor,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sample_ids", "   ", "sample IDs contain blanks"),
        ("recording_ids", "\t", "blank recording IDs"),
        ("track_ids", "", "blank track IDs"),
    ],
)
def test_prediction_scope_rejects_blank_identity_values(
    field: str, replacement: str, message: str
) -> None:
    prediction, anchor = synthetic_prediction_and_anchor()
    prediction[field][0] = replacement
    with pytest.raises(RuntimeError, match=message):
        align_scope(
            scope="grouped_crossfit_oof",
            prediction=prediction,
            anchor=anchor,
            expectation=synthetic_contract(),
            fold_by_recording={"g0": "fold-0", "g1": "fold-1"},
        )


def test_probability_sum_tolerance_is_one_part_per_million() -> None:
    validate_probabilities(
        np.asarray([[0.2, 0.3, 0.5000005]], dtype=float), name="within_tolerance"
    )
    with pytest.raises(RuntimeError, match="sum to one"):
        validate_probabilities(
            np.asarray([[0.2, 0.3, 0.5000011]], dtype=float), name="outside_tolerance"
        )


def test_probability_bound_tolerance_is_one_part_per_million() -> None:
    validate_probabilities(
        np.asarray([[1.0000005, -0.0000005, 0.0]], dtype=float),
        name="within_bound_tolerance",
    )
    for values in (
        np.asarray([[1.0000011, -0.0000011, 0.0]], dtype=float),
        np.asarray([[-0.0000011, 0.50000055, 0.50000055]], dtype=float),
    ):
        with pytest.raises(RuntimeError, match="tolerated"):
            validate_probabilities(values, name="outside_bound_tolerance")


def test_true_class_slices_use_recall_and_confusion_not_f1() -> None:
    slices = build_slice_metrics(synthetic_scope())
    sitting = slices.loc[
        slices["slice_type"].eq("class") & slices["slice_value"].eq("sitting")
    ].iloc[0]
    assert sitting["class_metric_status"] == "true_class_recall_and_confusion_only"
    assert sitting["baseline_true_class_recall"] == pytest.approx(0.5)
    assert pd.isna(sitting["baseline_macro_f1"])
    baseline_counts = [sitting[f"baseline_predicted_{name}_count"] for name in CLASS_NAMES]
    baseline_fractions = [sitting[f"baseline_predicted_{name}_fraction"] for name in CLASS_NAMES]
    assert sum(baseline_counts) == sitting["rows"]
    assert sum(baseline_fractions) == pytest.approx(1.0)
    overall = slices.loc[slices["slice_type"].eq("overall")].iloc[0]
    assert overall["class_metric_status"] == "all_three_true_classes_supported"
    assert not pd.isna(overall["baseline_macro_f1"])


def test_shared_resample_plan_is_seed_sensitive_and_zero_under_identity() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 3)
    groups = np.repeat(np.asarray(["a", "b", "c"]), 3)
    candidate = probabilities(labels.tolist(), confidence=0.85)
    order, sampled = recording_resample_indices(groups, resamples=250, seed=17)
    order_again, sampled_again = recording_resample_indices(groups, resamples=250, seed=17)
    other_order, other_sampled = recording_resample_indices(groups, resamples=250, seed=18)
    assert np.array_equal(order, order_again)
    assert np.array_equal(sampled, sampled_again)
    assert np.array_equal(order, other_order)
    assert not np.array_equal(sampled, other_sampled)
    assert recording_resample_plan_sha256(order, sampled) != recording_resample_plan_sha256(
        other_order, other_sampled
    )

    identity = recording_cluster_bootstrap(
        labels,
        candidate,
        candidate,
        groups,
        recording_order=order,
        sampled_recording_indices=sampled,
    )
    for metric in ("macro_f1", "negative_log_likelihood", "brier_score"):
        assert identity[metric]["point_estimate"] == pytest.approx(0.0)
        assert identity[metric]["resample_min"] == pytest.approx(0.0)
        assert identity[metric]["resample_max"] == pytest.approx(0.0)
        assert "two_sided_p" not in identity[metric]


def test_occluded_subgroup_reports_supported_resamples_and_exact_swap() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 3)
    groups = np.repeat(np.asarray(["a", "b", "c"]), 3)
    candidate = probabilities(labels.tolist(), confidence=0.85)
    baseline = probabilities([1, 1, 2, 0, 2, 2, 0, 1, 0], confidence=0.70)
    occluded = groups == "a"
    order, sampled = recording_resample_indices(groups, resamples=500, seed=29)
    result = recording_cluster_bootstrap(
        labels,
        candidate,
        baseline,
        groups,
        recording_order=order,
        sampled_recording_indices=sampled,
        subgroup_mask=occluded,
        estimand="window_any_occluded_true_rows",
    )
    assert 0 < result["macro_f1"]["valid_resamples"] < 500
    assert result["macro_f1"]["unsupported_resamples"] > 0
    assert (
        result["negative_log_likelihood"]["valid_resamples"]
        == result["macro_f1"]["valid_resamples"]
    )
    exact = exact_recording_swap(
        labels,
        candidate,
        baseline,
        groups,
        subgroup_mask=occluded,
        estimand="window_any_occluded_true_rows",
    )
    assert exact["available"] is True
    assert exact["rows"] == 3
    assert exact["recordings"] == 1
    assert exact["permutations"] == 2
    assert "two_sided_p" in exact["macro_f1"]

    oracle = np.where(occluded[:, None], baseline, candidate)
    oracle_zero = recording_cluster_bootstrap(
        labels,
        oracle,
        baseline,
        groups,
        recording_order=order,
        sampled_recording_indices=sampled,
        subgroup_mask=occluded,
        estimand="window_any_occluded_true_rows",
    )
    assert oracle_zero["macro_f1"]["resample_min"] == pytest.approx(0.0)
    assert oracle_zero["macro_f1"]["resample_max"] == pytest.approx(0.0)


def test_fold_sensitivity_exposes_conditional_sign_reversal() -> None:
    labels = np.tile(np.arange(3, dtype=np.int64), 4)
    baseline_predictions = [1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2]
    candidate_predictions = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0]
    data = ScopeData(
        scope="grouped_crossfit_oof",
        development_role="train",
        sample_ids=np.asarray([f"s{index}" for index in range(12)]),
        recording_ids=np.repeat(np.asarray(["g0", "g1"]), 6),
        track_ids=np.asarray([f"t{index}" for index in range(12)]),
        folds=np.repeat(np.asarray(["fold-0", "fold-1"]), 6),
        labels=labels,
        transition=np.zeros(12, dtype=bool),
        occluded=np.zeros(12, dtype=bool),
        baseline=probabilities(baseline_predictions),
        candidate=probabilities(candidate_predictions),
        static=probabilities(labels.tolist()),
    )
    result = fold_level_sensitivity(data, fold_recordings={"fold-0": ("g0",), "fold-1": ("g1",)})
    assert result["overall_delta"] > 0.0
    assert result["folds"]["fold-1"]["held_out_macro_f1_delta"] < 0.0
    assert result["held_out_fold_sign_reversal_present"] is True
    assert "fixed five-fold" in result["conditional_inference_limit"]


def test_fold_sensitivity_declares_fixed_label_zero_division_semantics() -> None:
    data = synthetic_scope()
    result = fold_level_sensitivity(data, fold_recordings={"fold-0": ("g0",), "fold-1": ("g1",)})
    semantics = result["macro_f1_semantics"]
    assert semantics["class_order"] == list(CLASS_NAMES)
    assert semantics["averaging"] == "unweighted_mean_over_all_three_fixed_labels"
    assert semantics["zero_division"] == 0.0
    assert "Inspect all_true_classes_supported" in semantics["missing_true_class_policy"]
    assert result["folds"]["fold-0"]["all_true_classes_supported"] is False
    assert result["folds"]["fold-0"]["held_out_macro_f1_delta"] == pytest.approx(1.0 / 3.0)


def test_summary_manifest_attestation_does_not_require_manifest_input(tmp_path: Path) -> None:
    summary = {
        "status": "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION",
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "folds": 5,
        "seeds": [42, 43, 44, 45, 46],
        "grouped_crossfit_oof": {"samples": 4977, "recordings": 11},
        "development_validation": {"samples": 1383, "recordings": 3},
        "artifact_sha256": {
            "crossfit_oof_predictions.npz": EXPECTED_SOURCE_SHA256["crossfit_oof_predictions"],
            "validation_predictions.npz": EXPECTED_SOURCE_SHA256["validation_predictions"],
        },
        "source_sha256": {
            "manifest": "0" * 64,
            "baseline_predictions": EXPECTED_SOURCE_SHA256["validation_anchor"],
        },
    }
    with pytest.raises(RuntimeError, match="manifest lock mismatch"):
        validate_development_summary(json.dumps(summary).encode("utf-8"))
    assert ATTESTED_DEVELOPMENT_MANIFEST_SHA256 not in EXPECTED_SOURCE_SHA256.values()
    assert "development_manifest" not in InputPaths.__dataclass_fields__


def test_source_hash_drift_fails_before_parsing(monkeypatch) -> None:
    captured = {name: f"locked-{name}".encode("ascii") for name in EXPECTED_SOURCE_SHA256}
    for name, content in captured.items():
        monkeypatch.setitem(analyzer.EXPECTED_SOURCE_SHA256, name, analyzer.sha256_bytes(content))
    captured["development_summary"] += b"x"
    with pytest.raises(RuntimeError, match="Locked SHA-256 mismatch for development_summary"):
        verify_source_hashes(captured)


def test_locked_loader_reads_only_allowlist_once_and_uses_captured_bytes_after_swap(
    tmp_path: Path, monkeypatch
) -> None:
    paths = InputPaths(
        development_summary=tmp_path / "summary.json",
        crossfit_oof_predictions=tmp_path / "oof.npz",
        validation_predictions=tmp_path / "validation.npz",
        crossfit_oof_anchor=tmp_path / "oof-anchor.npz",
        validation_anchor=tmp_path / "validation-anchor.npz",
    )
    source_paths = paths.as_dict()
    original_content = {name: f"captured::{name}".encode("ascii") for name in source_paths}
    replacements: dict[str, Path] = {}
    for name, path in source_paths.items():
        path.write_bytes(original_content[name])
        replacement = tmp_path / f"replacement-{name}"
        replacement.write_bytes(f"swapped::{name}".encode("ascii"))
        replacements[name] = replacement
        monkeypatch.setitem(
            analyzer.EXPECTED_SOURCE_SHA256,
            name,
            analyzer.sha256_bytes(original_content[name]),
        )

    allowed = {path.resolve() for path in source_paths.values()}
    read_calls: list[Path] = []
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved not in allowed:
            raise AssertionError(f"undeclared source read: {resolved}")
        read_calls.append(resolved)
        return original_read_bytes(path)

    observed_parse_bytes: dict[str, bytes] = {}

    def fake_validate_summary(content: bytes) -> dict[str, str]:
        observed_parse_bytes["development_summary"] = content
        # Swap every path after capture and hash validation but before any NPZ parser runs.
        for name, path in source_paths.items():
            replacements[name].replace(path)
        return {"status": "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION"}

    def fake_load_npz(
        content: bytes, schema: dict[str, ArraySpec], *, name: str
    ) -> dict[str, np.ndarray]:
        del schema
        observed_parse_bytes[name] = content
        return {"placeholder": np.asarray([0])}

    def fake_align_scope(
        *, scope: str, prediction: dict[str, np.ndarray], anchor: dict[str, np.ndarray]
    ) -> ScopeData:
        del prediction, anchor
        base = synthetic_scope()
        if scope == "grouped_crossfit_oof":
            return base
        return ScopeData(
            scope="fixed_development_validation",
            development_role="validation",
            sample_ids=np.asarray([f"v-{value}" for value in base.sample_ids]),
            recording_ids=np.asarray([f"v-{value}" for value in base.recording_ids]),
            track_ids=np.asarray([f"v-{value}" for value in base.track_ids]),
            folds=np.full(len(base.labels), "not_applicable"),
            labels=base.labels.copy(),
            transition=base.transition.copy(),
            occluded=base.occluded.copy(),
            baseline=base.baseline.copy(),
            candidate=base.candidate.copy(),
            static=base.static.copy(),
        )

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(analyzer, "validate_development_summary", fake_validate_summary)
    monkeypatch.setattr(analyzer, "load_npz", fake_load_npz)
    monkeypatch.setattr(analyzer, "align_scope", fake_align_scope)

    _, audit = analyzer.load_locked_development(paths)

    assert len(read_calls) == len(source_paths) == 5
    assert set(read_calls) == allowed
    assert all(read_calls.count(path) == 1 for path in allowed)
    assert observed_parse_bytes == original_content
    assert audit["source_sha256"] == {
        name: analyzer.sha256_bytes(content) for name, content in original_content.items()
    }
    for name, path in source_paths.items():
        assert original_read_bytes(path) == f"swapped::{name}".encode("ascii")


def test_output_overlap_is_rejected_before_input_reads_or_writes(tmp_path: Path) -> None:
    input_parent = tmp_path / "locked-inputs"
    input_parent.mkdir()
    sentinel = input_parent / "summary.json"
    sentinel.write_bytes(b"immutable")
    paths = InputPaths(
        development_summary=sentinel,
        crossfit_oof_predictions=input_parent / "predictions.npz",
        validation_predictions=input_parent / "validation.npz",
        crossfit_oof_anchor=input_parent / "anchor.npz",
        validation_anchor=input_parent / "validation-anchor.npz",
    )
    with pytest.raises(RuntimeError, match="overlaps locked input"):
        run_analysis(paths, input_parent, bootstrap_resamples=2)
    assert sentinel.read_bytes() == b"immutable"


def test_real_locked_evidence_writes_complete_deterministic_diagnostic(tmp_path: Path) -> None:
    input_paths = InputPaths.from_root(ROOT)
    if not all(path.is_file() for path in input_paths.as_dict().values()):
        pytest.skip("retained CPTR development evidence unavailable")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_analysis(input_paths, first_dir, bootstrap_resamples=100, bootstrap_seed=91)
    second = run_analysis(input_paths, second_dir, bootstrap_resamples=100, bootstrap_seed=91)

    assert first == second
    assert first["status"] == STATUS
    assert first["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert first["analysis_access_accounting"] == {
        "protected_raw_rows_read_by_current_analysis": 0,
        "protected_raw_fields_read_by_current_analysis": 0,
        "protected_raw_arrays_read_by_current_analysis": 0,
        "features_or_images_read_by_analysis": 0,
        "checkpoints_read_by_analysis": 0,
    }
    assert first["external_operator_context"] == {
        "prior_interim_protected_raw_rows_scanned": 1979,
        "prior_interim_protected_rows_displayed": 5,
        "performed_by_current_analyzer": False,
        "used_by_current_analysis": False,
        "note": (
            "Before this refactored analysis, an interim implementation scanned 1,979 "
            "protected mixed-manifest raw rows and a separate operator view displayed five "
            "rows. The current analyzer has no mixed-manifest input and reads zero protected "
            "raw rows, fields, or arrays."
        ),
    }
    assert first["model_fits"] == 0
    assert first["source_integrity"]["mixed_role_development_manifest_opened"] is False
    assert (
        first["source_integrity"]["development_manifest_sha256_attested_by_locked_summary"]
        == ATTESTED_DEVELOPMENT_MANIFEST_SHA256
    )
    oof = first["headline"]["grouped_crossfit_oof"]
    validation = first["headline"]["fixed_development_validation"]
    assert oof["rows"] == 4977
    assert oof["recordings"] == 11
    assert validation["rows"] == 1383
    assert validation["recordings"] == 3
    assert oof["candidate_minus_baseline"]["macro_f1"] == pytest.approx(-0.0020380621610198713)
    assert validation["candidate_minus_baseline"]["macro_f1"] == pytest.approx(0.008050670493972123)
    assert oof["oracle_occlusion_fallback_minus_candidate"]["macro_f1"] == pytest.approx(
        0.0034030336801574856
    )

    expected_outputs = {
        "paired_rows.csv",
        "slice_metrics.csv",
        "cluster_uncertainty.json",
        "recording_resample_indices.npz",
        "summary.json",
    }
    assert {path.name for path in first_dir.iterdir()} == expected_outputs
    for name in expected_outputs:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    paired = pd.read_csv(first_dir / "paired_rows.csv")
    slices = pd.read_csv(first_dir / "slice_metrics.csv")
    uncertainty = json.loads((first_dir / "cluster_uncertainty.json").read_text())
    assert len(paired) == 4977 + 1383
    assert set(paired["development_role"]) == {"train", "validation"}
    assert set(slices["slice_type"]) == {
        "overall",
        "occlusion",
        "transition",
        "class",
        "recording",
        "fold",
        "occlusion_x_class",
    }
    for scope, clusters, permutations in (
        ("grouped_crossfit_oof", 11, 2048),
        ("fixed_development_validation", 3, 8),
    ):
        scope_result = uncertainty["scopes"][scope]
        comparisons = scope_result["comparisons"]
        assert set(comparisons) == {
            "candidate_vs_baseline",
            "oracle_occlusion_fallback_vs_baseline",
            "oracle_occlusion_fallback_vs_candidate",
        }
        plan_hashes = set()
        for comparison_id, comparison in comparisons.items():
            plan_hashes.add(comparison["resample_plan_sha256"])
            assert comparison["cluster_bootstrap"]["clusters"] == clusters
            assert comparison["exact_recording_swap"]["permutations"] == permutations
            assert (
                comparison["delta_orientation"]
                == comparison["cluster_bootstrap"]["delta_definition"]
            )
            assert (
                comparison["delta_orientation"]
                == comparison["exact_recording_swap"]["delta_definition"]
            )
            assert "two_sided_p" not in comparison["cluster_bootstrap"]["macro_f1"]
            assert "two_sided_p" in comparison["exact_recording_swap"]["macro_f1"]
            assert comparison["occluded_subgroup"]["estimand"].endswith("window_any_occluded=true")
            subgroup_bootstrap = comparison["occluded_subgroup"]["cluster_bootstrap"]
            assert subgroup_bootstrap["macro_f1"]["valid_resamples"] > 0
            assert subgroup_bootstrap["negative_log_likelihood"]["valid_resamples"] > 0
            assert comparison["occluded_subgroup"]["exact_recording_swap"]["available"]
            if comparison_id.startswith("oracle_"):
                assert comparison["oracle_labeled"] is True
                assert comparison["nondeployable"] is True
                assert comparison["selection_eligible"] is False
            else:
                assert comparison["oracle_labeled"] is False
                assert comparison["nondeployable"] is False
                assert comparison["selection_eligible"] is False
        assert plan_hashes == {scope_result["resample_plan_sha256"]}
    assert uncertainty["scopes"]["fixed_development_validation"]["precision_limited"]
    assert (
        "Only three validation recordings"
        in uncertainty["scopes"]["fixed_development_validation"]["precision_limit"]
    )
    assert uncertainty["resample_plan_artifact"]["sha256"] == sha256_file(
        first_dir / "recording_resample_indices.npz"
    )
    with np.load(first_dir / "recording_resample_indices.npz", allow_pickle=False) as plans:
        for scope, clusters in (
            ("grouped_crossfit_oof", 11),
            ("fixed_development_validation", 3),
        ):
            order = plans[f"{scope}__recording_order"]
            sampled = plans[f"{scope}__sampled_recording_indices"]
            assert order.shape == (clusters,)
            assert sampled.shape == (100, clusters)
            assert int(sampled.min()) >= 0
            assert int(sampled.max()) < clusters
            assert (
                recording_resample_plan_sha256(order, sampled)
                == uncertainty["scopes"][scope]["resample_plan_sha256"]
            )
    oof_comparisons = uncertainty["scopes"]["grouped_crossfit_oof"]["comparisons"]
    assert oof_comparisons["oracle_occlusion_fallback_vs_baseline"]["cluster_bootstrap"][
        "macro_f1"
    ]["point_estimate"] == pytest.approx(
        oof["oracle_occlusion_fallback_minus_baseline"]["macro_f1"]
    )
    assert oof_comparisons["oracle_occlusion_fallback_vs_candidate"]["cluster_bootstrap"][
        "macro_f1"
    ]["point_estimate"] == pytest.approx(
        oof["oracle_occlusion_fallback_minus_candidate"]["macro_f1"]
    )
    for comparison_id in (
        "oracle_occlusion_fallback_vs_baseline",
        "oracle_occlusion_fallback_vs_candidate",
    ):
        macro_interval = oof_comparisons[comparison_id]["cluster_bootstrap"]["macro_f1"]
        assert macro_interval["ci_95_low"] < 0.0 < macro_interval["ci_95_high"]
    assert first["oracle_occlusion_fallback"] == {
        "oracle_labeled": True,
        "requires_ground_truth_occlusion": True,
        "nondeployable": True,
        "selection_eligible": False,
        "promotion_eligible": False,
        "rule": (
            "use exact locked temporal baseline when window_any_occluded is true; "
            "otherwise use candidate"
        ),
        "purpose": "annotation-conditioned retrospective fallback diagnostic",
        "claim_limit": (
            "This oracle-labeled comparison is nondeployable, was not used for selection, "
            "cannot support promotion, and is not guaranteed to upper-bound either model."
        ),
    }
    assert len(first["fold_level_sensitivity"]["folds"]) == 5
    assert "fixed five-fold" in first["fold_level_sensitivity"]["conditional_inference_limit"]
    for name, evidence in first["artifacts"].items():
        assert sha256_file(first_dir / name) == evidence["sha256"]
