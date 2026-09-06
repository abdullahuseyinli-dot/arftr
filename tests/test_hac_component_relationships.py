import hashlib

import numpy as np
import pytest

from experiments.analyze_hac_component_relationships import (
    class_standardized_contrast,
    clustered_model_draws,
    error_overlap,
    locked_bytes,
    metrics,
    run,
    subgroup_contrast,
)


def fixture_predictions():
    # Unequal group sizes expose accidentally averaging per-group F1/loss.
    labels = np.array([0, 1, 2, 0, 2, 1])
    groups = np.array([0, 0, 0, 1, 2, 2])
    baseline = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
        ]
    )
    candidate = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
        ]
    )
    indices = np.array([[0, 0, 0], [0, 1, 2], [2, 1, 2], [1, 1, 1]])
    return labels, groups, baseline, candidate, indices


def test_cluster_sufficient_statistics_match_literal_row_replicates():
    labels, groups, baseline, candidate, indices = fixture_predictions()
    result = clustered_model_draws(
        labels, {"base": baseline, "candidate": candidate}, groups, indices
    )
    for i, draw in enumerate(indices):
        selected = np.concatenate([np.flatnonzero(groups == group) for group in draw])
        for j, values in enumerate((baseline, candidate)):
            expected = metrics(labels[selected], values[selected])
            np.testing.assert_allclose(result[i, j], [expected["macro_f1"], expected["nll"]])


def test_subgroup_interaction_keeps_empty_draws_visible_and_matches_row_bootstrap():
    labels, groups, baseline, candidate, indices = fixture_predictions()
    subgroup = np.array([True, False, True, False, False, True])
    result = subgroup_contrast(labels, baseline, candidate, groups, indices, subgroup)
    delta = (candidate.argmax(1) != labels).astype(float) - (baseline.argmax(1) != labels)
    expected = []
    for draw in indices:
        selected = np.concatenate([np.flatnonzero(groups == group) for group in draw])
        flag = subgroup[selected]
        if flag.any() and (~flag).any():
            expected.append(delta[selected][flag].mean() - delta[selected][~flag].mean())
    assert result["error_rate"]["invalid_draws"] == 1
    np.testing.assert_allclose(result["error_rate"]["ci95"], np.quantile(expected, [0.025, 0.975]))
    assert result["error_rate"]["point"] == pytest.approx(
        delta[subgroup].mean() - delta[~subgroup].mean()
    )


def test_error_overlap_does_not_call_accuracy_oracle_a_macro_f1_bound():
    labels, _, baseline, candidate, _ = fixture_predictions()
    result = error_overlap(labels, baseline, candidate)
    assert result["rescued"] == 3
    assert result["harmed"] == 1
    assert result["both_wrong"] == 0
    assert result["either_correct_accuracy_oracle"] == 1
    assert "not a macro-F1 bound" in result["oracle_note"]


def test_class_standardization_matches_explicit_recall_contrasts():
    labels, groups, baseline, candidate, indices = fixture_predictions()
    subgroup = np.array([True, False, True, False, False, True])
    result = class_standardized_contrast(labels, baseline, candidate, groups, indices, subgroup)
    error_delta = (candidate.argmax(1) != labels).astype(float) - (baseline.argmax(1) != labels)
    contrasts = [
        error_delta[subgroup & (labels == c)].mean() - error_delta[~subgroup & (labels == c)].mean()
        for c in range(3)
    ]
    assert result["error_rate"]["point"] == pytest.approx(np.mean(contrasts))
    assert result["error_rate"]["valid_draws"] == 1
    assert result["error_rate"]["invalid_draws"] == 3
    np.testing.assert_allclose(result["error_rate"]["ci95"], [np.mean(contrasts)] * 2)


def test_locked_input_rejects_mutation_before_parsing(tmp_path):
    path = tmp_path / "evidence.csv"
    original = b"arbitrary opaque evidence"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    assert locked_bytes(path, digest) == original
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        locked_bytes(path, digest)


def test_existing_output_rejected_before_any_source_read(tmp_path, monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError("Should not access evidence after output preflight fails")

    monkeypatch.setattr(
        "experiments.analyze_hac_component_relationships.locked_bytes", forbidden_read
    )
    with pytest.raises(ValueError, match="new directory"):
        run(tmp_path, tmp_path)
