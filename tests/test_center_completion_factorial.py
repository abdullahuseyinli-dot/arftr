from __future__ import annotations

import numpy as np
import pytest

from hac.center_completion_factorial import (
    cosine_errors,
    locked_factorial_decision,
    robust_consensus,
    uniform_consensus,
)
from hac.center_evidence_completion import transport_consensus


def synthetic_inputs():
    values = np.array(
        [
            [[1.0, 8.0], [2.0, 4.0]],
            [[3.0, 6.0], [4.0, 2.0]],
            [[5.0, 4.0], [8.0, 0.0]],
            [[7.0, 2.0], [16.0, -2.0]],
        ],
        dtype=np.float32,
    )
    observed = np.array([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=bool)
    logits = np.array(
        [[0.0, 2.0], [0.0, 0.0], [0.0, -np.inf], [-np.inf, -np.inf]],
        dtype=np.float64,
    )
    return values, observed, logits


def test_uniform_consensus_uses_only_common_support() -> None:
    values, observed, logits = synthetic_inputs()
    result = uniform_consensus(values, observed, logits)
    np.testing.assert_allclose(result.features, [[3.0, 6.0], [3.0, 3.0]])
    np.testing.assert_array_equal(result.donor_count, [3, 2])
    assert result.available.all()


def test_robust_consensus_matches_original_p3_implementation() -> None:
    values, observed, logits = synthetic_inputs()
    actual = robust_consensus(values, observed, logits)
    expected = transport_consensus(
        values,
        observed,
        logits,
        np.zeros((2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(actual.features, expected.features)
    np.testing.assert_array_equal(actual.available, expected.available)


def test_robust_consensus_matches_original_on_random_packets() -> None:
    generator = np.random.default_rng(42)
    for _ in range(20):
        values = generator.normal(size=(4, 7, 13)).astype(np.float32)
        observed = generator.random((4, 7)) > 0.25
        observed[0] = True
        logits = generator.normal(size=(4, 7))
        logits[~observed] = -np.inf
        actual = robust_consensus(values, observed, logits)
        expected = transport_consensus(
            values,
            observed,
            logits,
            np.zeros((7, 13), dtype=np.float32),
        )
        np.testing.assert_array_equal(actual.features, expected.features)


def test_cosine_errors_full_dimension() -> None:
    candidate = np.array([[1.0, 0.0], [0.0, 1.0]])
    teacher = np.array([[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(cosine_errors(candidate, teacher), [0.0, 1.0])


def test_decision_transport_survives_when_locked_requirements_hold() -> None:
    means = {"A": 1.0, "B": 0.99, "C": 0.93, "D": 0.90}
    folds = {key: [value] * 5 for key, value in means.items()}
    decision = locked_factorial_decision(means, folds)
    assert decision["verdict"]["transport_survives"] is True
    assert decision["verdict"]["stop_transport_branch"] is False


def test_decision_stops_when_neither_transport_contrast_clears_five_percent() -> None:
    means = {"A": 1.0, "B": 0.96, "C": 0.97, "D": 0.93}
    folds = {key: [value] * 5 for key, value in means.items()}
    decision = locked_factorial_decision(means, folds)
    assert decision["verdict"]["transport_survives"] is False
    assert decision["verdict"]["stop_transport_branch"] is True


def test_invalid_contract_is_rejected() -> None:
    values, observed, logits = synthetic_inputs()
    with pytest.raises(ValueError):
        uniform_consensus(values[:3], observed[:3], logits[:3])
