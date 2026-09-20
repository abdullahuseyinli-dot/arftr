import numpy as np
import pytest

from hac.correspondence_field_training import (
    ActorUniformSampler,
    conditional_motion_candidate,
    upright_balanced_accuracy,
    upright_class_weights,
)


def test_upright_weights_are_mean_one_and_reject_sitting():
    labels = np.asarray([1, 1, 1, 2])
    weights = upright_class_weights(labels, np.arange(4))
    assert np.isclose(weights[(labels - 1)].mean(), 1.0)
    with pytest.raises(ValueError, match="upright-only"):
        upright_class_weights(np.asarray([0, 1, 2]), np.arange(3))


def test_actor_uniform_sampler_is_deterministic_and_not_row_uniform():
    rows = np.arange(5)
    recordings = np.asarray(["r", "r", "r", "r", "r"])
    tracks = np.asarray(["large", "large", "large", "large", "small"])
    left = ActorUniformSampler(rows, recordings, tracks, seed=7).batch(4000)
    right = ActorUniformSampler(rows, recordings, tracks, seed=7).batch(4000)
    assert np.array_equal(left, right)
    assert 0.43 < np.mean(left == 4) < 0.57


def test_conditional_candidate_retains_exact_rows_and_never_switches_sitting():
    anchor = np.asarray(
        [[0.05, 0.85, 0.10], [0.70, 0.20, 0.10], [0.05, 0.85, 0.10]],
        dtype=np.float64,
    )
    candidate, eligible = conditional_motion_candidate(
        anchor, np.asarray([0.9, 0.9, 0.9]), np.asarray([True, True, False])
    )
    assert eligible.tolist() == [True, False, False]
    assert candidate[0].argmax() == 2
    assert np.array_equal(candidate[1:], anchor[1:])


def test_upright_balanced_accuracy_has_locked_definition():
    labels = np.asarray([0, 1, 1, 2, 2])
    probability = np.asarray([0.9, 0.1, 0.6, 0.8, 0.9])
    assert upright_balanced_accuracy(labels, probability) == pytest.approx(0.75)


def test_upright_balanced_accuracy_resolves_exact_tie_to_standing_like_candidate():
    labels = np.asarray([1, 2])
    assert upright_balanced_accuracy(labels, np.asarray([0.5, 0.5])) == pytest.approx(0.5)
