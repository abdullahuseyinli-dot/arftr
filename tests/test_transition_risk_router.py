from __future__ import annotations

import numpy as np

from hac.transition_risk_router import fit_transition_risk_router, router_features


def test_router_features_are_label_blind_and_aligned():
    anchor = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    candidate = np.array([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]])
    values, names = router_features(
        anchor,
        candidate,
        availability=np.array([[1.0, 0.0], [1.0, 1.0]]),
        quality=np.array([[0.4], [0.8]]),
    )
    assert values.shape == (2, len(names))
    assert np.isfinite(values).all()
    assert not any("label" in name or "scenario" in name for name in names)


def test_router_retains_anchor_exactly_when_action_is_false():
    anchor = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    candidate = np.array([[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]])
    features, names = router_features(anchor, candidate)
    labels = np.array([0, 1])
    router = fit_transition_risk_router(
        features,
        names,
        anchor,
        candidate,
        labels,
        harm_cost=2.0,
        random_state=42,
    )
    output, action = router.apply(anchor, candidate, features)
    assert output.shape == anchor.shape
    assert np.array_equal(output[~action], anchor[~action])


def test_router_rejects_subunit_harm_cost():
    anchor = np.tile(np.array([[0.8, 0.1, 0.1]]), (4, 1))
    candidate = np.tile(np.array([[0.1, 0.8, 0.1]]), (4, 1))
    features, names = router_features(anchor, candidate)
    with np.testing.assert_raises(ValueError):
        fit_transition_risk_router(
            features,
            names,
            anchor,
            candidate,
            np.array([0, 0, 0, 0]),
            harm_cost=0.5,
        )
