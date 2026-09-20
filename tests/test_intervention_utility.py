from __future__ import annotations

import numpy as np

from hac.intervention_utility import fit_intervention_verifier, verifier_features


def data(rows=36):
    rng = np.random.default_rng(5)
    anchor = rng.dirichlet(np.ones(3), size=rows)
    actions = np.repeat(anchor[:, None, :], 4, axis=1)
    for action in range(1, 4):
        actions[:, action] = rng.dirichlet(np.ones(3), size=rows)
    correction = rng.uniform(-0.4, 0.4, size=(rows, 2))
    scaffold = rng.uniform(0, 2, size=(rows, 4))
    quality = rng.normal(size=(rows, 6))
    labels = np.arange(rows) % 3
    return anchor, actions, correction, scaffold, quality, labels


def test_features_exclude_labels_and_fine_embedding():
    anchor, actions, correction, scaffold, quality, _ = data()
    x, names = verifier_features(anchor, actions, correction, scaffold, quality)
    assert x.shape == (36, 3, len(names))
    assert not any("label" in name or "support" in name or "fine" in name for name in names)
    assert np.isfinite(x).all()


def test_verifier_has_explicit_retain_and_deterministic_replay():
    anchor, actions, correction, scaffold, quality, labels = data()
    x, names = verifier_features(anchor, actions, correction, scaffold, quality)
    first, receipt = fit_intervention_verifier(x, anchor, actions, labels)
    first.feature_names = names
    p1, c1 = first.apply(anchor, actions, x)
    p2, c2 = first.apply(anchor, actions, x)
    assert np.array_equal(p1, p2)
    assert np.array_equal(c1, c2)
    assert np.array_equal(p1[c1 == 0], anchor[c1 == 0])
    assert set(np.unique(c1)) <= {0, 1, 2, 3}
    assert receipt["harm_utility"] == -2.0
