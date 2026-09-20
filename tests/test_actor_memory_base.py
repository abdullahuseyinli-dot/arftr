"""Checks for exact base math and nested scenario/label ancestry."""

from __future__ import annotations

import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import hac.actor_memory_base as base_module
from hac.actor_memory_base import (
    NestedBaseCache,
    _decode,
    _fit,
    _predict,
    canonical_hash,
    group_splits,
    probability_metrics,
    training_populations,
)
from hac.video_fusion import decode_factorized_probabilities


@pytest.fixture
def fixture_data():
    rng = np.random.default_rng(914)
    scenarios = np.repeat(np.asarray([f"scenario-{i}" for i in range(9)]), 6)
    labels = np.tile(np.asarray([0, 1, 2, 0, 1, 2]), 9)
    x = rng.normal(size=(len(labels), 6)).astype(np.float32)
    x[:, :3] += np.eye(3, dtype=np.float32)[labels]
    features = {
        "base_long_vjepa": x[:, :3],
        "base_dual_scale": x,
        "base_ostm_posture": x[:, :5],
        "base_ostm_motion": x[:, 1:],
    }
    config = {
        "C_values": [0.01],
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 2000,
        "tolerance": 0.0001,
        "inner_folds": 3,
        "inner_seed": 42,
    }
    return {
        "features": features,
        "labels": labels,
        "scenarios": scenarios,
        "sample_ids": np.asarray([f"sample-{i:03d}" for i in range(len(labels))]),
        "config": config,
        "feature_sha256": "synthetic-immutable-feature-digest",
    }


def test_outside_label_mutation_changes_neither_request_nor_fitted_predictions(
    tmp_path, fixture_data
):
    population = np.arange(36)
    original = NestedBaseCache(tmp_path / "original", **fixture_data)
    changed_data = dict(fixture_data)
    changed_data["labels"] = fixture_data["labels"].copy()
    changed_data["labels"][36:] = (changed_data["labels"][36:] + 1) % 3
    changed = NestedBaseCache(tmp_path / "changed", **changed_data)
    assert original.request(population) == changed.request(population)
    assert original.directory(population).name == changed.directory(population).name
    # Separate empty caches force independent fits; a reused cache could hide leakage.
    with threadpool_limits(limits=1):
        original.prepare(population)
        changed.prepare(population)
    p1, receipt1 = original.load(population)
    p2, receipt2 = changed.load(population)
    np.testing.assert_array_equal(p1, p2)
    assert p1.dtype == np.float64
    assert receipt1["selection"] == receipt2["selection"]
    assert receipt1["selection_splits"] == receipt2["selection_splits"]


def test_training_label_change_changes_cache_identity(tmp_path, fixture_data):
    population = np.arange(36)
    original = NestedBaseCache(tmp_path, **fixture_data)
    changed_data = dict(fixture_data)
    changed_data["labels"] = fixture_data["labels"].copy()
    changed_data["labels"][0] = 1
    changed = NestedBaseCache(tmp_path, **changed_data)
    assert original.directory(population) != changed.directory(population)


def test_row_aligned_training_scenarios_are_part_of_cache_identity(tmp_path, fixture_data):
    population = np.arange(36)
    original = NestedBaseCache(tmp_path, **fixture_data)
    changed_data = dict(fixture_data)
    changed_data["scenarios"] = fixture_data["scenarios"].copy()
    changed_data["scenarios"][[0, 6]] = changed_data["scenarios"][[6, 0]]
    changed = NestedBaseCache(tmp_path, **changed_data)
    assert set(original.scenarios[population]) == set(changed.scenarios[population])
    assert original.directory(population) != changed.directory(population)


def test_scenario_crossing_predictions_are_refused_before_cache_read(tmp_path, fixture_data):
    cache = NestedBaseCache(tmp_path, **fixture_data)
    with pytest.raises(RuntimeError, match="scenario"):
        cache.held_predictions(np.arange(36), np.asarray([1, 40]))


def test_group_splits_cover_exact_population_and_exclude_scenarios(fixture_data):
    population = np.arange(36)
    labels, scenarios = fixture_data["labels"], fixture_data["scenarios"]
    splits = group_splits(labels, scenarios, population)
    np.testing.assert_array_equal(np.sort(np.concatenate([held for _, held in splits])), population)
    for train, held in splits:
        assert not set(scenarios[train]) & set(scenarios[held])
        assert set(labels[train]) == {0, 1, 2}
        assert not set(train) & set(held)
        assert set(train) | set(held) == set(population)
    with pytest.raises(ValueError, match="unique and sorted"):
        group_splits(labels, scenarios, population[::-1])


def test_enumerated_cache_populations_cover_every_required_nested_ancestor(fixture_data):
    y, scenarios = fixture_data["labels"], fixture_data["scenarios"]
    folds = np.repeat(np.arange(9) % 3, 6)
    populations = training_populations(y, scenarios, folds)
    for fold in np.unique(folds):
        outer_train = np.flatnonzero(folds != fold)
        outer_scenarios = set(scenarios[folds == fold])
        assert canonical_hash(outer_train.tolist()) in populations
        for inner_train, _ in group_splits(y, scenarios, outer_train):
            assert canonical_hash(inner_train.tolist()) in populations
            for base_train, _ in group_splits(y, scenarios, inner_train):
                assert canonical_hash(base_train.tolist()) in populations
                assert not set(scenarios[base_train]) & outer_scenarios
                # This final selection level must retain all three classes too.
                for fit, _ in group_splits(y, scenarios, base_train):
                    assert set(y[fit]) == {0, 1, 2}


def test_meta_probabilities_have_complete_excluded_scenario_ancestry(
    tmp_path, fixture_data, monkeypatch
):
    cache = NestedBaseCache(tmp_path, **fixture_data)
    population = np.arange(36)
    observed = []

    def held_predictions(train, held):
        assert not set(cache.scenarios[train]) & set(cache.scenarios[held])
        observed.append((train.copy(), held.copy()))
        return np.broadcast_to([0.2, 0.3, 0.5], (len(held), 3)).copy()

    monkeypatch.setattr(cache, "held_predictions", held_predictions)
    probabilities, ancestry = cache.meta_probabilities(population)
    assert len(ancestry) == 4
    np.testing.assert_array_equal(
        np.sort(np.concatenate([a["predicted_rows"] for a in ancestry])),
        np.arange(len(cache.labels)),
    )
    np.testing.assert_allclose(probabilities, np.broadcast_to([0.2, 0.3, 0.5], probabilities.shape))
    for record, (train, held) in zip(ancestry, observed, strict=True):
        assert record["base_cache"] == cache.directory(train).name
        assert record["predicted_rows"] == held.tolist()


def test_cache_receipt_tampering_is_rejected(tmp_path, fixture_data):
    cache = NestedBaseCache(tmp_path, **fixture_data)
    train = np.arange(36)
    with threadpool_limits(limits=1):
        cache.prepare(train)
    path = cache.directory(train) / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["request"]["train_labels_sha256"] = "tampered"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ancestry"):
        cache.load(train)


def test_factorized_class_order_matches_original_p5_decoder():
    upright = np.asarray([0.0, 0.1, 0.35, 0.9, 1.0], dtype=np.float64)
    motion = np.asarray([0.1, 0.9, 0.5, 1.0, 0.0], dtype=np.float64)
    reference = decode_factorized_probabilities(1 - upright, motion)
    np.testing.assert_allclose(_decode(upright, motion), reference, atol=1e-15, rtol=0)


def test_float32_decode_matches_original_p5_rounding_points_exactly():
    rng = np.random.default_rng(20260908)
    upright = rng.uniform(size=1000).astype(np.float32)
    motion = rng.uniform(size=1000).astype(np.float32)
    expected = decode_factorized_probabilities(1 - upright, motion)
    actual = _decode(upright, motion)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    # Complementing after promotion is a different algorithm at the last bits.
    wrong_order = decode_factorized_probabilities(1 - upright.astype(np.float64), motion)
    assert np.max(np.abs(wrong_order - expected)) > 1e-8


def test_float32_inner_factorized_selection_matches_historical_sitting_buffer(
    tmp_path, fixture_data
):
    cache = NestedBaseCache(tmp_path, **fixture_data)
    train = np.arange(36)
    splits = group_splits(cache.labels, cache.scenarios, train)
    posture_x = cache.features["base_ostm_posture"]
    motion_x = cache.features["base_ostm_motion"]
    sitting = np.full(len(cache.labels), np.nan)
    moving = np.full(len(cache.labels), np.nan)
    c = cache.config["C_values"][0]
    with threadpool_limits(limits=1):
        for fit_rows, held in splits:
            upright_rows = fit_rows[cache.labels[fit_rows] != 0]
            posture_fit = _fit(
                posture_x[fit_rows],
                (cache.labels[fit_rows] != 0).astype(int),
                cache.config,
                c,
                binary=True,
            )
            motion_fit = _fit(
                motion_x[upright_rows],
                (cache.labels[upright_rows] == 2).astype(int),
                cache.config,
                c,
                binary=True,
            )
            native_posture = _predict(posture_fit, posture_x[held])[:, 1]
            assert native_posture.dtype == np.float32
            sitting[held] = 1 - native_posture
            moving[held] = _predict(motion_fit, motion_x[held])[:, 1]
        expected = probability_metrics(
            cache.labels[train], decode_factorized_probabilities(sitting[train], moving[train])
        )
        _, selection, _ = cache._factorized(posture_x, motion_x, train, splits)
    assert selection["candidates"][0]["metrics"] == expected


def test_final_inference_uses_a_separate_exact_complement_batch(
    tmp_path, fixture_data, monkeypatch
):
    cache = NestedBaseCache(tmp_path, **fixture_data)
    train = np.arange(36)
    splits = group_splits(cache.labels, cache.scenarios, train)
    observed_batch_sizes = []
    original = base_module._predict

    def record_predict(fit, values):
        observed_batch_sizes.append(len(values))
        return original(fit, values)

    monkeypatch.setattr(base_module, "_predict", record_predict)
    with threadpool_limits(limits=1):
        cache._direct(cache.features["base_long_vjepa"], train, splits)
        assert observed_batch_sizes[-2:] == [54, 18]
        cache._factorized(
            cache.features["base_ostm_posture"], cache.features["base_ostm_motion"], train, splits
        )
        assert observed_batch_sizes[-4:] == [54, 54, 18, 18]


def test_tied_candidate_selection_prefers_lower_c_for_both_heads(tmp_path, fixture_data):
    data = dict(fixture_data)
    data["features"] = {key: np.zeros_like(value) for key, value in data["features"].items()}
    data["config"] = {**data["config"], "C_values": [0.01, 0.0001]}
    cache = NestedBaseCache(tmp_path, **data)
    train = np.arange(36)
    with threadpool_limits(limits=1):
        receipt = cache.prepare(train)
    for name in ("base_long_vjepa", "base_dual_scale"):
        assert receipt["selection"][name]["selected"]["C"] == 0.0001
    selected = receipt["selection"]["base_ostm_factorized"]["selected"]
    assert selected["posture_C"] == selected["motion_C"] == 0.0001
