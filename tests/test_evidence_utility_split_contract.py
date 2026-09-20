from copy import deepcopy

import audit_okutama_evidence_utility as audit
import numpy as np
import pytest
import run_okutama_evidence_utility as runner
import torch


def test_protocol_and_locked_task_population():
    protocol = runner.read_json(runner.PROTOCOL)
    runner.validate_protocol(protocol)
    data = runner.load_task_metadata()
    assert len(data["labels"]) == 4977
    assert np.bincount(data["folds"]).tolist() == [1196, 926, 1061, 956, 838]
    for fold in range(5):
        assert not set(data["scenarios"][data["folds"] == fold]) & set(
            data["scenarios"][data["folds"] != fold]
        )


def test_protocol_rejects_loss_denominator_or_fit_count_change():
    protocol = deepcopy(runner.read_json(runner.PROTOCOL))
    protocol["reader"]["loss"] = "sum weighted loss divided by sum weights"
    with pytest.raises(RuntimeError, match="protocol changed"):
        runner.validate_protocol(protocol)
    protocol = deepcopy(runner.read_json(runner.PROTOCOL))
    protocol["reader"]["initial_fits"] = 19
    with pytest.raises(RuntimeError, match="protocol changed"):
        runner.validate_protocol(protocol)


def test_uniform_center_schedule_is_reproducible_and_arm_independent():
    train = np.arange(137, dtype=np.int64)
    first = runner.batch_schedule(train, steps=7, batch_size=64, seed=42)
    second = runner.batch_schedule(train, steps=7, batch_size=64, seed=42)
    assert np.array_equal(first, second)
    assert first.shape == (7, 64)
    assert set(np.unique(first)) == set(train)
    assert not np.array_equal(first, runner.batch_schedule(train, steps=7, batch_size=64, seed=43))


def test_held_feature_poison_cannot_change_scaler_or_checkpoint():
    rng = np.random.default_rng(8)
    train_x = rng.normal(size=(70, 11526)).astype(np.float32)
    train_y = np.arange(70, dtype=np.int64) % 3
    held = rng.normal(size=(3, 11526)).astype(np.float32)
    kwargs = dict(
        steps=2,
        batch_size=64,
        seed=42,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        device=torch.device("cpu"),
    )
    checkpoint_a, predictions_a, _ = runner.fit_reader(train_x, train_y, held, **kwargs)
    poisoned = held.copy()
    poisoned[:, :200] += 1000
    checkpoint_b, predictions_b, _ = runner.fit_reader(train_x, train_y, poisoned, **kwargs)
    assert checkpoint_a.keys() == checkpoint_b.keys()
    assert all(np.array_equal(checkpoint_a[key], checkpoint_b[key]) for key in checkpoint_a)
    assert not np.array_equal(predictions_a, predictions_b)


def test_class_weights_are_outer_training_only_formula():
    labels = np.asarray([0, 0, 1, 2, 2, 2])
    assert np.allclose(runner.class_weights(labels), [1.0, 2.0, 2 / 3])
    with pytest.raises(RuntimeError, match="missing a class"):
        runner.class_weights(np.asarray([0, 1, 1]))


def test_metrics_pin_all_three_classes_and_ece_edges():
    labels = np.asarray([0, 0, 1])
    probabilities = np.asarray([[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.1, 0.9, 0.0]])
    metrics = audit.fixed_metrics(labels, probabilities)
    assert metrics["per_class_f1"] == [1.0, 1.0, 0.0]
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["ece15"] == pytest.approx((0 + 0.4 + 0.1) / 3)


def test_paired_scenario_bootstrap_uses_pooled_fixed_class_confusions():
    labels = np.tile(np.arange(3), 11)
    scenarios = np.repeat(np.asarray([f"s{i}" for i in range(11)]), 3)
    candidate = np.eye(3)[labels].astype(float)
    reference = np.roll(candidate, 1, axis=1)
    result = audit.paired_scenario_bootstrap(
        labels, candidate, reference, scenarios, resamples=1000
    )
    assert result["observed"] == pytest.approx(1.0)
    assert result["interval95"][0] == pytest.approx(1.0)
