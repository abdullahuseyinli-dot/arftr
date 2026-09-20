from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedGroupKFold

from experiments import audit_okutama_sear_matrix as audit
from experiments.run_okutama_video_probe import holm_adjust, paired_statistics
from hac.actor_memory_base import probability_metrics
from hac.sear_evaluation import reference_strata_summary, slot_diagnostics
from hac.sear_training import make_model


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def protocol():
    value = audit.read_json(audit.ROOT / "experiments/okutama_sear_matrix_protocol.json")
    value["training"]["max_epochs"] = 1
    return value


def make_workload(directory: Path, arm="a5_sear"):
    p = protocol()
    scenarios = np.repeat(np.arange(15), 3)
    data = {
        "sample_ids": np.array([f"sample-{i}" for i in range(45)]),
        "labels": np.tile(np.arange(3), 15),
        "scenarios": scenarios,
        "folds": scenarios % 5,
    }
    lock = {"protocol": p, "parameter_counts": {arm: 1}}
    write_json(directory / "execution_lock.json", lock)
    lock_hash = audit.sha256(directory / "execution_lock.json")
    train, held = np.flatnonzero(data["folds"] != 0), np.flatnonzero(data["folds"] == 0)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = [
        (train[a], train[b])
        for a, b in splitter.split(
            np.zeros(len(train)), data["labels"][train], data["scenarios"][train]
        )
    ]
    fit_records, prediction_records = [], []

    def fit(fit_rows, held_rows, lr, wd, seed, inner=None, epochs=None):
        request = audit.independent_request(
            data,
            arm,
            fit_rows,
            held_rows,
            lr,
            wd,
            seed,
            "cuda",
            lock_hash,
            0,
            inner=inner,
            epochs=epochs,
        )
        path = directory / "fits" / audit.canonical(request)
        path.mkdir(parents=True)
        write_json(path / "request.json", request)
        confidence = 0.6 if lr == 0.0003 else 0.7
        probabilities = np.full((len(held_rows), 3), (1 - confidence) / 2, dtype=np.float32)
        probabilities[np.arange(len(held_rows)), data["labels"][held_rows]] = confidence
        arrays = {
            "probabilities": probabilities,
            "local_logits": np.log(probabilities),
            "video_logits": np.zeros_like(probabilities),
        }
        if arm in audit.TEMPLATE_ARMS:
            arrays["slot_masses"] = np.full((len(held_rows), 6), 0.1, dtype=np.float32)
            arrays["slot_positions"] = np.zeros((len(held_rows), 6, 2), dtype=np.float32)
            arrays["slot_descriptors"] = np.ones((len(held_rows), 6, 32), dtype=np.float32)
        if epochs is not None:
            arrays["coordinate_shuffle_probabilities"] = probabilities.copy()
            arrays["outer_border_mask_probabilities"] = probabilities.copy()
            shifted = arrays["local_logits"] - arrays["local_logits"].max(1, keepdims=True)
            exp = np.exp(shifted)
            arrays["video_removed_probabilities"] = exp / exp.sum(1, keepdims=True)
        torch.save(
            {"state_dict": {"weight": torch.ones(1)}, "request": request, "epoch": 1},
            path / "checkpoint.pt",
        )
        np.savez_compressed(
            path / "predictions.npz", sample_ids=data["sample_ids"][held_rows], **arrays
        )
        measured = audit.metrics(data["labels"][held_rows], probabilities)
        history = [{"epoch": 1, "training_loss": 0.9, "classification_loss": 0.8}]
        if epochs is None:
            history[0]["validation_metrics"] = measured
        torch.save(
            {
                "request": request,
                "maximum_epochs": 1,
                "next_epoch": 2,
                "history": history,
                "best_epoch": 1 if epochs is None else 0,
                "best_state": {"weight": torch.ones(1)} if epochs is None else None,
                "model_state": {"weight": torch.ones(1)},
            },
            path / "progress.pt",
        )
        receipt = {
            "status": "SEAR_FIT_COMPLETE",
            "request": request,
            "selected_epoch": 1,
            "epochs_run": 1,
            "history": history,
            "train_rows": len(fit_rows),
            "held_rows": len(held_rows),
            "parameters": 1,
            "artifacts": {
                name: audit.file_record(path / name)
                for name in ("request.json", "checkpoint.pt", "predictions.npz", "progress.pt")
            },
        }
        if epochs is None:
            receipt["held_metrics"] = measured
        else:
            receipt["outer_held_metrics_embargoed"] = True
        write_json(path / "receipt.json", receipt)
        fit_records.append(audit.file_record(path / "receipt.json"))
        prediction_records.append(receipt["artifacts"]["predictions.npz"])
        return arrays

    candidates = []
    for lr in p["training"]["learning_rates"]:
        for wd in p["training"]["weight_decays"]:
            oof = np.full((45, 3), np.nan)
            for inner, (a, b) in enumerate(splits):
                oof[b] = fit(a, b, lr, wd, 42, inner=inner)["probabilities"]
            candidates.append(
                {
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "inner_metrics": audit.metrics(data["labels"][train], oof[train]),
                    "inner_best_epochs": [1, 1, 1],
                }
            )
    selected = min(
        candidates,
        key=lambda x: (
            -x["inner_metrics"]["macro_f1"],
            x["inner_metrics"]["nll"],
            x["learning_rate"],
            x["weight_decay"],
        ),
    )
    outputs = [
        fit(train, held, selected["learning_rate"], selected["weight_decay"], seed, epochs=1)
        for seed in (42, 43, 44)
    ]
    path = directory / "workloads" / arm / "fold-0"
    path.mkdir(parents=True)
    arrays = {"seed_" + name: np.stack([value[name] for value in outputs]) for name in outputs[0]}
    np.savez_compressed(path / "predictions.npz", sample_ids=data["sample_ids"][held], **arrays)
    receipt = {
        "status": "SEAR_WORKLOAD_COMPLETE",
        "request": {"execution_lock_sha256": lock_hash, "arm": arm, "fold": 0},
        "fit_device": "cuda",
        "selected": selected,
        "candidates": candidates,
        "outer_refit_epochs": 1,
        "unique_fits": 15,
        "fit_receipts": fit_records,
        "fit_prediction_receipts": prediction_records,
        "artifact": audit.file_record(path / "predictions.npz"),
    }
    write_json(path / "receipt.json", receipt)
    return lock, data, receipt


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independent_metrics_and_randomization_match_reference(dtype):
    rng = np.random.default_rng(884)
    labels = rng.integers(3, size=121)
    scenarios = np.repeat(np.arange(11), 11)
    candidate = rng.dirichlet([0.5, 1, 2], len(labels)).astype(dtype)
    reference = rng.dirichlet([1, 0.5, 2], len(labels)).astype(dtype)
    audit.same(probability_metrics(labels, candidate), audit.metrics(labels, candidate), "metrics")
    expected = paired_statistics(
        labels, candidate, reference, scenarios, bootstrap_resamples=1000, bootstrap_seed=55
    )
    actual = audit.contrast(labels, candidate, reference, scenarios, resamples=1000, seed=55)
    audit.same(expected, actual, "contrasts")


def test_holm_and_missing_class_bootstrap():
    values = {"z": 0.01, "a": 0.01, "b": 0.04, "c": 0.3, "d": 0.9, "e": 1.0}
    assert audit.holm(values) == holm_adjust(values)
    labels = np.repeat(np.arange(3), 4)
    p = np.eye(3)[labels] * 0.8 + 0.2 / 3
    q = np.roll(p, 1, axis=1)
    result = audit.contrast(labels, p, q, labels, resamples=1000, seed=4)
    assert result["bootstrap_valid_fraction"] < 1
    assert result["bootstrap_draws_used"] == 1000
    assert result["exact_swap_assignments"] == 8


def test_independent_reference_strata_include_empty_and_single_class():
    rng = np.random.default_rng(31)
    labels = rng.integers(3, size=40)
    candidate, reference = (rng.dirichlet([1, 1, 1], 40) for _ in range(2))
    shared = rng.random(40) < 0.4
    masks = {
        "empty": np.zeros(40, dtype=bool),
        "single": labels == 0,
        "all": np.ones(40, dtype=bool),
    }
    expected = reference_strata_summary(labels, candidate, {"ref": reference}, masks, shared)[
        "references"
    ]["ref"]["strata"]
    for name, mask in masks.items():
        audit.same(
            expected[name], audit.stratum_pair(labels, candidate, reference, mask, shared), name
        )
    assert audit.metrics(labels[:0], candidate[:0])["macro_f1"] is None


def test_per_seed_slot_mass_core_matches_report_without_cross_seed_slot_alignment():
    rng = np.random.default_rng(13)
    masses = rng.uniform(size=(3, 27, 6)).astype(np.float32)
    masses[:, 0] = 0
    masses[:, 1, 1:] = 0
    expected = slot_diagnostics(masses, seed_ids=[42, 43, 44])
    actual = audit.slot_mass_core(masses, 0.05)
    audit.same(actual, expected, "per-seed mass core")
    assert actual["per_seed"]["42"]["zero_or_negligible_total_mass_rows"] == 1


def test_historical_boundary_gate_keeps_old_source_p6_comparison_basis():
    results = {arm: {"macro_f1": 0.84, "nll": 0.3, "brier": 0.2} for arm in audit.ARMS}
    new = {
        "all": {"net_corrections": 3},
        "legacy_boundary": {"harms": 0},
        "pure_persistent_error": {"original_shared_repairs": 2},
    }
    old = {"legacy_boundary": {"harms": 73}}
    gates = audit.decision_gates(results, new, {str(i): 0.0 for i in range(11)}, old)
    assert not gates["legacy_boundary_harms_fewer_than_historical_73_vs_old_source_p6"]


@pytest.mark.parametrize("arm", ["a0_center_cls", "a5_sear"])
def test_reconstructs_fifteen_fits_selection_and_exact_outer_arrays(tmp_path, arm):
    lock, data, _ = make_workload(tmp_path, arm)
    arrays, fits, receipt = audit.audit_workload(
        tmp_path, audit.Evidence(), lock, data, arm, 0, "cuda"
    )
    assert len(fits) == 15
    assert receipt["selected"]["learning_rate"] == 0.001
    assert receipt["selected"]["weight_decay"] == 0.0001
    assert arrays["seed_probabilities"].shape == (3, 9, 3)
    assert ("seed_slot_masses" in arrays) == (arm == "a5_sear")


def test_checkpoint_schema_must_match_label_blind_resource(tmp_path):
    lock, data, _ = make_workload(tmp_path)
    with pytest.raises(RuntimeError, match="schema differs"):
        audit.audit_workload(
            tmp_path,
            audit.Evidence(),
            lock,
            data,
            "a5_sear",
            0,
            "cuda",
            {"wrong": {"shape": [1], "dtype": "torch.float32"}},
        )


@pytest.mark.parametrize("arm", audit.ARMS)
def test_independently_counted_active_state_matches_actual_architecture(arm):
    model = make_model(protocol(), arm, device="cpu")
    state = model.state_dict()
    excluded = {
        name
        for name in state
        if arm in (audit.ARMS[3], audit.ARMS[4]) and name.startswith("extractor.nuisance.")
    }
    if arm == audit.ARMS[4]:
        excluded.add("template_positions")
    assert (
        sum(value.numel() for name, value in state.items() if name not in excluded)
        == audit.EXPECTED_PARAMETERS[arm]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "selected",
        "held_ids",
        "checkpoint_request",
        "outer_validation",
        "workload_seed",
        "workload_lock",
        "fit_ancestry",
        "hash",
        "progress_state",
        "progress_history",
    ],
)
def test_rejects_corrupted_ancestry_selection_and_predictions(tmp_path, mutation):
    lock, data, workload = make_workload(tmp_path)
    workload_path = tmp_path / "workloads/a5_sear/fold-0/receipt.json"
    fit_path = Path(workload["fit_receipts"][-1]["path"]).parent
    receipt = audit.read_json(fit_path / "receipt.json")
    if mutation == "selected":
        workload["selected"] = copy.deepcopy(workload["candidates"][0])
    elif mutation == "workload_lock":
        workload["request"]["execution_lock_sha256"] = "0" * 64
    elif mutation == "fit_ancestry":
        workload["fit_receipts"][0] = workload["fit_receipts"][1]
    elif mutation == "workload_seed":
        path = workload_path.parent / "predictions.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays["seed_probabilities"][0, 0] = [0.2, 0.3, 0.5]
        np.savez_compressed(path, **arrays)
        workload["artifact"] = audit.file_record(path)
    elif mutation == "held_ids":
        path = fit_path / "predictions.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
        np.savez_compressed(path, **arrays)
        receipt["artifacts"]["predictions.npz"] = audit.file_record(path)
    elif mutation == "checkpoint_request":
        path = fit_path / "checkpoint.pt"
        value = torch.load(path, weights_only=True)
        value["request"]["seed"] = 99
        torch.save(value, path)
        receipt["artifacts"]["checkpoint.pt"] = audit.file_record(path)
    elif mutation == "outer_validation":
        receipt["history"][0]["validation_metrics"] = {"macro_f1": 1.0}
    elif mutation == "hash":
        receipt["artifacts"]["checkpoint.pt"]["sha256"] = "0" * 64
    elif mutation in ("progress_state", "progress_history"):
        path = fit_path / "progress.pt"
        value = torch.load(path, weights_only=True)
        if mutation == "progress_state":
            value["model_state"]["weight"] += 1
        else:
            value["history"][0]["training_loss"] += 1
        torch.save(value, path)
        receipt["artifacts"]["progress.pt"] = audit.file_record(path)
    write_json(fit_path / "receipt.json", receipt)
    write_json(workload_path, workload)
    with pytest.raises(RuntimeError):
        audit.audit_workload(tmp_path, audit.Evidence(), lock, data, "a5_sear", 0, "cuda")


def test_request_outer_labels_do_not_enter_identity():
    data = {"sample_ids": np.array(["a", "b", "c"]), "labels": np.array([0, 1, 2])}
    args = (data, "a5_sear", np.array([0, 1]), np.array([2]), 0.001, 0.01, 42, "cuda", "hash", 0)
    first = audit.independent_request(*args, epochs=3)
    data["labels"][2] = 0
    assert audit.independent_request(*args, epochs=3) == first
    assert first["selection_labels_sha256"] is None


def test_partial_matrix_does_not_open_scores(tmp_path, monkeypatch):
    p = protocol()
    p["training"]["max_epochs"] = 30
    write_json(
        tmp_path / "execution_lock.json",
        {
            "status": "SEAR_SIX_ARM_450_FIT_LOCKED_BEFORE_CLASSIFIER_FITTING",
            "protocol": p,
            "classifier_fits_completed_at_lock": 0,
            "protected_rows_read": 0,
        },
    )
    opened = []
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(RuntimeError, match="Incomplete or foreign workload inventory"):
        audit.audit(tmp_path)
    assert not opened


def test_evidence_refuses_path_substitution_hash_changes_and_live_edits(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"original")
    b.write_bytes(b"original")
    evidence = audit.Evidence()
    with pytest.raises(RuntimeError, match="path substitution"):
        evidence.check(audit.file_record(a), b)
    evidence.check(audit.file_record(a), a)
    a.write_bytes(b"changed payload")
    with pytest.raises(RuntimeError, match="Input changed"):
        evidence.finish()


def test_unmatched_inner_history_stopping_is_rejected(tmp_path):
    lock, data, workload = make_workload(tmp_path)
    lock["protocol"]["training"]["max_epochs"] = 30
    # The fit's one-epoch history now violates the unchanged seven-epoch patience.
    # Keep the on-disk execution lock fixed: the helper must catch the history rule.
    path = Path(workload["fit_receipts"][0]["path"])
    receipt = audit.read_json(path)
    expected = receipt["request"]
    outer_train = np.flatnonzero(data["folds"] != 0)
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    a, b = next(
        split.split(
            np.zeros(len(outer_train)), data["labels"][outer_train], data["scenarios"][outer_train]
        )
    )
    with pytest.raises(RuntimeError, match="stopped early"):
        audit.audit_fit(
            tmp_path,
            audit.Evidence(),
            expected,
            data,
            outer_train[a],
            outer_train[b],
            lock["protocol"],
            1,
        )


def test_recursive_comparison_rejects_nonfinite_and_preserves_tolerance():
    audit.same({"a": [0.1]}, {"a": [0.1 + 1e-12], "extra": "allowed"}, "test")
    for value in (float("nan"), 0.100001):
        with pytest.raises(RuntimeError, match="Numeric mismatch"):
            audit.same(0.1, value, "test")


def test_population_equality_handles_unicode_ids_nan_metadata_and_dtype_changes():
    assert audit.equal_array(np.array(["sample-a"]), np.array(["sample-a"]))
    assert not audit.equal_array(np.array(["sample-a"]), np.array(["sample-b"]))
    assert audit.equal_array(np.array([float("nan")]), np.array([float("nan")]))
    assert not audit.equal_array(np.array([1], dtype=np.int32), np.array([1], dtype=np.int64))


@pytest.mark.parametrize("mutation", [None, "missing", "extra", "incomplete_status", "orphan"])
def test_exact_complete_inventory_gate(tmp_path, mutation):
    p = protocol()
    p["training"]["max_epochs"] = 30
    lock = {
        "status": "SEAR_SIX_ARM_450_FIT_LOCKED_BEFORE_CLASSIFIER_FITTING",
        "protocol": p,
        "classifier_fits_completed_at_lock": 0,
        "protected_rows_read": 0,
    }
    write_json(tmp_path / "execution_lock.json", lock)
    for arm in audit.ARMS:
        for fold in range(5):
            write_json(
                tmp_path / "workloads" / arm / f"fold-{fold}" / "receipt.json",
                {"status": "SEAR_WORKLOAD_COMPLETE"},
            )
    for i in range(449 if mutation == "missing" else 451 if mutation == "extra" else 450):
        path = tmp_path / "fits" / f"{i:064x}"
        write_json(
            path / "receipt.json",
            {
                "status": "INCOMPLETE"
                if i == 0 and mutation == "incomplete_status"
                else "SEAR_FIT_COMPLETE"
            },
        )
        for name in ("request.json", "checkpoint.pt", "predictions.npz", "progress.pt"):
            (path / name).write_bytes(b"not evaluated by the inventory gate")
    if mutation == "orphan":
        (tmp_path / "fits/orphan").mkdir()
    write_json(
        tmp_path / audit.RESULT_VERSION / "summary.json", {"not": "evaluated by inventory gate"}
    )
    (tmp_path / audit.RESULT_VERSION / "oof_probabilities.npz").write_bytes(b"not evaluated")
    (tmp_path / audit.RESULT_VERSION / "slot_evidence.npz").write_bytes(b"not evaluated")
    if mutation is None:
        _, fits = audit.completed_inventory(tmp_path)
        assert len(fits) == 450
    else:
        with pytest.raises(RuntimeError):
            audit.completed_inventory(tmp_path)
