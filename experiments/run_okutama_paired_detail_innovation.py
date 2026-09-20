"""Run the prospective nested Paired Detail Innovation experiment.

Stages are immutable and resumable: prepare -> preflight -> queue -> summarize.
The queue fits all inner producers for one arm/fold before the training-only
capacity gate. It fits the final outer producer only after that gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import run_okutama_fine_detail_probe as fine_base
from experiments import run_okutama_fine_detail_probe_v2 as fine_v2
from experiments import run_okutama_source_posture as source_posture
from experiments import run_okutama_transition_risk_router as router_base
from hac.actor_memory_base import canonical_hash, file_sha256
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors
from hac.intervention_utility import (
    InterventionVerifier,
    fit_intervention_verifier,
    verifier_features,
)
from hac.paired_detail_innovation import ARMS, PairedDetailInnovation, pdi_loss

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_paired_detail_innovation_protocol.json"
REVIEW_PROTOCOL = ROOT / ".runs/research_20260919/team_aerial_review_v1/NEXT_PHASE_PROTOCOL.json"
ARFTR = ROOT / ".runs/research_20260912/arftr_v1/results/v0001/oof_probabilities.npz"
INNER_ANCESTORS = ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_ancestors"
DEFAULT_RUN = ROOT / ".runs/research_20260919/paired_detail_innovation_v1"
LOCK_STATUS = "PAIRED_DETAIL_INNOVATION_LOCKED_BEFORE_TASK_FITS"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], value, equal_nan=True)
                for name, value in arrays.items()
            ):
                raise RuntimeError(f"immutable array differs: {path}")
        return
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def data() -> dict[str, np.ndarray]:
    protocol = fine_v2.read_json(fine_v2.PROTOCOL_PATH)
    d = fine_base.validate_inputs(protocol)
    with np.load(ARFTR, allow_pickle=False) as saved:
        arm_index = list(saved["arms"].astype(str)).index("r5_arftr_full")
        d["anchor"] = saved["mean_probabilities"][arm_index].astype(np.float64)
        d["scenarios"] = saved["scenarios"].astype(str)
        if (
            not np.array_equal(saved["sample_ids"].astype(str), d["sample_ids"])
            or not np.array_equal(saved["labels"], d["labels"])
            or not np.array_equal(saved["folds"], d["folds"])
        ):
            raise RuntimeError("PDI cohort differs from retained ARFTR")
    return d


def scenario_assignment(d: dict[str, np.ndarray], outer: int) -> dict[str, int]:
    train = np.flatnonzero(d["folds"] != outer)
    scenarios = sorted(set(d["scenarios"][train].tolist()))
    k = min(5, len(scenarios))
    return {scenario: index % k for index, scenario in enumerate(scenarios)}


def inner_rows(d: dict[str, np.ndarray], outer: int, inner: int) -> tuple[np.ndarray, np.ndarray]:
    assignment = scenario_assignment(d, outer)
    outer_train = np.flatnonzero(d["folds"] != outer)
    held = outer_train[np.asarray([assignment[str(d["scenarios"][row])] == inner for row in outer_train])]
    train = np.setdiff1d(outer_train, held, assume_unique=True)
    return train, held


def _f_artifact(d: dict[str, np.ndarray], outer: int, inner: int) -> tuple[dict, dict[str, np.ndarray]]:
    expected_train, expected_held = inner_rows(d, outer, inner)
    router_train, router_held = router_base._inner_rows(router_base.data(), outer, inner)
    if not np.array_equal(expected_train, router_train) or not np.array_equal(expected_held, router_held):
        raise RuntimeError("PDI lexicographic split does not match audited ancestor split")
    pid = router_base.population_id(outer, inner, router_base.data())
    receipt, arrays = router_base._f_artifact(INNER_ANCESTORS, pid)
    prediction_rows = arrays["prediction_rows"].astype(np.int64)
    locations = np.searchsorted(prediction_rows, expected_held)
    if np.any(locations >= len(prediction_rows)) or not np.array_equal(prediction_rows[locations], expected_held):
        raise RuntimeError("audited inner F does not cover held rows")
    arrays["pdi_held_locations"] = locations
    arrays["pdi_train_rows"] = expected_train
    arrays["pdi_held_rows"] = expected_held
    return receipt, arrays


def inner_anchors(d: dict[str, np.ndarray], outer: int, inner: int) -> tuple[np.ndarray, np.ndarray, dict]:
    receipt, arrays = _f_artifact(d, outer, inner)
    train, held = arrays["pdi_train_rows"], arrays["pdi_held_rows"]
    parameters = ARFTRParameters(*arrays["parameters"].tolist())
    protocol = read_json(ROOT / "experiments/okutama_arftr_protocol.json")
    train_anchor = apply_arftr(
        arrays["m4_inner"], arrays["p6_inner"], arrays["a3_inner"],
        exact_track_neighbors(router_base.data(), train), parameters,
        epsilon=protocol["factorization"]["epsilon"],
    )
    held_anchor = arrays["mean_probabilities"][arrays["pdi_held_locations"]]
    ancestry = {
        "population_id": receipt["population_id"],
        "population_receipt_sha256": file_sha256(
            INNER_ANCESTORS / "populations" / receipt["population_id"] / "receipt.json"
        ),
        "train_scenarios": sorted(set(d["scenarios"][train].tolist())),
        "held_scenarios": sorted(set(d["scenarios"][held].tolist())),
        "outer_prediction_labels_read": receipt.get("outer_prediction_labels_read"),
    }
    if set(ancestry["train_scenarios"]) & set(ancestry["held_scenarios"]):
        raise RuntimeError("direct inner ancestry scenario overlap")
    if ancestry["outer_prediction_labels_read"] != 0:
        raise RuntimeError("inner ARFTR ancestor read prediction labels")
    return train_anchor, held_anchor, ancestry


def outer_anchors(d: dict[str, np.ndarray], outer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    plan = source_posture.fixed_recipe_plan(
        d["labels"], d["scenarios"], d["folds"], d["sample_ids"], enforce_canonical_counts=True
    )
    train, held, train_anchor, ancestry = source_posture._fold_priors(router_base.data(), plan, outer)
    outer_anchor = d["anchor"][held]
    # The source/posture helper reads the exact retained per-fold probabilities.
    source_train, source_held, source_train_anchor, _ = source_posture._fold_priors(router_base.data(), plan, outer)
    if not np.array_equal(train, source_train) or not np.array_equal(held, source_held) or not np.array_equal(train_anchor, source_train_anchor):
        raise RuntimeError("outer ancestry helper is nondeterministic")
    with np.load(ROOT / f".runs/research_20260912/arftr_v1/fold-{outer}/predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["held_rows"], held):
            raise RuntimeError("retained ARFTR fold identity changed")
        retained_fold = saved["seed_probabilities"][5].mean(0)
    if not np.array_equal(retained_fold, outer_anchor):
        raise RuntimeError("rebuilt outer anchor does not exactly match retained ARFTR array")
    return train, held, train_anchor, {"ancestry": ancestry, "anchor_sha256": hashlib.sha256(outer_anchor.tobytes()).hexdigest()}


def prepare(run: Path) -> dict[str, Any]:
    d = data()
    protocol = read_json(PROTOCOL)
    if tuple(protocol["arms"]) != ARMS or protocol["primary"] != "paired_spatial_temporal":
        raise RuntimeError("PDI arm contract changed")
    if protocol["training"]["optimizer_updates"] != 256 or protocol["representation"]["factor_bound"] != 0.5:
        raise RuntimeError("PDI training contract changed")
    dependencies = [
        PROTOCOL, REVIEW_PROTOCOL, Path(__file__), ROOT / "src/hac/paired_detail_innovation.py",
        ROOT / "src/hac/intervention_utility.py", ROOT / "tests/test_paired_detail_innovation.py",
        ROOT / "tests/test_intervention_utility.py", ROOT / "tests/test_nested_producer_ancestry.py",
        ROOT / "experiments/audit_okutama_paired_detail_innovation.py",
        ARFTR, INNER_ANCESTORS / "execution_lock.json",
        ROOT / ".runs/research_20260916/source_posture_failure_router_v1/inner_cache_plan.json",
        ROOT / ".runs/research_20260913/generalized_arftr_ancestors_v2/independent_audit.json",
        ROOT / ".runs/research_20260913/generalized_arftr_parity_v1/audit.json",
        fine_v2.LOCK_PATH,
    ]
    populations = {}
    ancestor_receipts = []
    for outer in range(5):
        mapping = scenario_assignment(d, outer)
        if mapping != read_json(router_base.PLAN)["outer_plans"][outer]["scenario_assignment"]:
            raise RuntimeError("PDI split differs from completed inner ancestor plan")
        for inner in range(5):
            train, held = inner_rows(d, outer, inner)
            receipt, _arrays = _f_artifact(d, outer, inner)
            receipt_path = INNER_ANCESTORS / "populations" / receipt["population_id"] / "receipt.json"
            ancestor_receipts.append(receipt_path)
            if not np.array_equal(np.unique(d["labels"][train]), np.arange(3)):
                raise RuntimeError("PDI inner training population lacks a class")
            populations[f"o{outer}_i{inner}"] = {
                "train_rows": len(train), "held_rows": len(held),
                "train_ids_sha256": canonical_hash(d["sample_ids"][train].tolist()),
                "held_ids_sha256": canonical_hash(d["sample_ids"][held].tolist()),
                "train_scenarios": sorted(set(d["scenarios"][train].tolist())),
                "held_scenarios": sorted(set(d["scenarios"][held].tolist())),
                "ancestor_population_id": receipt["population_id"],
            }
    dependencies.extend(ancestor_receipts)
    torch.manual_seed(42)
    parameter_count = PairedDetailInnovation().trainable_parameters
    lock = {
        "status": LOCK_STATUS,
        "rows": len(d["labels"]), "arms": list(ARMS), "outer_folds": 5,
        "inner_populations": populations, "scenario_assignments": {
            str(outer): scenario_assignment(d, outer) for outer in range(5)
        },
        "sample_ids_sha256": canonical_hash(d["sample_ids"].tolist()),
        "parameter_count": parameter_count, "device": "cuda",
        "task_fits_at_lock": 0, "outer_labels_read_at_lock": 0,
        "dependencies": [record(path) for path in sorted(set(dependencies), key=str)],
    }
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "execution_lock.json", lock)
    return lock


def validate_lock(run: Path) -> dict[str, Any]:
    lock = read_json(run / "execution_lock.json")
    if lock.get("status") != LOCK_STATUS or lock.get("task_fits_at_lock") != 0:
        raise RuntimeError("PDI execution lock invalid")
    for item in lock["dependencies"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"locked PDI dependency changed: {item['path']}")
    return lock


def _stats(d: dict, rows: np.ndarray) -> tuple[np.ndarray, ...]:
    return fine_base.training_stats(d, rows)


def _batch(
    d: dict, rows: np.ndarray, stats: tuple[np.ndarray, ...], arm: str, anchors: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    mean, std, qmean, qstd, _weights = stats
    input_arm = "coarse_control" if arm == "coarse_residual" else "fine_both"
    fine, times, quality, coarse = fine_base.batch_inputs(
        d, rows, mean, std, qmean, qstd, input_arm, device
    )
    return fine, coarse, times, quality, torch.from_numpy(np.asarray(anchors, dtype=np.float32)).to(device)


def _seed(sample_ids: np.ndarray) -> int:
    return 42 + int(canonical_hash(sample_ids.tolist())[:8], 16) % 1_000_000


def fit_predict(
    d: dict,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
    fit_anchor: np.ndarray,
    held_anchor: np.ndarray,
    arm: str,
    directory: Path,
    lock_sha256: str,
    device: str,
    ancestry: dict,
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("execution_lock_sha256") != lock_sha256:
            raise RuntimeError("completed PDI fit belongs to another lock")
        return receipt
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"partial PDI fit retained; use a fresh directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    stats = _stats(d, fit_rows)
    seed = _seed(d["sample_ids"][fit_rows])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PairedDetailInnovation().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = torch.from_numpy(stats[-1]).to(device)
    rng = np.random.default_rng(seed)
    checkpoints = {
        0: {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        64: None, 128: None, 256: None,
    }
    losses = []
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    started = time.perf_counter()
    model.train()
    for update in range(1, 257):
        if cursor + 128 > len(order):
            order = rng.permutation(len(fit_rows))
            cursor = 0
        local = order[cursor : min(cursor + 128, len(order))]
        cursor += len(local)
        rows = fit_rows[local]
        batch = _batch(d, rows, stats, arm, fit_anchor[local], device)
        labels = torch.from_numpy(d["labels"][rows]).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(*batch, arm=arm)
        loss, pieces = pdi_loss(output, batch[-1], labels, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite PDI loss")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        if update == 1 or update % 16 == 0:
            losses.append({"update": update, "loss": float(loss.detach()), "gradient_norm": gradient,
                           **{name: float(value.detach()) for name, value in pieces.items()}})
        if update in checkpoints:
            checkpoints[update] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    elapsed = time.perf_counter() - started
    torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed}, directory / "checkpoint.pt")
    torch.save({"updates": checkpoints, "arm": arm, "seed": seed}, directory / "training_checkpoints.pt")
    np.savez(directory / "normalization.npz", mean=stats[0], std=stats[1], qmean=stats[2], qstd=stats[3], class_weights=stats[4])
    predictions, actions, corrections, scaffold = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(held_rows), 128):
            subset = held_rows[start : start + 128]
            batch = _batch(d, subset, stats, arm, held_anchor[start : start + len(subset)], device)
            output = model(*batch, arm=arm)
            predictions.append(output["probabilities"].cpu().numpy())
            actions.append(output["actions"].cpu().numpy())
            corrections.append(output["correction"].cpu().numpy())
            scaffold.append(output["coarse_scaffold_norms"].cpu().numpy())
    write_npz(
        directory / "predictions.npz", sample_ids=d["sample_ids"][held_rows], held_rows=held_rows,
        anchor_probabilities=np.asarray(held_anchor, dtype=np.float64),
        probabilities=np.concatenate(predictions).astype(np.float64),
        actions=np.concatenate(actions).astype(np.float64),
        correction=np.concatenate(corrections).astype(np.float64),
        scaffold_norms=np.concatenate(scaffold).astype(np.float64),
    )
    receipt = {
        "status": "PDI_PRODUCER_FIT_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_sha256, "arm": arm, "seed": seed,
        "fit_rows": len(fit_rows), "held_rows": len(held_rows), "updates": 256,
        "elapsed_seconds": elapsed, "outer_held_labels_read": 0,
        "fit_ids_sha256": canonical_hash(d["sample_ids"][fit_rows].tolist()),
        "held_ids_sha256": canonical_hash(d["sample_ids"][held_rows].tolist()),
        "checkpoint_sha256": file_sha256(directory / "checkpoint.pt"),
        "training_checkpoints_sha256": file_sha256(directory / "training_checkpoints.pt"),
        "normalization_sha256": file_sha256(directory / "normalization.npz"),
        "predictions_sha256": file_sha256(directory / "predictions.npz"),
        "loss_curve": losses, "ancestry": ancestry,
    }
    write_json(receipt_path, receipt)
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return receipt


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _capacity(labels: np.ndarray, anchor: np.ndarray, actions: np.ndarray, groups: np.ndarray) -> dict:
    anchor_class = anchor.argmax(1)
    action_class = actions.argmax(2)
    wrong = anchor_class != labels
    reachable = wrong & (action_class == labels[:, None]).any(1)
    oracle = anchor.copy()
    for row in np.flatnonzero(reachable):
        choice = np.flatnonzero(action_class[row] == labels[row])[0]
        oracle[row] = actions[row, choice]
    base_f1 = f1_score(labels, anchor_class, labels=[0, 1, 2], average="macro", zero_division=0)
    oracle_f1 = f1_score(labels, oracle.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
    unique = np.unique(groups)
    per_group = {str(group): int(reachable[groups == group].sum()) for group in unique}
    required = math.ceil(50 * len(labels) / 4977)
    return {
        "diagnostic_only_training_rows": True, "reachable_errors": int(reachable.sum()),
        "required_reachable_errors": required, "macro_f1_headroom": float(oracle_f1 - base_f1),
        "per_inner_group": per_group,
        "passes": bool(reachable.sum() >= required and oracle_f1 - base_f1 >= 0.01 and all(value >= 1 for value in per_group.values())),
    }


def _save_verifier(path: Path, verifier: InterventionVerifier, names: list[str], receipt: dict) -> None:
    arrays = {"scaler_mean": verifier.scaler.mean_, "scaler_scale": verifier.scaler.scale_,
              "feature_names": np.asarray(names), "receipt_json": np.asarray(json.dumps(receipt, sort_keys=True))}
    for index, model in enumerate(verifier.class_models):
        arrays[f"class_coef_{index}"] = model.coef_; arrays[f"class_intercept_{index}"] = np.asarray([model.intercept_])
    for index, model in enumerate(verifier.nll_models):
        arrays[f"nll_coef_{index}"] = model.coef_; arrays[f"nll_intercept_{index}"] = np.asarray([model.intercept_])
    write_npz(path, **arrays)


def run_arm_fold(run: Path, d: dict, arm: str, outer: int, device: str) -> dict:
    validate_lock(run)
    fold_dir = run / arm / f"fold-{outer}"
    fold_receipt = fold_dir / "receipt.json"
    if fold_receipt.is_file():
        return read_json(fold_receipt)
    lock_hash = file_sha256(run / "execution_lock.json")
    outer_train = np.flatnonzero(d["folds"] != outer)
    inner_anchor = np.full((len(d["labels"]), 3), np.nan)
    inner_actions = np.full((len(d["labels"]), 4, 3), np.nan)
    inner_correction = np.full((len(d["labels"]), 2), np.nan)
    inner_scaffold = np.full((len(d["labels"]), 4), np.nan)
    inner_group = np.full(len(d["labels"]), -1, dtype=np.int64)
    inner_receipts = []
    for inner in range(5):
        train, held = inner_rows(d, outer, inner)
        train_anchor, held_anchor, ancestry = inner_anchors(d, outer, inner)
        directory = fold_dir / f"inner-{inner}"
        receipt = fit_predict(d, train, held, train_anchor, held_anchor, arm, directory, lock_hash, device, ancestry)
        saved = _load_predictions(directory / "predictions.npz")
        if not np.array_equal(saved["held_rows"], held) or not np.array_equal(saved["anchor_probabilities"], held_anchor):
            raise RuntimeError("inner PDI identity/anchor mismatch")
        inner_anchor[held] = held_anchor
        inner_actions[held] = saved["actions"]
        inner_correction[held] = saved["correction"]
        inner_scaffold[held] = saved["scaffold_norms"]
        inner_group[held] = inner
        inner_receipts.append(record(directory / "receipt.json"))
    if not np.isfinite(inner_actions[outer_train]).all() or np.any(inner_group[outer_train] < 0):
        raise RuntimeError("inner PDI coverage incomplete")
    capacity = _capacity(d["labels"][outer_train], inner_anchor[outer_train], inner_actions[outer_train], inner_group[outer_train])
    write_json(fold_dir / "inner_capacity_DIAGNOSTIC_ONLY.json", capacity)
    if not capacity["passes"]:
        receipt = {
            "status": "PDI_ARM_FOLD_STOPPED_INNER_CAPACITY_NO_GO", "arm": arm, "outer_fold": outer,
            "outer_held_labels_read": 0, "capacity": capacity, "inner_receipts": inner_receipts,
        }
        write_json(fold_receipt, receipt)
        return receipt
    quality = d["quality"][outer_train].astype(np.float64)
    features, names = verifier_features(
        inner_anchor[outer_train], inner_actions[outer_train], inner_correction[outer_train],
        inner_scaffold[outer_train], quality,
    )
    verifier, verifier_receipt = fit_intervention_verifier(
        features, inner_anchor[outer_train], inner_actions[outer_train], d["labels"][outer_train]
    )
    verifier.feature_names = names
    write_npz(
        fold_dir / "inner_oof_verifier_inputs.npz", rows=outer_train,
        sample_ids=d["sample_ids"][outer_train], anchor_probabilities=inner_anchor[outer_train],
        actions=inner_actions[outer_train], correction=inner_correction[outer_train],
        scaffold_norms=inner_scaffold[outer_train], quality=d["quality"][outer_train],
        verifier_features=features, inner_group=inner_group[outer_train],
    )
    train, held, train_anchor, outer_ancestry = outer_anchors(d, outer)
    if not np.array_equal(train, outer_train):
        raise RuntimeError("final PDI training population changed")
    final_dir = fold_dir / "final"
    final_receipt = fit_predict(
        d, train, held, train_anchor, d["anchor"][held], arm, final_dir,
        lock_hash, device, outer_ancestry,
    )
    saved = _load_predictions(final_dir / "predictions.npz")
    outer_features, outer_names = verifier_features(
        saved["anchor_probabilities"], saved["actions"], saved["correction"],
        saved["scaffold_norms"], d["quality"][held].astype(np.float64),
    )
    if names != outer_names:
        raise RuntimeError("PDI verifier schema changed")
    routed, choices = verifier.apply(saved["anchor_probabilities"], saved["actions"], outer_features)
    write_npz(
        fold_dir / "predictions.npz", sample_ids=d["sample_ids"][held], held_rows=held,
        anchor_probabilities=saved["anchor_probabilities"], raw_probabilities=saved["probabilities"],
        actions=saved["actions"], correction=saved["correction"], scaffold_norms=saved["scaffold_norms"],
        verifier_features=outer_features, routed_probabilities=routed, choices=choices,
    )
    _save_verifier(fold_dir / "verifier.npz", verifier, names, verifier_receipt)
    receipt = {
        "status": "PDI_ARM_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED", "arm": arm, "outer_fold": outer,
        "outer_held_labels_read": 0, "capacity": capacity, "inner_receipts": inner_receipts,
        "final_receipt": record(final_dir / "receipt.json"), "verifier_receipt": verifier_receipt,
        "inner_oof_verifier_inputs_sha256": file_sha256(fold_dir / "inner_oof_verifier_inputs.npz"),
        "verifier_sha256": file_sha256(fold_dir / "verifier.npz"),
        "predictions_sha256": file_sha256(fold_dir / "predictions.npz"),
    }
    write_json(fold_receipt, receipt)
    return receipt


def queue(run: Path, device: str, *, maximum_fits: int | None = None) -> dict:
    validate_lock(run)
    d = data()
    started = time.perf_counter()
    completed_before = sum((run / arm / f"fold-{outer}" / "receipt.json").is_file() for arm in ARMS for outer in range(5))
    newly_completed = 0
    for arm in ARMS:
        for outer in range(5):
            path = run / arm / f"fold-{outer}" / "receipt.json"
            if path.is_file():
                continue
            receipt = run_arm_fold(run, d, arm, outer, device)
            newly_completed += 1
            print(json.dumps({"event": "pdi_arm_fold_complete", "arm": arm, "outer": outer,
                              "status": receipt["status"], "elapsed_queue_seconds": time.perf_counter() - started}), flush=True)
            if maximum_fits is not None and newly_completed >= maximum_fits:
                break
        if maximum_fits is not None and newly_completed >= maximum_fits:
            break
    state = {
        "status": "PDI_QUEUE_COMPLETE" if completed_before + newly_completed == 15 else "PDI_QUEUE_PARTIAL",
        "arm_folds_complete": completed_before + newly_completed, "arm_folds_total": 15,
        "new_arm_folds": newly_completed, "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run / f"queue_receipt_{completed_before + newly_completed:02d}.json", state)
    print(json.dumps(state, indent=2), flush=True)
    return state


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(1)
    onehot = np.eye(3)[labels]
    precision, recall, f1, support = precision_recall_fscore_support(labels, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "rows": len(labels), "macro_f1": float(f1.mean()), "accuracy": float((pred == labels).mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier_sum": float(np.square(probabilities - onehot).sum(1).mean()), "errors": int((pred != labels).sum()),
        "per_class_f1": f1.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
        "support": support.tolist(), "confusion": confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist(),
    }


def _bootstrap(d: dict, candidate: np.ndarray, anchor: np.ndarray) -> dict:
    groups = np.unique(d["scenarios"])
    cm_c, cm_a = [], []
    for group in groups:
        rows = d["scenarios"] == group
        cm_c.append(confusion_matrix(d["labels"][rows], candidate[rows].argmax(1), labels=[0, 1, 2]))
        cm_a.append(confusion_matrix(d["labels"][rows], anchor[rows].argmax(1), labels=[0, 1, 2]))
    cm_c, cm_a = np.asarray(cm_c), np.asarray(cm_a)
    rng = np.random.default_rng(20260919)
    draws = rng.integers(0, len(groups), size=(10000, len(groups)))
    def f1(cm):
        diagonal = np.diagonal(cm, axis1=-2, axis2=-1); denominator = cm.sum(-1) + cm.sum(-2)
        return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0).mean(-1)
    delta = f1(cm_c[draws].sum(1)) - f1(cm_a[draws].sum(1))
    return {"draws": 10000, "seed": 20260919, "interval": np.quantile(delta, [.025, .975]).tolist(), "mean": float(delta.mean())}


def summarize(run: Path) -> dict:
    validate_lock(run)
    audit_path = run / "independent_audit.json"
    if not audit_path.is_file() or read_json(audit_path).get("status") != "PDI_INDEPENDENT_FULL_PRODUCER_AND_VERIFIER_REPLAY_PASS":
        raise RuntimeError("complete independent PDI replay must pass before metrics release")
    d = data()
    results = {"arftr": d["anchor"]}
    arm_status = {}
    for arm in ARMS:
        output = np.full_like(d["anchor"], np.nan)
        statuses = []
        for outer in range(5):
            receipt_path = run / arm / f"fold-{outer}" / "receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError("PDI queue is incomplete")
            receipt = read_json(receipt_path); statuses.append(receipt["status"])
            if receipt["status"] == "PDI_ARM_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED":
                saved = _load_predictions(run / arm / f"fold-{outer}" / "predictions.npz")
                output[saved["held_rows"]] = saved["routed_probabilities"]
        arm_status[arm] = statuses
        if np.isfinite(output).all():
            results[arm] = output
    scores = {name: metrics(d["labels"], probabilities) for name, probabilities in results.items()}
    details = {}
    for arm, probabilities in results.items():
        if arm == "arftr":
            continue
        anchor_correct = d["anchor"].argmax(1) == d["labels"]
        correct = probabilities.argmax(1) == d["labels"]
        nets = [int((correct[d["folds"] == fold] & ~anchor_correct[d["folds"] == fold]).sum()
                    - (~correct[d["folds"] == fold] & anchor_correct[d["folds"] == fold]).sum()) for fold in range(5)]
        details[arm] = {
            "rescues": int((correct & ~anchor_correct).sum()), "harms": int((~correct & anchor_correct).sum()),
            "net": int(correct.sum() - anchor_correct.sum()), "per_fold_net": nets,
            "bootstrap": _bootstrap(d, probabilities, d["anchor"]),
        }
    primary = "paired_spatial_temporal"
    gates = None
    if primary in scores and "coarse_residual" in scores:
        candidate, anchor = scores[primary], scores["arftr"]
        info = details[primary]
        gates = {
            "macro_f1": candidate["macro_f1"] >= 0.8588364807373623,
            "delta_vs_coarse": candidate["macro_f1"] - scores["coarse_residual"]["macro_f1"] >= .0025,
            "net": info["net"] >= 25, "positive_each_fold": all(value > 0 for value in info["per_fold_net"]),
            "bootstrap": info["bootstrap"]["interval"][0] > 0,
            "nll": candidate["nll"] <= anchor["nll"] + 1e-6,
            "brier": candidate["brier_sum"] <= anchor["brier_sum"] + 1e-6,
            "class_f1": min(np.asarray(candidate["per_class_f1"]) - np.asarray(anchor["per_class_f1"])) >= -.005,
        }
    summary = {
        "status": "PDI_SUMMARY_COMPLETE" if gates is not None else "PDI_INNER_CAPACITY_NO_GO_INCOMPLETE_ARCHITECTURE_TEST",
        "scores": scores, "details": details, "arm_fold_status": arm_status, "promotion_gates": gates,
        "promoted": bool(gates is not None and all(gates.values())), "retained_arftr_changed": False,
    }
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def preflight(run: Path, device: str) -> dict:
    lock = validate_lock(run); d = data()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Validate all 25 anchors and recursive direct scenario exclusions without fitting.
    ancestries = []
    for outer in range(5):
        train, held, train_anchor, _ = outer_anchors(d, outer)
        if not np.isfinite(d["anchor"][held]).all() or not np.isfinite(train_anchor).all():
            raise RuntimeError("outer ARFTR parity failed")
        for inner in range(5):
            fit, inner_held = inner_rows(d, outer, inner)
            fit_anchor, held_anchor, ancestry = inner_anchors(d, outer, inner)
            if not np.isfinite(fit_anchor).all() or not np.isfinite(held_anchor).all():
                raise RuntimeError("inner anchor contains nonfinite values")
            if set(d["scenarios"][fit]) & set(d["scenarios"][inner_held]):
                raise RuntimeError("inner split scenario overlap")
            ancestries.append(ancestry)
    result = {
        "status": "PDI_PREFLIGHT_PASS_QUEUE_AUTHORIZED", "rows": len(d["labels"]),
        "inner_populations_checked": len(ancestries), "classifier_fits": 0,
        "cuda_available": torch.cuda.is_available(), "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameter_count": lock["parameter_count"], "outer_labels_read": 0,
    }
    write_json(run / "preflight.json", result)
    print(json.dumps(result, indent=2))
    return result


def profile_primary_fit(run: Path, device: str) -> dict:
    """Execute and retain the first real inner primary fit for honest timing."""
    validate_lock(run)
    if not (run / "preflight.json").is_file() or read_json(run / "preflight.json").get("status") != "PDI_PREFLIGHT_PASS_QUEUE_AUTHORIZED":
        raise RuntimeError("PDI preflight must pass before the first fit")
    d = data(); outer = inner = 0; arm = "paired_spatial_temporal"
    train, held = inner_rows(d, outer, inner)
    train_anchor, held_anchor, ancestry = inner_anchors(d, outer, inner)
    directory = run / arm / f"fold-{outer}" / f"inner-{inner}"
    existed = (directory / "receipt.json").is_file()
    receipt = fit_predict(
        d, train, held, train_anchor, held_anchor, arm, directory,
        file_sha256(run / "execution_lock.json"), device, ancestry,
    )
    seconds = float(receipt["elapsed_seconds"])
    result = {
        "status": "PDI_FULL_POPULATION_PROFILE_COMPLETE",
        "scientific_fit_retained_for_queue": True,
        "fit_preexisting": existed,
        "arm": arm, "outer": outer, "inner": inner,
        "fit_rows": len(train), "held_rows": len(held), "updates": 256,
        "measured_fit_seconds": seconds,
        "projected_90_residual_fit_seconds": 90 * seconds,
        "projected_90_residual_fit_hours": 90 * seconds / 3600,
        "qualification": "Linear single-fit projection; arm/fold sizes and final producers differ; ARFTR ancestors already exist.",
        "receipt_sha256": file_sha256(directory / "receipt.json"),
    }
    write_json(run / "timing_profile.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", required=True, choices=("prepare", "preflight", "profile", "queue", "summarize"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-arm-folds", type=int)
    args = parser.parse_args()
    run = args.run.resolve(); run.relative_to(ROOT.resolve())
    if args.stage == "prepare":
        print(json.dumps(prepare(run), indent=2))
    elif args.stage == "preflight":
        preflight(run, args.device)
    elif args.stage == "profile":
        profile_primary_fit(run, args.device)
    elif args.stage == "queue":
        queue(run, args.device, maximum_fits=args.maximum_arm_folds)
    else:
        summarize(run)


if __name__ == "__main__":
    main()
