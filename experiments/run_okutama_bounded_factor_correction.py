"""Run the frozen bounded factor-correction experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_arftr as arftr_run
import run_okutama_frame_supervision as fsar
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors
from hac.bounded_factor_correction import (
    BoundedFactorCorrection,
    apply_numpy_correction,
    correction_loss,
)
from hac.cached_frame_supervision import validate_cached_frame_supervision_receipt
from hac.frame_supervised_residual import class_weights, seed_frame_training
from hac.matr_artifacts import load_cached_study_data, selected_fold_artifacts, selected_input_paths
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
ARFTR_RUN = ROOT / ".runs/research_20260912/arftr_v1"
FRAME_DATA = ROOT / ".runs/research_20260912/cached_frame_supervision_v2/data"
PROTOCOL = ROOT / "experiments/okutama_bounded_factor_correction_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/bounded_factor_correction_v1"
SEEDS = (42, 43, 44)
ARMS = (
    "b0_exact_arftr",
    "b1_bounded_separate_center_only",
    "b2_bounded_separate_all_frames",
    "b3_bounded_shared_all_frames",
    "b4_unbounded_separate_all_frames",
)
TRAINED_ARMS = ARMS[1:]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path.relative_to(ROOT.resolve())).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    training, architecture, statistics = (
        protocol["training"],
        protocol["architecture"],
        protocol["statistics"],
    )
    if (
        tuple(protocol["arms"]) != ARMS
        or protocol["primary_arm"] != ARMS[2]
        or tuple(training["outer_folds"]) != tuple(range(5))
        or tuple(training["outer_seeds"]) != SEEDS
        or training["model_fits"] != 60
        or training["epochs"] != 8
        or training["batch_size"] != 2048
        or architecture["posture_bound"] != 0.5
        or architecture["motion_bound"] != 0.5
        or statistics["scenario_bootstrap_resamples"] != 10000
        or statistics["exact_scenario_swaps"] != 2048
        or statistics["arftr_comparison_seed_rule"] != "seed_plus_arm_index"
        or statistics["center_control_seed"] != 20260922
        or statistics["required_diagnostics"]
        != [
            "per_class_precision_recall",
            "known_pure_support_rescues_harms",
            "known_mixed_support_rescues_harms",
            "unknown_support_rescues_harms",
        ]
    ):
        raise RuntimeError("Frozen bounded-correction protocol changed")


def model_spec(arm: str) -> tuple[bool, bool, bool]:
    if arm not in TRAINED_ARMS:
        raise ValueError("Unknown trained arm")
    center_only = arm == ARMS[1]
    shared = arm == ARMS[3]
    bounded = arm != ARMS[4]
    return center_only, shared, bounded


def make_model(protocol: dict[str, Any], arm: str) -> BoundedFactorCorrection:
    _, shared, bounded = model_spec(arm)
    architecture = protocol["architecture"]
    return BoundedFactorCorrection(
        input_dim=architecture["input_dim"],
        width=architecture["width"],
        dropout=architecture["dropout"],
        posture_bound=architecture["posture_bound"],
        motion_bound=architecture["motion_bound"],
        shared_output=shared,
        bounded=bounded,
        parameter_limit=architecture["parameter_limit"],
    )


def arftr_anchors(
    data: dict[str, np.ndarray], fold: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cached = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    train, held = cached.train_rows, cached.held_rows
    cache = arftr_run._p6_cache(data)
    p6_all, ancestry = cache.meta_probabilities(train)
    p6_inner = p6_all[train]
    p6_outer = cache.held_predictions(train, held)
    checkpoint_path = ARFTR_RUN / f"fold-{fold}/checkpoint.npz"
    prediction_path = ARFTR_RUN / f"fold-{fold}/predictions.npz"
    receipt_path = ARFTR_RUN / f"fold-{fold}/receipt.json"
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "ARFTR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED"
        or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or receipt.get("predictions_sha256") != file_sha256(prediction_path)
        or receipt.get("p6_inner_ancestry") != ancestry
    ):
        raise RuntimeError("Frozen ARFTR fold ancestry changed")
    with np.load(checkpoint_path, allow_pickle=False) as saved:
        parameters = ARFTRParameters(*saved["parameters"].tolist())
    inner_neighbors = exact_track_neighbors(data, train)
    outer_neighbors = exact_track_neighbors(data, held)
    inner = apply_arftr(
        cached.m4_inner["probabilities"],
        p6_inner,
        cached.a3_inner["probabilities"],
        inner_neighbors,
        parameters,
    )
    with np.load(prediction_path, allow_pickle=False) as saved:
        if (
            not np.array_equal(saved["sample_ids"], data["sample_ids"][held])
            or tuple(saved["arms"].tolist()) != arftr_run.ARMS
            or tuple(saved["seeds"].tolist()) != SEEDS
        ):
            raise RuntimeError("Frozen ARFTR prediction identity changed")
        outer = saved["seed_probabilities"][5].copy()
    for seed_index in range(3):
        replay = apply_arftr(
            cached.m4_outer["probabilities"][seed_index],
            p6_outer,
            cached.a3_outer["probabilities"][seed_index],
            outer_neighbors,
            parameters,
        )
        if not np.array_equal(replay, outer[seed_index]):
            raise RuntimeError("Frozen ARFTR outer replay changed")
    return (
        inner,
        outer,
        {
            "parameters": parameters.as_array().tolist(),
            "p6_inner_ancestry": ancestry,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "predictions_sha256": file_sha256(prediction_path),
        },
    )


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], str]:
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    data = load_cached_study_data(SOURCE)
    frames, _ = validate_cached_frame_supervision_receipt(ROOT, FRAME_DATA)
    arftr_lock = ARFTR_RUN / "execution_lock.json"
    arftr_run.validate_lock(ARFTR_RUN, data, file_sha256(arftr_lock))
    paths = set(selected_input_paths(ROOT, SOURCE, SEAR, data, seeds=SEEDS))
    paths.update(
        {
            PROTOCOL,
            Path(__file__),
            ROOT / "experiments/audit_okutama_bounded_factor_correction.py",
            ROOT / "experiments/run_okutama_arftr.py",
            ROOT / "experiments/run_okutama_frame_supervision.py",
            ROOT / "src/hac/bounded_factor_correction.py",
            ROOT / "src/hac/arftr.py",
            ROOT / "src/hac/cached_frame_supervision.py",
            ROOT / "src/hac/frame_supervised_residual.py",
            FRAME_DATA / "frame_supervision.npz",
            FRAME_DATA / "receipt.json",
            arftr_lock,
            ARFTR_RUN / "results/v0001/summary.json",
            ARFTR_RUN / "results/v0001/oof_probabilities.npz",
        }
    )
    for fold in range(5):
        arftr_anchors(data, fold)
        for name in ("checkpoint.npz", "predictions.npz", "receipt.json"):
            paths.add(ARFTR_RUN / f"fold-{fold}" / name)
    for key in ("short_feature", "long_feature"):
        paths.add(ROOT / fsar.SOURCE_PATHS[key])
    if any(not path.is_file() for path in paths):
        raise RuntimeError("A bounded-correction locked input is missing")
    lock = {
        "status": "BOUNDED_FACTOR_CORRECTION_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "rows": len(data["labels"]),
        "model_fits": 60,
        "outer_held_labels_read": 0,
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "protocol_sha256": file_sha256(PROTOCOL),
        "input_files": [file_record(path) for path in sorted(paths, key=str)],
    }
    run.mkdir(parents=True, exist_ok=True)
    immutable_json(run / "execution_lock.json", lock)
    lock_hash = file_sha256(run / "execution_lock.json")
    validate_lock(run, data, lock_hash)
    return protocol, data, frames, lock_hash


def validate_lock(run: Path, data: dict[str, np.ndarray], lock_hash: str) -> dict[str, Any]:
    path = run / "execution_lock.json"
    if file_sha256(path) != lock_hash:
        raise RuntimeError("Bounded-correction execution lock changed")
    lock = read_json(path)
    if (
        lock.get("status") != "BOUNDED_FACTOR_CORRECTION_EXECUTION_LOCKED"
        or lock.get("sample_ids_sha256") != canonical_hash(data["sample_ids"].tolist())
        or lock.get("model_fits") != 60
    ):
        raise RuntimeError("Bounded-correction execution identity changed")
    for record in lock["input_files"]:
        input_path = (ROOT / record["path"]).resolve()
        input_path.relative_to(ROOT.resolve())
        if (
            not input_path.is_file()
            or input_path.stat().st_size != record["size_bytes"]
            or file_sha256(input_path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked input changed: {record['path']}")
    return lock


def fit_paths(run: Path, arm: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    directory = run / "models" / arm / f"fold-{fold}" / f"seed-{seed}"
    return directory / "predictions.npz", directory / "checkpoint.npz", directory / "receipt.json"


def checkpoint(model: BoundedFactorCorrection) -> dict[str, np.ndarray]:
    return {
        name.replace(".", "__"): value.detach().cpu().numpy()
        for name, value in model.state_dict().items()
    }


def resource_gate(protocol: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_frame_training(20260912)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = make_model(protocol, ARMS[2]).to(device)
    batch = 2048
    features = torch.randn(batch, 768, device=device)
    anchor = torch.softmax(torch.randn(batch, 3, device=device), 1)
    labels = torch.arange(batch, device=device) % 3
    loss, _ = correction_loss(
        model,
        center_features=features,
        anchor_probabilities=anchor,
        center_labels=labels,
        auxiliary_features=features,
        auxiliary_labels=labels,
        auxiliary_weights=torch.ones(batch, device=device),
        class_weight=torch.ones(3, device=device),
        center_loss_weight=1.0,
        auxiliary_loss_weight=0.5,
        anchor_kl_weight=0.1,
    )
    loss.backward()
    if not torch.isfinite(loss) or any(
        value.grad is None or not torch.isfinite(value.grad).all() for value in model.parameters()
    ):
        raise RuntimeError("Resource gate found a missing or non-finite gradient")
    return {
        "status": "BOUNDED_FACTOR_CORRECTION_RESOURCE_GATE_PASSED",
        "execution_lock_sha256": lock_hash,
        "labels_read": 0,
        "optimizer_steps": 0,
        "parameters": model.trainable_parameters,
        "batch_size": batch,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device": torch.cuda.get_device_name(device),
    }


def validate_probability(values: np.ndarray, rows: int) -> None:
    if (
        values.shape != (rows, 3)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise RuntimeError("Malformed correction probabilities")


def fit_one(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
    store: fsar.FrameFeatureStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    validate_lock(run, data, lock_hash)
    center_only, _, _ = model_spec(arm)
    selected = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    train, held = selected.train_rows, selected.held_rows
    inner_anchor, outer_anchors, ancestry = arftr_anchors(data, fold)
    seed_index = SEEDS.index(seed)
    outer_anchor = outer_anchors[seed_index]
    prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "BOUNDED_FACTOR_CORRECTION_FIT_COMPLETE"
            or receipt.get("execution_lock_sha256") != lock_hash
            or receipt.get("arm") != arm
            or receipt.get("fold") != fold
            or receipt.get("seed") != seed
            or receipt.get("predictions_sha256") != file_sha256(prediction_path)
            or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError("Completed correction fit changed")
        return receipt
    if receipt_path.parent.exists():
        raise RuntimeError("Partial correction fit retained; use a fresh run path")
    staging = receipt_path.parent.with_name(receipt_path.parent.name + ".incomplete")
    if staging.exists():
        raise RuntimeError("Incomplete correction fit retained; use a fresh run path")
    staging.mkdir(parents=True)
    candidate = fsar._candidate_observations(frames, train)
    config = protocol["training"]
    steps = math.ceil(len(candidate[0]) / config["batch_size"])
    center_features, held_features = store.centers(train), store.centers(held)
    device = torch.device("cuda")
    seed_frame_training(seed)
    model = make_model(protocol, arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    weights = torch.as_tensor(class_weights(data["labels"], train), device=device)
    history = []
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        rng = np.random.default_rng(seed * 1000 + epoch)
        total = steps * config["batch_size"]
        center_order = fsar._cycled_permutations(len(train), total, rng)
        auxiliary_order = fsar._cycled_permutations(
            len(train) if center_only else len(candidate[0]), total, rng
        )
        sums = {"loss": 0.0, "center": 0.0, "auxiliary": 0.0, "anchor_kl": 0.0}
        model.train()
        for step in range(steps):
            left, right = step * config["batch_size"], (step + 1) * config["batch_size"]
            center_position = center_order[left:right]
            aux_position = auxiliary_order[left:right]
            if center_only:
                auxiliary_x = center_features[aux_position]
                auxiliary_y = data["labels"][train[aux_position]]
                auxiliary_w = np.ones(len(aux_position), dtype=np.float32)
            else:
                auxiliary_x = store.observations(
                    candidate[0][aux_position],
                    candidate[1][aux_position],
                    candidate[2][aux_position],
                )
                auxiliary_y, auxiliary_w = candidate[3][aux_position], candidate[4][aux_position]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = correction_loss(
                model,
                center_features=torch.as_tensor(center_features[center_position], device=device),
                anchor_probabilities=torch.as_tensor(
                    inner_anchor[center_position], dtype=torch.float32, device=device
                ),
                center_labels=torch.as_tensor(
                    data["labels"][train[center_position]], device=device
                ),
                auxiliary_features=torch.as_tensor(auxiliary_x, device=device),
                auxiliary_labels=torch.as_tensor(auxiliary_y, device=device),
                auxiliary_weights=torch.as_tensor(auxiliary_w, device=device),
                class_weight=weights,
                center_loss_weight=config["center_loss_weight"],
                auxiliary_loss_weight=config["auxiliary_loss_weight"],
                anchor_kl_weight=config["anchor_kl_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Correction loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(value).all() for value in model.parameters()):
                raise RuntimeError("Correction optimizer produced non-finite parameters")
            sums["loss"] += float(loss.detach())
            for name, value in parts.items():
                sums[name] += float(value.detach())
        history.append(
            {"epoch": epoch + 1, **{name: value / steps for name, value in sums.items()}}
        )
        print(
            json.dumps(
                {
                    "event": "bounded_correction_epoch",
                    "arm": arm,
                    "fold": fold,
                    "seed": seed,
                    **history[-1],
                }
            ),
            flush=True,
        )
    model.eval()
    correction_chunks, auxiliary_chunks = [], []
    with torch.inference_mode():
        for left in range(0, len(held), config["batch_size"]):
            output = model(
                torch.as_tensor(held_features[left : left + config["batch_size"]], device=device),
                torch.as_tensor(
                    outer_anchor[left : left + config["batch_size"]],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            correction_chunks.append(output["correction"].cpu().numpy())
            auxiliary_chunks.append(output["auxiliary_logits"].cpu().numpy())
    corrections = np.concatenate(correction_chunks).astype(np.float64)
    auxiliary_logits = np.concatenate(auxiliary_chunks).astype(np.float64)
    probabilities = apply_numpy_correction(outer_anchor, corrections)
    validate_probability(probabilities, len(held))
    np.savez_compressed(staging / "checkpoint.npz", **checkpoint(model))
    np.savez_compressed(
        staging / "predictions.npz",
        held_rows=held,
        sample_ids=data["sample_ids"][held],
        anchor_probabilities=outer_anchor,
        corrections=corrections,
        auxiliary_logits=auxiliary_logits,
        probabilities=probabilities,
    )
    receipt = {
        "status": "BOUNDED_FACTOR_CORRECTION_FIT_COMPLETE",
        "execution_lock_sha256": lock_hash,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "outer_held_labels_read": 0,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "optimizer_steps": int(steps * config["epochs"]),
        "parameters": model.trainable_parameters,
        "center_only": center_only,
        "history": history,
        "seconds": time.perf_counter() - started,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "arftr_ancestry": ancestry,
        "predictions_sha256": file_sha256(staging / "predictions.npz"),
        "checkpoint_sha256": file_sha256(staging / "checkpoint.npz"),
    }
    immutable_json(staging / "receipt.json", receipt)
    validate_lock(run, data, lock_hash)
    os.replace(staging, receipt_path.parent)
    print(
        json.dumps(
            {
                "event": "bounded_correction_fit_complete",
                "arm": arm,
                "fold": fold,
                "seed": seed,
                "seconds": receipt["seconds"],
            }
        ),
        flush=True,
    )
    del model, optimizer
    torch.cuda.empty_cache()
    return receipt


def transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    before, after = anchor.argmax(1) == labels, candidate.argmax(1) == labels
    rescues, harms = int((after & ~before).sum()), int((~after & before).sum())
    return {
        "rescues": rescues,
        "harms": harms,
        "net_corrections": rescues - harms,
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def class_precision_recall(metric: dict[str, Any]) -> dict[str, list[float]]:
    confusion = np.asarray(metric["confusion"], dtype=np.float64)
    diagonal = np.diag(confusion)
    precision = np.divide(
        diagonal,
        confusion.sum(0),
        out=np.zeros(3, dtype=np.float64),
        where=confusion.sum(0) != 0,
    )
    recall = np.divide(
        diagonal,
        confusion.sum(1),
        out=np.zeros(3, dtype=np.float64),
        where=confusion.sum(1) != 0,
    )
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def summarize(
    run: Path, protocol: dict[str, Any], data: dict[str, np.ndarray], lock_hash: str
) -> dict[str, Any]:
    rows = len(data["labels"])
    probabilities = np.full((len(ARMS), 3, rows, 3), np.nan)
    receipts = {}
    for fold in range(5):
        held = np.flatnonzero(data["folds"] == fold)
        _, anchors, _ = arftr_anchors(data, fold)
        probabilities[0][:, held] = anchors
        for arm_index, arm in enumerate(TRAINED_ARMS, start=1):
            for seed_index, seed in enumerate(SEEDS):
                prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
                receipt = read_json(receipt_path)
                if (
                    receipt["execution_lock_sha256"] != lock_hash
                    or receipt.get("arm") != arm
                    or receipt.get("fold") != fold
                    or receipt.get("seed") != seed
                    or receipt.get("outer_held_labels_read") != 0
                    or receipt.get("held_sample_ids_sha256")
                    != canonical_hash(data["sample_ids"][held].tolist())
                    or receipt["predictions_sha256"] != file_sha256(prediction_path)
                    or receipt["checkpoint_sha256"] != file_sha256(checkpoint_path)
                ):
                    raise RuntimeError("Correction fit inventory changed")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["sample_ids"], data["sample_ids"][held]):
                        raise RuntimeError("Correction output population changed")
                    probabilities[arm_index, seed_index, held] = saved["probabilities"]
                    if not np.array_equal(saved["anchor_probabilities"], anchors[seed_index]):
                        raise RuntimeError("Correction anchor bytes changed")
                receipts[f"{arm}/{fold}/{seed}"] = file_sha256(receipt_path)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Correction OOF coverage is incomplete")
    reference = np.load(ARFTR_RUN / "results/v0001/oof_probabilities.npz", allow_pickle=False)
    if not np.array_equal(probabilities[0], reference["seed_probabilities"][5]):
        raise RuntimeError("Exact ARFTR reference replay failed")
    reference.close()
    averaged = probabilities.mean(1)
    labels, scenarios = data["labels"], data["scenarios"]
    metrics = {arm: probability_metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)}
    seed_metrics = {
        arm: [probability_metrics(labels, probabilities[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    statistics, comparison_transitions = {}, {}
    for arm_index, arm in enumerate(TRAINED_ARMS, start=1):
        statistics[arm] = fsar._scenario_statistics(
            labels,
            averaged[arm_index],
            averaged[0],
            scenarios,
            resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
            seed=protocol["statistics"]["seed"] + arm_index,
        )
        comparison_transitions[arm] = transitions(labels, averaged[arm_index], averaged[0])
    primary = averaged[2]
    center_statistics = fsar._scenario_statistics(
        labels,
        primary,
        averaged[1],
        scenarios,
        resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
        seed=protocol["statistics"]["center_control_seed"],
    )
    scenario_metrics = {
        str(group): {
            arm: probability_metrics(
                labels[scenarios == group], averaged[index, scenarios == group]
            )
            for index, arm in enumerate(ARMS)
        }
        for group in np.unique(scenarios)
    }
    gain = metrics[ARMS[2]]["macro_f1"] - metrics[ARMS[0]]["macro_f1"]
    center_gain = metrics[ARMS[2]]["macro_f1"] - metrics[ARMS[1]]["macro_f1"]
    seed_sd = 100 * float(np.std([item["macro_f1"] for item in seed_metrics[ARMS[2]]], ddof=1))
    improved_scenarios = sum(
        scenario_metrics[group][ARMS[2]]["macro_f1"] > scenario_metrics[group][ARMS[0]]["macro_f1"]
        for group in scenario_metrics
    )
    class_losses = np.asarray(metrics[ARMS[0]]["per_class_f1"]) - np.asarray(
        metrics[ARMS[2]]["per_class_f1"]
    )
    support_masks = {
        "known_pure_support": data["node_support_complete"] & ~data["node_support_boundary"],
        "known_mixed_support": data["node_support_complete"] & data["node_support_boundary"],
        "unknown_support": ~data["node_support_complete"],
    }
    strata = {
        name: {
            "rows": int(mask.sum()),
            "metrics": {
                arm: probability_metrics(labels[mask], averaged[index, mask])
                for index, arm in enumerate(ARMS)
            },
            "primary_transitions_vs_arftr": transitions(
                labels[mask], primary[mask], averaged[0, mask]
            ),
        }
        for name, mask in support_masks.items()
    }
    gates = protocol["screen_gates"]
    checks = {
        "minimum_primary_macro_f1": metrics[ARMS[2]]["macro_f1"]
        >= gates["minimum_primary_macro_f1"],
        "minimum_gain_over_arftr_points": 100 * gain >= gates["minimum_gain_over_arftr_points"],
        "minimum_gain_over_center_control_points": 100 * center_gain
        >= gates["minimum_gain_over_center_control_points"],
        "minimum_net_corrections_over_arftr": comparison_transitions[ARMS[2]]["net_corrections"]
        >= gates["minimum_net_corrections_over_arftr"],
        "minimum_known_pure_net_corrections": strata["known_pure_support"][
            "primary_transitions_vs_arftr"
        ]["net_corrections"]
        >= gates["minimum_known_pure_net_corrections"],
        "minimum_known_mixed_net_corrections": strata["known_mixed_support"][
            "primary_transitions_vs_arftr"
        ]["net_corrections"]
        >= gates["minimum_known_mixed_net_corrections"],
        "both_primary_scenario_swap_p_maximum": statistics[ARMS[2]]["scenario_swap_one_sided_p"]
        <= gates["both_primary_scenario_swap_p_maximum"]
        and center_statistics["scenario_swap_one_sided_p"]
        <= gates["both_primary_scenario_swap_p_maximum"],
        "maximum_nll_delta_over_arftr_one_sided_upper95": statistics[ARMS[2]][
            "nll_delta_one_sided_upper95"
        ]
        <= gates["maximum_nll_delta_over_arftr_one_sided_upper95"],
        "maximum_brier_delta_over_arftr_one_sided_upper95": statistics[ARMS[2]][
            "brier_delta_one_sided_upper95"
        ]
        <= gates["maximum_brier_delta_over_arftr_one_sided_upper95"],
        "maximum_worst_class_f1_loss_points": 100 * float(class_losses.max())
        <= gates["maximum_worst_class_f1_loss_points"],
        "minimum_improved_scenarios": improved_scenarios >= gates["minimum_improved_scenarios"],
        "maximum_seed_macro_f1_sd_points": seed_sd <= gates["maximum_seed_macro_f1_sd_points"],
        "exact_arftr_replay": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result_dir = run / "results/v0001"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("Stale bounded-correction result retained")
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = result_dir / "oof_probabilities.npz"
    np.savez_compressed(
        oof_path,
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=scenarios,
        folds=data["folds"],
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=probabilities,
        mean_probabilities=averaged,
    )
    summary = {
        "status": "BOUNDED_FACTOR_CORRECTION_60_FIT_ADAPTIVE_SCREEN_COMPLETE",
        "complete": True,
        "study_id": protocol["study_id"],
        "model_fits": 60,
        "primary_arm": ARMS[2],
        "execution_lock_sha256": lock_hash,
        "fit_receipt_sha256": receipts,
        "oof_probabilities_sha256": file_sha256(oof_path),
        "metrics": metrics,
        "per_class_precision_recall": {
            arm: class_precision_recall(metric) for arm, metric in metrics.items()
        },
        "seed_metrics": seed_metrics,
        "transitions_vs_arftr": comparison_transitions,
        "statistics_vs_arftr": statistics,
        "primary_vs_center_control_statistics": center_statistics,
        "mechanism_effects_points": {
            "all_frames_minus_center_only": 100 * center_gain,
            "separate_minus_shared": 100
            * (metrics[ARMS[2]]["macro_f1"] - metrics[ARMS[3]]["macro_f1"]),
            "bounded_minus_unbounded": 100
            * (metrics[ARMS[2]]["macro_f1"] - metrics[ARMS[4]]["macro_f1"]),
        },
        "primary_effects": {
            "macro_f1_gain_over_arftr_points": 100 * gain,
            "accuracy_gain_over_arftr_points": 100
            * (metrics[ARMS[2]]["accuracy"] - metrics[ARMS[0]]["accuracy"]),
            "improved_scenarios": int(improved_scenarios),
            "seed_macro_f1_sd_points": seed_sd,
        },
        "scenario_metrics": scenario_metrics,
        "support_strata": strata,
        "screen_checks": checks,
        "screen_gate_passed": bool(all(checks.values())),
        "interpretation": "Adaptive screen on repeatedly inspected scenarios; external confirmation required.",
    }
    immutable_json(result_dir / "summary.json", summary)
    validate_lock(run, data, lock_hash)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "primary_macro_f1": metrics[ARMS[2]]["macro_f1"],
                "arftr_macro_f1": metrics[ARMS[0]]["macro_f1"],
                "screen_gate_passed": summary["screen_gate_passed"],
                "output": str(result_dir / "summary.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resource-gate-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    protocol, data, frames, lock_hash = prepare(run)
    gate_path = run / "resource_gate.json"
    if gate_path.exists():
        gate = read_json(gate_path)
        if gate.get("execution_lock_sha256") != lock_hash:
            raise RuntimeError("Resource gate belongs to another lock")
    else:
        gate = resource_gate(protocol, lock_hash)
        immutable_json(gate_path, gate)
    print(json.dumps(gate, indent=2), flush=True)
    if args.resource_gate_only:
        return
    store = fsar.FrameFeatureStore()
    for arm in TRAINED_ARMS:
        for fold in range(5):
            for seed in SEEDS:
                fit_one(run, protocol, data, frames, store, lock_hash, arm, fold, seed)
    summarize(run, protocol, data, lock_hash)


if __name__ == "__main__":
    main()
