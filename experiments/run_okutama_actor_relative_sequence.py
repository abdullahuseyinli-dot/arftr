"""Run the frozen actor-relative timestamped-frame evidence screen."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Actor-relative sequence training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import run_okutama_frame_supervision as fsar
import torch

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.actor_relative_sequence import (
    ARMS,
    ActorRelativeStateSequence,
    seed_sequence_training,
    sequence_loss,
)
from hac.cached_frame_supervision import (
    CENTER_INDEX,
    SOURCE_PATHS,
    validate_cached_frame_supervision_receipt,
)
from hac.frame_supervised_residual import class_weights
from hac.matr_artifacts import load_cached_study_data
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
FRAME_DATA = ROOT / ".runs/research_20260912/cached_frame_supervision_v2/data"
ARFTR_RUN = ROOT / ".runs/research_20260912/arftr_v1"
PROTOCOL = ROOT / "experiments/okutama_actor_relative_sequence_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/actor_relative_sequence_v1"
SEEDS = (42, 43, 44)


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
        or protocol["primary_arm"] != ARMS[-1]
        or tuple(training["outer_folds"]) != tuple(range(5))
        or tuple(training["outer_seeds"]) != SEEDS
        or training["model_fits"] != len(ARMS) * 5 * len(SEEDS)
        or training["epochs"] != 8
        or training["batch_size"] != 256
        or training["relation_loss_weight"] != 0.15
        or architecture["input_dim"] != 768
        or architecture["width"] != 64
        or architecture["temporal_blocks"] != 2
        or architecture["sequence_slots"] != 32
        or architecture["center_source"] != "short stream, immutable slot 8"
        or statistics["scenario_bootstrap_resamples"] != 10000
        or statistics["exact_scenario_swaps"] != 2048
        or statistics["primary_mechanism_contrasts"] != [[ARMS[4], ARMS[2]], [ARMS[4], ARMS[3]]]
    ):
        raise RuntimeError("Frozen actor-relative sequence protocol changed")


class SequenceStore:
    """Read the two immutable feature streams as timestamped node sequences."""

    def __init__(self, frames: dict[str, np.ndarray]) -> None:
        self.features = (
            np.load(ROOT / SOURCE_PATHS["short_feature"], mmap_mode="r", allow_pickle=False),
            np.load(ROOT / SOURCE_PATHS["long_feature"], mmap_mode="r", allow_pickle=False),
        )
        if any(value.shape != (4977, 16, 1, 768) or value.dtype != np.float16 for value in self.features):
            raise RuntimeError("Frozen DINO feature cache contract changed")
        for value in self.features:
            for start in range(0, len(value), 256):
                if not np.isfinite(value[start : start + 256]).all():
                    raise RuntimeError("Frozen DINO cache is non-finite")
        self.source_frames = np.concatenate(
            (frames["short_source_frames"], frames["long_source_frames"]), axis=1
        ).astype(np.int64)
        self.valid = np.concatenate((frames["short_valid"], frames["long_valid"]), axis=1)
        self.labels = np.concatenate(
            (frames["short_frame_labels"], frames["long_frame_labels"]), axis=1
        ).astype(np.int64)
        self.weights = np.concatenate(
            (frames["physical_weight"][0], frames["physical_weight"][1]), axis=1
        ).astype(np.float32)
        if self.source_frames.shape != (4977, 32) or self.valid.shape != (4977, 32):
            raise RuntimeError("Cached sequence metadata shape changed")
        raw_streams = np.concatenate((np.zeros(16, np.int64), np.ones(16, np.int64)))
        sort_key = np.where(
            self.valid,
            self.source_frames * 2 + raw_streams[None, :],
            np.iinfo(np.int64).max,
        )
        # Stable ordering retains crop variants at the same exact source timestamp.
        self.order = np.argsort(sort_key, axis=1, kind="stable")
        self.streams = np.broadcast_to(raw_streams, self.source_frames.shape)[
            np.arange(4977)[:, None], self.order
        ].astype(np.int64)
        self.sorted_frames = self.source_frames[np.arange(4977)[:, None], self.order]
        self.sorted_valid = self.valid[np.arange(4977)[:, None], self.order]
        self.sorted_labels = self.labels[np.arange(4977)[:, None], self.order]
        self.sorted_weights = self.weights[np.arange(4977)[:, None], self.order]
        self.center_indices = np.argmax(self.order == CENTER_INDEX, axis=1).astype(np.int64)
        if (
            not np.all(self.order[np.arange(4977), self.center_indices] == CENTER_INDEX)
            or not self.sorted_valid[np.arange(4977), self.center_indices].all()
            or np.any(self.sorted_labels[self.sorted_valid] < 0)
            or np.any(self.sorted_weights[self.sorted_valid] <= 0)
            or np.any(self.sorted_weights[~self.sorted_valid] != 0)
        ):
            raise RuntimeError("Cached timestamp sequence identity changed")
        self.center_frames = frames["center_frames"].astype(np.int64)

    def batch(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0 or rows.min() < 0 or rows.max() >= 4977:
            raise ValueError("Sequence rows are malformed")
        raw = np.concatenate(
            (self.features[0][rows, :, 0], self.features[1][rows, :, 0]), axis=1
        ).astype(np.float32)
        local = np.arange(len(rows))[:, None]
        feature = raw[local, self.order[rows]]
        timestamp = (self.sorted_frames[rows] - self.center_frames[rows, None]).astype(np.float32) / 30.0
        timestamp[~self.sorted_valid[rows]] = 0.0
        if not np.isfinite(feature).all() or not np.isfinite(timestamp).all():
            raise RuntimeError("Sequence batch is non-finite")
        return {
            "features": feature,
            "timestamps": timestamp,
            "streams": self.streams[rows],
            "valid": self.sorted_valid[rows],
            "center_indices": self.center_indices[rows],
            "labels": self.sorted_labels[rows],
            "weights": self.sorted_weights[rows],
        }


def make_model(protocol: dict[str, Any], arm: str) -> ActorRelativeStateSequence:
    architecture = protocol["architecture"]
    return ActorRelativeStateSequence(
        arm,
        input_dim=architecture["input_dim"],
        width=architecture["width"],
        dropout=architecture["dropout"],
        temporal_blocks=architecture["temporal_blocks"],
        parameter_limit=architecture["parameter_limit"],
    )


def expected_parameters(protocol: dict[str, Any], arm: str) -> int:
    return make_model(protocol, arm).trainable_parameters


def fit_paths(run: Path, arm: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    directory = run / "models" / arm / f"fold-{fold}" / f"seed-{seed}"
    return directory / "predictions.npz", directory / "checkpoint.npz", directory / "receipt.json"


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], str]:
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    data = load_cached_study_data(SOURCE)
    frames, frame_receipt = validate_cached_frame_supervision_receipt(ROOT, FRAME_DATA)
    for name in ("sample_ids", "labels", "scenarios", "folds", "recordings", "tracks"):
        if not np.array_equal(data[name], frames[name]):
            raise RuntimeError(f"Cached sequence identity differs from study data: {name}")
    arftr_oof = ARFTR_RUN / "results/v0001/oof_probabilities.npz"
    with np.load(arftr_oof, allow_pickle=False) as saved:
        if (
            tuple(saved["arms"].tolist())[5] != "r5_arftr_full"
            or tuple(saved["seeds"].tolist()) != SEEDS
            or not np.array_equal(saved["sample_ids"], data["sample_ids"])
            or not np.array_equal(saved["labels"], data["labels"])
        ):
            raise RuntimeError("Frozen ARFTR reference identity changed")
    paths = {
        PROTOCOL,
        Path(__file__),
        ROOT / "experiments/audit_okutama_actor_relative_sequence.py",
        ROOT / "src/hac/actor_relative_sequence.py",
        ROOT / "src/hac/cached_frame_supervision.py",
        ROOT / "src/hac/frame_supervised_residual.py",
        ROOT / "src/hac/matr_artifacts.py",
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/source_swap_data.py",
        FRAME_DATA / "frame_supervision.npz",
        FRAME_DATA / "receipt.json",
        arftr_oof,
        *(ROOT / value for value in SOURCE_PATHS.values()),
        *(Path(record["path"]) for record in frame_receipt["source_code"].values()),
    }
    if any(not path.is_file() for path in paths):
        raise RuntimeError("A locked actor-relative sequence input is missing")
    if run.exists() and any(run.iterdir()) and not (run / "execution_lock.json").is_file():
        raise RuntimeError("Refusing a nonempty actor-relative sequence run without a lock")
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "execution_lock.json"
    lock = {
        "status": "ACTOR_RELATIVE_SEQUENCE_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "rows": len(data["labels"]),
        "model_fits": protocol["training"]["model_fits"],
        "outer_held_labels_read": 0,
        "run_directory": str(run.resolve()),
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "input_files": [file_record(path) for path in sorted(paths, key=str)],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise RuntimeError("Actor-relative sequence execution lock changed")
    else:
        immutable_json(lock_path, lock)
    return protocol, data, frames, file_sha256(lock_path)


def validate_lock(run: Path, data: dict[str, np.ndarray], lock_hash: str) -> None:
    path = run / "execution_lock.json"
    if file_sha256(path) != lock_hash:
        raise RuntimeError("Actor-relative sequence execution lock hash changed")
    lock = read_json(path)
    if (
        lock.get("sample_ids_sha256") != canonical_hash(data["sample_ids"].tolist())
        or Path(lock.get("run_directory", "")).resolve() != run.resolve()
        or lock.get("outer_held_labels_read") != 0
    ):
        raise RuntimeError("Actor-relative sequence lock identity changed")
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if lock.get("environment") != environment:
        raise RuntimeError("Actor-relative sequence runtime changed")
    for record in lock["input_files"]:
        path = (ROOT / record["path"]).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked actor-relative sequence input changed: {record['path']}")


def to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(batch["features"], device=device),
        "timestamps": torch.as_tensor(batch["timestamps"], device=device),
        "streams": torch.as_tensor(batch["streams"], device=device),
        "valid": torch.as_tensor(batch["valid"], device=device),
        "center_indices": torch.as_tensor(batch["center_indices"], device=device),
        "labels": torch.as_tensor(batch["labels"], device=device),
        "physical_weights": torch.as_tensor(batch["weights"], device=device),
    }


def resource_gate(protocol: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Actor-relative sequence screen requires CUDA")
    device, batch_size = torch.device("cuda"), protocol["training"]["batch_size"]
    torch.cuda.reset_peak_memory_stats(device)
    models = {}
    for arm in ARMS:
        seed_sequence_training(20260912)
        model = make_model(protocol, arm).to(device)
        rows, slots = batch_size, protocol["architecture"]["sequence_slots"]
        valid = torch.ones(rows, slots, dtype=torch.bool, device=device)
        payload = {
            "features": torch.randn(rows, slots, 768, device=device),
            "timestamps": torch.linspace(-1.0, 1.0, slots, device=device).expand(rows, -1),
            "streams": (torch.arange(slots, device=device)[None] >= 16).expand(rows, -1).long(),
            "valid": valid,
            "center_indices": torch.full((rows,), 8, device=device, dtype=torch.long),
            "labels": torch.arange(slots, device=device)[None].expand(rows, -1).remainder(3).long(),
            "physical_weights": torch.ones(rows, slots, device=device),
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["training"]["learning_rate"])
        loss, _ = sequence_loss(
            model, **payload, class_weight=torch.ones(3, device=device),
            relation_loss_weight=protocol["training"]["relation_loss_weight"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["training"]["gradient_clip_norm"])
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("Sequence resource gate produced non-finite loss")
        models[arm] = model.trainable_parameters
        del model, optimizer
        torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak >= 4 * 1024**3 or max(models.values()) > protocol["architecture"]["parameter_limit"]:
        raise RuntimeError("Sequence resource gate exceeds the declared resource budget")
    return {
        "status": "ACTOR_RELATIVE_SEQUENCE_RESOURCE_GATE_PASSED",
        "execution_lock_sha256": lock_hash,
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "parameters": models,
        "peak_allocated_bytes": peak,
        "labels_read": 0,
        "optimizer_steps": 0,
    }


def checkpoint(model: ActorRelativeStateSequence) -> dict[str, np.ndarray]:
    return {name.replace(".", "__"): value.detach().cpu().numpy() for name, value in model.state_dict().items()}


def fit_one(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    store: SequenceStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
    if receipt_path.is_file():
        return read_json(receipt_path)
    if any(path.exists() for path in (prediction_path, checkpoint_path)):
        raise RuntimeError("Partial actor-relative fit exists outside an immutable receipt")
    train, held = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
    seed_sequence_training(seed)
    device = torch.device("cuda")
    model = make_model(protocol, arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol["training"]["learning_rate"], weight_decay=protocol["training"]["weight_decay"]
    )
    weight = torch.as_tensor(class_weights(data["labels"], train), device=device)
    rng, batch_size = np.random.default_rng(seed + 10_000 * fold), protocol["training"]["batch_size"]
    start = time.perf_counter()
    steps = 0
    for epoch in range(protocol["training"]["epochs"]):
        order = rng.permutation(train)
        for left in range(0, len(train), batch_size):
            rows = order[left : left + batch_size]
            payload = to_device(store.batch(rows), device)
            model.train()
            loss, parts = sequence_loss(
                model, **payload, class_weight=weight,
                relation_loss_weight=protocol["training"]["relation_loss_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Actor-relative sequence loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["training"]["gradient_clip_norm"])
            optimizer.step()
            steps += 1
        print(json.dumps({"event": "actor_relative_sequence_epoch", "arm": arm, "fold": fold, "seed": seed, "epoch": epoch + 1, "loss": float(loss.detach().cpu()), **{key: float(value.detach().cpu()) for key, value in parts.items()}}), flush=True)
    model.eval()
    probabilities, logits = [], []
    with torch.inference_mode():
        for left in range(0, len(held), batch_size):
            payload = to_device(store.batch(held[left : left + batch_size]), device)
            output = model(payload["features"], payload["timestamps"], payload["streams"], payload["valid"], payload["center_indices"])
            logits.append(output["center_logits"].detach().cpu().numpy())
            probabilities.append(torch.softmax(output["center_logits"], 1).detach().cpu().numpy())
    prediction = np.concatenate(probabilities).astype(np.float64)
    center_logits = np.concatenate(logits).astype(np.float64)
    if prediction.shape != (len(held), 3) or not np.isfinite(prediction).all() or not np.allclose(prediction.sum(1), 1.0, atol=1e-6):
        raise RuntimeError("Actor-relative held prediction is malformed")
    staging = receipt_path.parent.with_name(receipt_path.parent.name + ".incomplete")
    if staging.exists():
        raise RuntimeError("Stale incomplete actor-relative fit; use a fresh run directory")
    staging.mkdir(parents=True)
    with (staging / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, sample_ids=data["sample_ids"][held], held_rows=held, probabilities=prediction, center_logits=center_logits)
    with (staging / "checkpoint.npz").open("xb") as stream:
        np.savez_compressed(stream, **checkpoint(model))
    receipt = {
        "status": "ACTOR_RELATIVE_SEQUENCE_FIT_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_hash,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "outer_held_labels_read": 0,
        "optimizer_steps": steps,
        "parameters": model.trainable_parameters,
        "seconds": time.perf_counter() - start,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "predictions_sha256": file_sha256(staging / "predictions.npz"),
        "checkpoint_sha256": file_sha256(staging / "checkpoint.npz"),
    }
    immutable_json(staging / "receipt.json", receipt)
    validate_lock(run, data, lock_hash)
    os.replace(staging, receipt_path.parent)
    print(json.dumps({"event": "actor_relative_sequence_fit_complete", "arm": arm, "fold": fold, "seed": seed, "seconds": receipt["seconds"]}), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return receipt


def transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    before, after = anchor.argmax(1) == labels, candidate.argmax(1) == labels
    return {
        "rescues": int((after & ~before).sum()),
        "harms": int((~after & before).sum()),
        "net_corrections": int((after & ~before).sum() - (~after & before).sum()),
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def summarize(run: Path, protocol: dict[str, Any], data: dict[str, np.ndarray], lock_hash: str) -> dict[str, Any]:
    rows = len(data["labels"])
    probabilities = np.full((len(ARMS), len(SEEDS), rows, 3), np.nan)
    receipt_hashes = {}
    for fold in range(5):
        held = np.flatnonzero(data["folds"] == fold)
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
                receipt = read_json(receipt_path)
                if (
                    receipt.get("status") != "ACTOR_RELATIVE_SEQUENCE_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
                    or receipt.get("execution_lock_sha256") != lock_hash
                    or receipt.get("arm") != arm or receipt.get("fold") != fold or receipt.get("seed") != seed
                    or receipt.get("outer_held_labels_read") != 0
                    or receipt.get("held_sample_ids_sha256") != canonical_hash(data["sample_ids"][held].tolist())
                    or receipt.get("predictions_sha256") != file_sha256(prediction_path)
                    or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
                    or receipt.get("parameters") != expected_parameters(protocol, arm)
                ):
                    raise RuntimeError("Actor-relative fit receipt changed")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["sample_ids"], data["sample_ids"][held]) or not np.array_equal(saved["held_rows"], held):
                        raise RuntimeError("Actor-relative held identity changed")
                    probabilities[arm_index, seed_index, held] = saved["probabilities"]
                receipt_hashes[f"{arm}/{fold}/{seed}"] = file_sha256(receipt_path)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Actor-relative OOF coverage is incomplete")
    with np.load(ARFTR_RUN / "results/v0001/oof_probabilities.npz", allow_pickle=False) as saved:
        reference_seed = saved["seed_probabilities"][5].copy()
    if reference_seed.shape != (3, rows, 3):
        raise RuntimeError("ARFTR reference seed shape changed")
    average, reference = probabilities.mean(1), reference_seed.mean(0)
    labels, scenarios = data["labels"], data["scenarios"]
    metrics = {arm: probability_metrics(labels, average[index]) for index, arm in enumerate(ARMS)}
    arftr_metrics = probability_metrics(labels, reference)
    seed_metrics = {arm: [probability_metrics(labels, probabilities[index, seed]) for seed in range(3)] for index, arm in enumerate(ARMS)}
    statistics = {
        arm: fsar._scenario_statistics(labels, average[index], reference, scenarios, resamples=protocol["statistics"]["scenario_bootstrap_resamples"], seed=protocol["statistics"]["seed"] + index)
        for index, arm in enumerate(ARMS)
    }
    primary = average[-1]
    mechanism_statistics = {
        f"{ARMS[4]}_minus_{control}": fsar._scenario_statistics(labels, primary, average[ARMS.index(control)], scenarios, resamples=protocol["statistics"]["scenario_bootstrap_resamples"], seed=20261000 + ARMS.index(control))
        for control in (ARMS[2], ARMS[3])
    }
    scenario_metrics = {str(group): {"arftr": probability_metrics(labels[scenarios == group], reference[scenarios == group]), **{arm: probability_metrics(labels[scenarios == group], average[index, scenarios == group]) for index, arm in enumerate(ARMS)}} for group in np.unique(scenarios)}
    support_masks = {
        "known_pure_support_diagnostic_only": data["node_support_complete"] & ~data["node_support_boundary"],
        "known_mixed_support_diagnostic_only": data["node_support_complete"] & data["node_support_boundary"],
        "unknown_annotation_support_diagnostic_only": ~data["node_support_complete"],
    }
    support = {name: {"rows": int(mask.sum()), "arftr": probability_metrics(labels[mask], reference[mask]), "arms": {arm: probability_metrics(labels[mask], average[index, mask]) for index, arm in enumerate(ARMS)}, "primary_transitions_vs_arftr": transitions(labels[mask], primary[mask], reference[mask]), "not_an_inference_feature": True} for name, mask in support_masks.items()}
    gain = metrics[ARMS[4]]["macro_f1"] - arftr_metrics["macro_f1"]
    primary_transitions = transitions(labels, primary, reference)
    class_loss = np.asarray(arftr_metrics["per_class_f1"]) - np.asarray(metrics[ARMS[4]]["per_class_f1"])
    improved = sum(value[ARMS[4]]["macro_f1"] > value["arftr"]["macro_f1"] for value in scenario_metrics.values())
    seed_sd = 100 * float(np.std([entry["macro_f1"] for entry in seed_metrics[ARMS[4]]], ddof=1))
    gates = protocol["screen_gates"]
    checks = {
        "minimum_primary_macro_f1": metrics[ARMS[4]]["macro_f1"] >= gates["minimum_primary_macro_f1"],
        "minimum_gain_over_arftr_points": 100 * gain >= gates["minimum_gain_over_arftr_points"],
        "minimum_net_corrections_over_arftr": primary_transitions["net_corrections"] >= gates["minimum_net_corrections_over_arftr"],
        "maximum_worst_class_f1_loss_points": 100 * float(class_loss.max()) <= gates["maximum_worst_class_f1_loss_points"],
        "minimum_improved_scenarios": improved >= gates["minimum_improved_scenarios"],
        "maximum_seed_macro_f1_sd_points": seed_sd <= gates["maximum_seed_macro_f1_sd_points"],
        "primary_scenario_swap_p_maximum": statistics[ARMS[4]]["scenario_swap_one_sided_p"] <= gates["primary_scenario_swap_p_maximum"],
        "exact_arftr_replay": True,
    }
    result_dir = run / "results/v0001"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError("Stale actor-relative result exists")
    result_dir.mkdir(parents=True, exist_ok=True)
    oof_path = result_dir / "oof_probabilities.npz"
    np.savez_compressed(oof_path, sample_ids=data["sample_ids"], labels=labels, scenarios=scenarios, folds=data["folds"], arms=np.asarray(ARMS), seeds=np.asarray(SEEDS), seed_probabilities=probabilities, mean_probabilities=average, arftr_seed_probabilities=reference_seed, arftr_mean_probabilities=reference)
    summary = {
        "status": "ACTOR_RELATIVE_SEQUENCE_75_FIT_ADAPTIVE_SCREEN_COMPLETE", "complete": True,
        "study_id": protocol["study_id"], "model_fits": 75, "primary_arm": ARMS[4],
        "execution_lock_sha256": lock_hash, "fit_receipt_sha256": receipt_hashes, "oof_probabilities_sha256": file_sha256(oof_path),
        "metrics": metrics, "arftr_metrics": arftr_metrics, "seed_metrics": seed_metrics,
        "transitions_vs_arftr": {arm: transitions(labels, average[index], reference) for index, arm in enumerate(ARMS)},
        "statistics_vs_arftr": statistics, "primary_mechanism_statistics": mechanism_statistics,
        "scenario_metrics": scenario_metrics, "support_diagnostics": support,
        "primary_effects": {"macro_f1_gain_over_arftr_points": 100 * gain, "accuracy_gain_over_arftr_points": 100 * (metrics[ARMS[4]]["accuracy"] - arftr_metrics["accuracy"]), "net_corrections": primary_transitions["net_corrections"], "improved_scenarios": int(improved), "seed_macro_f1_sd_points": seed_sd},
        "screen_checks": {name: bool(value) for name, value in checks.items()}, "screen_gate_passed": bool(all(checks.values())),
        "limitations": "Adaptive repeated-development standalone evidence screen. Support strata are label-derived diagnostics, not inference routing inputs."
    }
    immutable_json(result_dir / "summary.json", summary)
    validate_lock(run, data, lock_hash)
    print(json.dumps({"status": summary["status"], "primary_macro_f1": metrics[ARMS[4]]["macro_f1"], "arftr_macro_f1": arftr_metrics["macro_f1"], "screen_gate_passed": summary["screen_gate_passed"], "output": str(result_dir / "summary.json")}, indent=2), flush=True)
    return summary


def run(run_directory: Path, *, resource_gate_only: bool = False) -> None:
    run_directory = run_directory.resolve()
    run_directory.relative_to(ROOT.resolve())
    protocol, data, frames, lock_hash = prepare(run_directory)
    validate_lock(run_directory, data, lock_hash)
    gate = resource_gate(protocol, lock_hash)
    immutable_json(run_directory / "resource_gate.json", gate)
    if resource_gate_only:
        print(json.dumps(gate, indent=2), flush=True)
        return
    store = SequenceStore(frames)
    for arm in ARMS:
        for fold in range(5):
            for seed in SEEDS:
                fit_one(run_directory, protocol, data, store, lock_hash, arm, fold, seed)
    summarize(run_directory, protocol, data, lock_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resource-gate-only", action="store_true")
    args = parser.parse_args()
    run(args.run, resource_gate_only=args.resource_gate_only)


if __name__ == "__main__":
    main()
