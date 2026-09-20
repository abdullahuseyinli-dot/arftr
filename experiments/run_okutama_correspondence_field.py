"""Plan, preflight, fit, and embargoed-summary the signed-field Stage-1 screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Field training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.nn import functional as F

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.correspondence_field_reader import ARMS, CorrespondenceFieldReader
from hac.correspondence_field_training import (
    ActorUniformSampler,
    conditional_motion_candidate,
    seed_field_training,
    upright_balanced_accuracy,
    upright_class_weights,
)
from hac.matr_artifacts import load_cached_study_data
from hac.source_swap_data import immutable_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
PROTOCOL = ROOT / "experiments/okutama_correspondence_field_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/correspondence_field_stage1_v1"
INITIAL_SEEDS = (42,)
CONFIRMATION_SEEDS = (42, 43, 44)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    inputs, architecture, training, preflight = (
        protocol["inputs"],
        protocol["architecture"],
        protocol["training"],
        protocol["preflight"],
    )
    if (
        tuple(protocol["arms"]) != ARMS
        or protocol["primary_arm"] != "c2"
        or inputs["pairs_per_center"] != 15
        or inputs["max_points_per_pair"] != 32
        or inputs["point_channels"] != 10
        or inputs["pair_channels"] != 5
        or inputs["collapsed_channels"] != 31
        or architecture["context_dim"] != 2304
        or architecture["width"] != 64
        or architecture["heads"] != 4
        or architecture["attention_blocks"] != 2
        or architecture["feedforward_dim"] != 128
        or architecture["maximum_parameter_difference_fraction"] != 0.1
        or tuple(training["outer_folds"]) != tuple(range(5))
        or training["initial_seed"] != 42
        or tuple(training["confirmation_seeds"]) != (43, 44)
        or training["optimizer_steps_per_fit"] != 800
        or training["batch_size"] != 128
        or training["learning_rate"] != 3e-4
        or training["weight_decay"] != 1e-4
        or training["gradient_clip_norm"] != 1.0
        or preflight["steps_per_arm"] * len(ARMS) > preflight["maximum_optimizer_steps"]
        or not preflight["uses_synthetic_targets"]
        or not protocol["execution_boundary"]["fit_metrics_embargoed_until_complete_inventory"]
        or not protocol["execution_boundary"]["stage2_forbidden_before_stage1_pass"]
    ):
        raise RuntimeError("Frozen correspondence-field protocol changed")


class FieldStore:
    """Observation-only cache; it contains no labels, support flags, or predictions."""

    def __init__(self, path: Path, expected_sample_ids: np.ndarray | None = None) -> None:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "sample_ids",
                "context",
                "point_features",
                "point_input_valid",
                "pair_features",
                "pair_input_valid",
                "midpoint_seconds",
                "collapsed_features",
            }
            if set(saved.files) != required:
                raise RuntimeError("Field cache member allowlist changed")
            for name in required:
                setattr(self, name, saved[name])
        rows = len(self.sample_ids)
        expected = {
            "context": (rows, 2304),
            "point_features": (rows, 15, 32, 10),
            "point_input_valid": (rows, 15, 32),
            "pair_features": (rows, 15, 5),
            "pair_input_valid": (rows, 15),
            "midpoint_seconds": (rows, 15),
            "collapsed_features": (rows, 31),
        }
        if (
            rows != 4977
            or self.sample_ids.shape != (rows,)
            or len(np.unique(self.sample_ids)) != rows
            or any(getattr(self, name).shape != shape for name, shape in expected.items())
            or self.point_input_valid.dtype != np.bool_
            or self.pair_input_valid.dtype != np.bool_
            or not np.array_equal(self.pair_input_valid, self.point_input_valid.any(2))
            or np.any(self.point_features[~self.point_input_valid] != 0)
            or np.any(self.pair_features[~self.pair_input_valid] != 0)
        ):
            raise RuntimeError("Field cache shape, identity, or mask contract changed")
        for name in (
            "context",
            "point_features",
            "pair_features",
            "midpoint_seconds",
            "collapsed_features",
        ):
            values = getattr(self, name)
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise RuntimeError(f"Field cache array is not finite float32: {name}")
        if expected_sample_ids is not None and not np.array_equal(
            self.sample_ids.astype(str), np.asarray(expected_sample_ids).astype(str)
        ):
            raise RuntimeError("Field cache order differs from task metadata")

    def batch(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0 or rows.min() < 0 or rows.max() >= len(self.sample_ids):
            raise ValueError("Malformed field batch rows")
        return {
            "context": self.context[rows],
            "points": self.point_features[rows],
            "point_valid": self.point_input_valid[rows],
            "pair_features": self.pair_features[rows],
            "pair_valid": self.pair_input_valid[rows],
            "midpoint_times": self.midpoint_seconds[rows],
            "summaries": self.collapsed_features[rows],
        }


def to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(values, device=device) for name, values in batch.items()}


def _input_paths(protocol: dict[str, Any]) -> list[Path]:
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "experiments/audit_okutama_correspondence_field.py",
        ROOT / "experiments/build_okutama_correspondence_field.py",
        ROOT / "src/hac/correspondence_field_data.py",
        ROOT / "src/hac/correspondence_field_reader.py",
        ROOT / "src/hac/correspondence_field_training.py",
        ROOT / "src/hac/actor_memory_base.py",
        ROOT / "src/hac/matr_artifacts.py",
        ROOT / "src/hac/source_swap_data.py",
        ROOT / protocol["inputs"]["field_cache"],
        ROOT / protocol["inputs"]["field_receipt"],
        ROOT / protocol["inputs"]["field_audit"],
        ROOT / protocol["inputs"]["arftr_oof"],
        ROOT / protocol["inputs"]["historical_residual_diagnostic"],
        ROOT / protocol["inputs"]["video_context"],
        ROOT / protocol["inputs"]["center_dino"],
        ROOT / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json",
        ROOT / protocol["inputs"]["correspondence_source"] / "summary.json",
        ROOT / protocol["inputs"]["correspondence_source"] / "ccac_features.npz",
    ]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("A locked correspondence-field input is missing")
    for key in (
        "field_cache",
        "field_receipt",
        "field_audit",
        "video_context",
        "center_dino",
        "arftr_oof",
        "historical_residual_diagnostic",
    ):
        expected = protocol["inputs"].get(f"{key}_sha256")
        if expected and file_sha256(ROOT / protocol["inputs"][key]) != expected:
            raise RuntimeError(f"Pinned protocol input changed: {key}")
    receipt = read_json(ROOT / protocol["inputs"]["field_receipt"])
    cache = ROOT / protocol["inputs"]["field_cache"]
    if (
        receipt.get("status") != "CORRESPONDENCE_FIELD_OBSERVATION_CACHE_COMPLETE"
        or receipt.get("labels_read") != 0
        or receipt.get("annotation_fields_used_for_inference") != []
        or receipt.get("output_sha256", {}).get(cache.name) != file_sha256(cache)
    ):
        raise RuntimeError("Observation-only field-cache receipt changed")
    source_paths = {
        "p8_summary": ROOT / protocol["inputs"]["correspondence_source"] / "summary.json",
        "p8_features": ROOT / protocol["inputs"]["correspondence_source"] / "ccac_features.npz",
        "memory": ROOT / protocol["inputs"]["video_context"],
        "dino": ROOT / protocol["inputs"]["center_dino"],
        "dino_summary": ROOT
        / ".runs/research_20260907/okutama_native_video_p0_r1/dinov2_full/summary.json",
        "builder": ROOT / "experiments/build_okutama_correspondence_field.py",
        "data_code": ROOT / "src/hac/correspondence_field_data.py",
    }
    if receipt.get("source_sha256") != {
        name: file_sha256(path) for name, path in source_paths.items()
    }:
        raise RuntimeError("Field-cache source/code ancestry changed")
    return paths


def prepare_lock(run: Path, protocol: dict[str, Any], sample_ids: np.ndarray) -> str:
    run = run.resolve()
    run.relative_to(ROOT.resolve())
    if run.exists() and any(run.iterdir()) and not (run / "execution_lock.json").is_file():
        raise RuntimeError("Refusing a nonempty field run without an execution lock")
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "execution_lock.json"
    lock = {
        "status": "CORRESPONDENCE_FIELD_STAGE1_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "run_directory": str(run),
        "rows": len(sample_ids),
        "sample_ids_sha256": canonical_hash(sample_ids.astype(str).tolist()),
        "outer_held_labels_read_by_fit": 0,
        "input_files": [file_record(path) for path in sorted(_input_paths(protocol), key=str)],
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
            raise RuntimeError("Correspondence-field execution lock changed")
    else:
        immutable_json(lock_path, lock)
    return file_sha256(lock_path)


def validate_lock(
    run: Path, protocol: dict[str, Any], sample_ids: np.ndarray, lock_hash: str
) -> None:
    lock_path = run / "execution_lock.json"
    if file_sha256(lock_path) != lock_hash:
        raise RuntimeError("Correspondence-field execution-lock hash changed")
    lock = read_json(lock_path)
    if (
        lock.get("sample_ids_sha256") != canonical_hash(sample_ids.astype(str).tolist())
        or lock.get("outer_held_labels_read_by_fit") != 0
        or Path(lock.get("run_directory", "")).resolve() != run.resolve()
    ):
        raise RuntimeError("Correspondence-field execution identity changed")
    for record in lock["input_files"]:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked field input changed: {record['path']}")


def model_for(protocol: dict[str, Any], arm: str) -> CorrespondenceFieldReader:
    return CorrespondenceFieldReader(arm, dropout=protocol["architecture"]["dropout"])


def metadata_plan(protocol: dict[str, Any], run: Path) -> dict[str, Any]:
    cache = ROOT / protocol["inputs"]["field_cache"]
    preflight_path = run / "preflight.json"
    preflight = read_json(preflight_path) if preflight_path.is_file() else None
    return {
        "status": "CORRESPONDENCE_FIELD_STAGE1_METADATA_PLAN",
        "study_id": protocol["study_id"],
        "field_cache_exists": cache.is_file(),
        "field_cache_bytes": cache.stat().st_size if cache.is_file() else None,
        "build_command": (
            ".\\.venv\\Scripts\\python.exe experiments/build_okutama_correspondence_field.py"
        ),
        "preflight_command": (
            ".\\.venv\\Scripts\\python.exe experiments/run_okutama_correspondence_field.py "
            f"--stage preflight --run {run.relative_to(ROOT).as_posix()}"
        ),
        "initial_fit_command": (
            ".\\.venv\\Scripts\\python.exe experiments/run_okutama_correspondence_field.py "
            f"--stage fit --phase initial --run {run.relative_to(ROOT).as_posix()}"
        ),
        "initial_fits": 15,
        "optimizer_steps_per_fit": 800,
        "total_initial_optimizer_steps": 12000,
        "preflight_complete": preflight is not None,
        "preflight_estimated_initial_minutes": (
            preflight.get("estimated_initial_fit_minutes") if preflight else None
        ),
        "stage2_status": "FORBIDDEN_UNTIL_STAGE1_GATE_PASSES",
        "task_labels_read": 0,
        "task_metrics_released": 0,
    }


def preflight(
    run: Path, protocol: dict[str, Any], store: FieldStore, lock_hash: str
) -> dict[str, Any]:
    path = run / "preflight.json"
    if path.is_file():
        result = read_json(path)
        if result.get("execution_lock_sha256") != lock_hash:
            raise RuntimeError("Completed preflight belongs to another execution lock")
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("Correspondence-field screen requires CUDA")
    device = torch.device("cuda")
    rows = np.arange(protocol["training"]["batch_size"], dtype=np.int64)
    payload = to_device(store.batch(rows), device)
    synthetic = torch.arange(len(rows), device=device).remainder(2).to(torch.float32)
    step_times, parameters = {}, {}
    full_window_gradient_by_pair = None
    torch.cuda.reset_peak_memory_stats(device)
    for arm_index, arm in enumerate(ARMS):
        seed_field_training(20260912 + arm_index)
        model = model_for(protocol, arm).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=protocol["training"]["learning_rate"],
            weight_decay=protocol["training"]["weight_decay"],
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        final_loss = None
        for _ in range(protocol["preflight"]["steps_per_arm"]):
            output = model(**payload)
            loss = F.binary_cross_entropy_with_logits(output["motion_logits"], synthetic)
            if not torch.isfinite(loss):
                raise RuntimeError("Synthetic preflight loss became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                protocol["training"]["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        step_times[arm] = {
            "seconds": seconds,
            "seconds_per_synthetic_optimizer_step": seconds
            / protocol["preflight"]["steps_per_arm"],
            "final_synthetic_loss": final_loss,
        }
        parameters[arm] = model.capacity_report
        if arm == "c2":
            probe = {
                "context": torch.randn(1, 2304, device=device),
                "points": torch.randn(1, 15, 32, 10, device=device, requires_grad=True),
                "point_valid": torch.ones(1, 15, 32, dtype=torch.bool, device=device),
                "pair_features": torch.ones(1, 15, 5, device=device),
                "pair_valid": torch.ones(1, 15, dtype=torch.bool, device=device),
                "midpoint_times": torch.linspace(-1, 1, 15, device=device).unsqueeze(0),
                "summaries": torch.randn(1, 31, device=device),
            }
            model.eval()
            model(**probe)["motion_logits"].sum().backward()
            full_window_gradient_by_pair = (
                probe["points"].grad.abs().sum(dim=(0, 2, 3)).detach().cpu().tolist()
            )
            if not all(value > 0 for value in full_window_gradient_by_pair):
                raise RuntimeError("C2 center query does not receive gradients from every pair")
        del model, optimizer
        torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device))
    estimated_seconds = sum(step_times[arm]["seconds_per_synthetic_optimizer_step"] for arm in ARMS)
    estimated_seconds *= 5 * protocol["training"]["optimizer_steps_per_fit"]
    estimated_seconds = 1.25 * estimated_seconds + 60.0
    result = {
        "status": "CORRESPONDENCE_FIELD_LABEL_BLIND_PREFLIGHT_PASSED",
        "execution_lock_sha256": lock_hash,
        "device": torch.cuda.get_device_name(device),
        "batch_size": len(rows),
        "synthetic_optimizer_steps": protocol["preflight"]["steps_per_arm"] * len(ARMS),
        "task_training_steps": 0,
        "task_labels_read": 0,
        "task_metrics_released": 0,
        "per_arm": step_times,
        "parameters": parameters,
        "c2_full_window_gradient_by_pair": full_window_gradient_by_pair,
        "peak_allocated_bytes": peak,
        "estimated_initial_fit_minutes": estimated_seconds / 60.0,
        "estimate_safety_factor": 1.25,
        "estimate_fixed_overhead_seconds": 60.0,
        "within_twenty_minute_boundary": estimated_seconds
        <= 60 * protocol["execution_boundary"]["maximum_unattended_expected_minutes"],
    }
    if peak >= protocol["preflight"]["maximum_peak_allocated_bytes"]:
        raise RuntimeError("Field preflight exceeds the declared VRAM budget")
    immutable_json(path, result)
    return result


def fit_paths(run: Path, arm: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    directory = run / "models" / f"seed-{seed}" / arm / f"fold-{fold}"
    return directory / "predictions.npz", directory / "checkpoint.pt", directory / "receipt.json"


def validated_fit_receipt_hashes(run: Path, seeds: tuple[int, ...]) -> dict[str, str]:
    """Validate every immutable fit/output triple without reading task labels."""

    result = {}
    for seed in seeds:
        for arm in ARMS:
            for fold in range(5):
                prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
                if not all(
                    path.is_file() for path in (prediction_path, checkpoint_path, receipt_path)
                ):
                    raise RuntimeError("Complete field fit inventory is missing")
                receipt = read_json(receipt_path)
                if (
                    receipt.get("status")
                    != "CORRESPONDENCE_FIELD_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
                    or receipt.get("predictions_sha256") != file_sha256(prediction_path)
                    or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
                    or receipt.get("outer_held_labels_read") != 0
                    or receipt.get("outer_metrics_computed") != 0
                ):
                    raise RuntimeError("Field fit/output triple changed")
                result[f"{arm}/{fold}/{seed}"] = file_sha256(receipt_path)
    return result


def _fit_request(
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    outer_train = np.flatnonzero(data["folds"] != fold)
    train = outer_train[data["labels"][outer_train] > 0]
    held = np.flatnonzero(data["folds"] == fold)
    if set(data["scenarios"][train]) & set(data["scenarios"][held]) or not np.array_equal(
        np.unique(data["labels"][train]), np.asarray([1, 2])
    ):
        raise RuntimeError("Outer field population is leaky or missing an upright class")
    request = {
        "execution_lock_sha256": lock_hash,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "optimizer_steps": protocol["training"]["optimizer_steps_per_fit"],
        "train_upright_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_upright_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "held_labels_sha256": None,
    }
    return request, train, held


def fit_one(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    store: FieldStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
    request, train, held = _fit_request(protocol, data, lock_hash, arm, fold, seed)
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("request") != request
            or receipt.get("predictions_sha256") != file_sha256(prediction_path)
            or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError("Completed field fit changed")
        return receipt
    destination = receipt_path.parent
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise RuntimeError("Partial field fit exists; use a fresh versioned run")
    seed_field_training(seed)
    device = torch.device("cuda")
    model = model_for(protocol, arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol["training"]["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    weights = torch.as_tensor(upright_class_weights(data["labels"], train), device=device)
    sampler = ActorUniformSampler(
        train,
        data["recordings"],
        data["tracks"],
        seed=seed + 10_000 * fold,
    )
    history, window_loss = [], 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, protocol["training"]["optimizer_steps_per_fit"] + 1):
        rows = sampler.batch(protocol["training"]["batch_size"])
        payload = to_device(store.batch(rows), device)
        target = torch.as_tensor(data["labels"][rows] - 1, dtype=torch.float32, device=device)
        output = model(**payload)
        element = F.binary_cross_entropy_with_logits(
            output["motion_logits"], target, reduction="none"
        )
        loss = (element * weights[target.long()]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Field-reader training loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            protocol["training"]["gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        window_loss += float(loss.detach().cpu())
        if step % protocol["training"]["log_interval_steps"] == 0:
            entry = {
                "step": step,
                "mean_training_loss": window_loss / protocol["training"]["log_interval_steps"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            window_loss = 0.0
            print(
                json.dumps(
                    {"event": "field_fit_progress", "arm": arm, "fold": fold, "seed": seed, **entry}
                ),
                flush=True,
            )
    model.eval()
    probabilities, logits, availability = [], [], []
    with torch.inference_mode():
        for left in range(0, len(held), protocol["training"]["batch_size"]):
            rows = held[left : left + protocol["training"]["batch_size"]]
            output = model(**to_device(store.batch(rows), device))
            probabilities.append(output["motion_probability"].cpu().numpy())
            logits.append(output["motion_logits"].cpu().numpy())
            availability.append(output["observation_available"].cpu().numpy())
    q = np.concatenate(probabilities).astype(np.float64)
    held_logits = np.concatenate(logits).astype(np.float64)
    observed = np.concatenate(availability).astype(bool)
    if (
        q.shape != (len(held),)
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
        or not np.array_equal(observed, store.pair_input_valid[held].any(1))
    ):
        raise RuntimeError("Held conditional-motion predictions are malformed")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    with (staging / "predictions.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=data["sample_ids"][held],
            held_rows=held,
            motion_probability=q,
            motion_logits=held_logits,
            observation_available=observed,
        )
    torch.save(
        {"model": model.state_dict(), "request": request, "capacity_report": model.capacity_report},
        staging / "checkpoint.pt",
    )
    receipt = {
        "status": "CORRESPONDENCE_FIELD_FIT_COMPLETE_OUTER_METRICS_EMBARGOED",
        "request": request,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "train_upright_rows": len(train),
        "held_rows": len(held),
        "training_actor_count": sampler.actor_count,
        "class_weights_standing_walking": weights.detach().cpu().tolist(),
        "optimizer_steps": protocol["training"]["optimizer_steps_per_fit"],
        "outer_held_labels_read": 0,
        "outer_metrics_computed": 0,
        "history": history,
        "capacity_report": model.capacity_report,
        "seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "predictions_sha256": file_sha256(staging / "predictions.npz"),
        "checkpoint_sha256": file_sha256(staging / "checkpoint.pt"),
    }
    immutable_json(staging / "receipt.json", receipt)
    validate_lock(run, protocol, data["sample_ids"], lock_hash)
    os.replace(staging, destination)
    print(
        json.dumps(
            {
                "event": "field_fit_complete_metrics_embargoed",
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


def _load_task_data(store: FieldStore) -> dict[str, np.ndarray]:
    data = load_cached_study_data(SOURCE)
    for required in ("recordings", "tracks"):
        if required not in data:
            raise RuntimeError(f"Task metadata is missing {required}")
    if not np.array_equal(store.sample_ids.astype(str), data["sample_ids"].astype(str)):
        raise RuntimeError("Observation cache and canonical task rows differ")
    return data


def fit_phase(run: Path, protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    store = FieldStore(ROOT / protocol["inputs"]["field_cache"])
    data = _load_task_data(store)
    lock_hash = prepare_lock(run, protocol, store.sample_ids)
    validate_lock(run, protocol, store.sample_ids, lock_hash)
    preflight_receipt = read_json(run / "preflight.json")
    if preflight_receipt.get("execution_lock_sha256") != lock_hash:
        raise RuntimeError("Run has no valid label-blind preflight")
    boundary = protocol["execution_boundary"]["maximum_unattended_expected_minutes"]
    estimated_minutes = preflight_receipt["estimated_initial_fit_minutes"] * (
        1 if phase == "initial" else 2
    )
    if estimated_minutes > boundary:
        raise RuntimeError(
            f"Estimated {phase} screen exceeds {boundary} minutes; command prepared but not started"
        )
    seeds = INITIAL_SEEDS if phase == "initial" else (43, 44)
    if phase == "confirmation":
        initial = summarize(run, protocol, "initial")
        if not initial.get("stage1_gate_passed"):
            raise RuntimeError(
                "Confirmation fits are forbidden because the initial screen did not pass"
            )
    for seed in seeds:
        for arm in ARMS:
            for fold in range(5):
                fit_one(run, protocol, data, store, lock_hash, arm, fold, seed)
    return {
        "status": "CORRESPONDENCE_FIELD_FIT_INVENTORY_COMPLETE_METRICS_STILL_EMBARGOED",
        "phase": phase,
        "seeds": list(seeds),
        "fits_completed": len(seeds) * len(ARMS) * 5,
        "next_command": (
            ".\\.venv\\Scripts\\python.exe experiments/audit_okutama_correspondence_field.py "
            f"--phase {phase} --run {run.relative_to(ROOT).as_posix()}"
        ),
    }


def _anchor(protocol: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    path = ROOT / protocol["inputs"]["arftr_oof"]
    with np.load(path, allow_pickle=False) as saved:
        arms = saved["arms"].tolist()
        index = arms.index("r5_arftr_full")
        if (
            not np.array_equal(saved["sample_ids"], data["sample_ids"])
            or not np.array_equal(saved["labels"], data["labels"])
            or not np.array_equal(saved["folds"], data["folds"])
        ):
            raise RuntimeError("Retained ARFTR identity changed")
        anchor = saved["mean_probabilities"][index].copy()
    return anchor


def _transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    before = anchor.argmax(1) == labels
    after = candidate.argmax(1) == labels
    return {
        "rescues": int((after & ~before).sum()),
        "harms": int((~after & before).sum()),
        "net_corrections": int((after & ~before).sum() - (~after & before).sum()),
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def _historical_unresolved_ids(protocol: dict[str, Any]) -> set[str]:
    path = ROOT / protocol["inputs"]["historical_residual_diagnostic"]
    result = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["combined_bank_unresolved_diagnostic_only"].lower() == "true":
                result.add(row["sample_id"])
    if len(result) != 304:
        raise RuntimeError("Historical unresolved diagnostic population changed")
    return result


def summarize(run: Path, protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    seeds = INITIAL_SEEDS if phase == "initial" else CONFIRMATION_SEEDS
    version = "v0001" if phase == "initial" else "v0002"
    result_dir = run / "results" / version
    summary_path = result_dir / "summary.json"
    existing_summary = summary_path.is_file()
    if result_dir.exists() and any(result_dir.iterdir()) and not existing_summary:
        raise RuntimeError("Partial field summary exists; use a fresh versioned run")
    store = FieldStore(ROOT / protocol["inputs"]["field_cache"])
    data = _load_task_data(store)
    lock_hash = prepare_lock(run, protocol, store.sample_ids)
    validate_lock(run, protocol, store.sample_ids, lock_hash)
    audit_path = run / "audit" / phase / "summary.json"
    audit = read_json(audit_path)
    if (
        audit.get("status") != "CORRESPONDENCE_FIELD_INDEPENDENT_AUDIT_PASSED"
        or audit.get("execution_lock_sha256") != lock_hash
        or tuple(audit.get("seeds", ())) != seeds
    ):
        raise RuntimeError("Independent correspondence-field audit is absent or failed")
    diagnostic_path = run / "audit" / phase / "diagnostic_probabilities.npz"
    if audit.get("diagnostic_probabilities_sha256") != file_sha256(diagnostic_path):
        raise RuntimeError("Independent structure-diagnostic probabilities changed")
    audited_receipts = validated_fit_receipt_hashes(run, seeds)
    if audit.get("fit_receipt_sha256") != audited_receipts:
        raise RuntimeError("Independent audit does not certify the current fit/output inventory")
    if existing_summary:
        existing = read_json(summary_path)
        oof_path = result_dir / "oof_probabilities.npz"
        if (
            existing.get("execution_lock_sha256") != lock_hash
            or existing.get("independent_audit_sha256") != file_sha256(audit_path)
            or existing.get("oof_probabilities_sha256") != file_sha256(oof_path)
        ):
            raise RuntimeError("Completed field summary or its ancestors changed")
        return existing
    q = np.full((len(ARMS), len(seeds), len(data["labels"])), np.nan, dtype=np.float64)
    observed = np.full((len(ARMS), len(seeds), len(data["labels"])), False)
    receipt_hashes = {}
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(seeds):
            for fold in range(5):
                request, _, held = _fit_request(protocol, data, lock_hash, arm, fold, seed)
                prediction_path, checkpoint_path, receipt_path = fit_paths(run, arm, fold, seed)
                receipt = read_json(receipt_path)
                if (
                    receipt.get("status")
                    != "CORRESPONDENCE_FIELD_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
                    or receipt.get("request") != request
                    or receipt.get("outer_held_labels_read") != 0
                    or receipt.get("outer_metrics_computed") != 0
                    or receipt.get("optimizer_steps")
                    != protocol["training"]["optimizer_steps_per_fit"]
                    or receipt.get("predictions_sha256") != file_sha256(prediction_path)
                    or receipt.get("checkpoint_sha256") != file_sha256(checkpoint_path)
                ):
                    raise RuntimeError("Field fit receipt or ancestry changed")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["held_rows"], held) or not np.array_equal(
                        saved["sample_ids"], data["sample_ids"][held]
                    ):
                        raise RuntimeError("Held field prediction identity changed")
                    q[arm_index, seed_index, held] = saved["motion_probability"]
                    observed[arm_index, seed_index, held] = saved["observation_available"]
                receipt_hashes[f"{arm}/{fold}/{seed}"] = file_sha256(receipt_path)
    if not np.isfinite(q).all():
        raise RuntimeError("Field OOF prediction coverage is incomplete")
    if not np.all(observed == observed[0, 0]):
        raise RuntimeError("Observation availability changed across fits")
    if receipt_hashes != audit.get("fit_receipt_sha256"):
        raise RuntimeError("Independent audit does not certify the current fit receipts")
    mean_q = q.mean(1)
    with np.load(diagnostic_path, allow_pickle=False) as diagnostic:
        if (
            not np.array_equal(diagnostic["sample_ids"], data["sample_ids"])
            or not np.array_equal(diagnostic["seeds"], np.asarray(seeds))
            or diagnostic["motion_probabilities"].shape != (2, len(seeds), len(data["labels"]))
            or not np.array_equal(
                diagnostic["diagnostics"],
                np.asarray(["spatial_reassignment", "timestamp_reassignment"]),
            )
        ):
            raise RuntimeError("Fixed structure-diagnostic identity changed")
        diagnostic_q = diagnostic["motion_probabilities"].mean(1)
    if not np.isfinite(diagnostic_q).all():
        raise RuntimeError("Fixed structure-diagnostic probabilities are incomplete")
    anchor = _anchor(protocol, data)
    candidates, eligible = [], []
    for arm_index in range(len(ARMS)):
        candidate, changed = conditional_motion_candidate(anchor, mean_q[arm_index], observed[0, 0])
        candidates.append(candidate)
        eligible.append(changed)
    candidates = np.asarray(candidates)
    eligible = np.asarray(eligible)
    balanced = {
        arm: upright_balanced_accuracy(data["labels"], mean_q[index])
        for index, arm in enumerate(ARMS)
    }
    fold_balanced = {
        arm: [
            upright_balanced_accuracy(
                data["labels"][data["folds"] == fold], mean_q[index, data["folds"] == fold]
            )
            for fold in range(5)
        ]
        for index, arm in enumerate(ARMS)
    }
    metrics = {
        arm: probability_metrics(data["labels"], candidates[index])
        for index, arm in enumerate(ARMS)
    }
    transitions = {
        arm: _transitions(data["labels"], candidates[index], anchor)
        for index, arm in enumerate(ARMS)
    }
    fold_transitions = {
        arm: [
            _transitions(
                data["labels"][data["folds"] == fold],
                candidates[index, data["folds"] == fold],
                anchor[data["folds"] == fold],
            )
            for fold in range(5)
        ]
        for index, arm in enumerate(ARMS)
    }
    unresolved_ids = _historical_unresolved_ids(protocol)
    unresolved = np.isin(data["sample_ids"].astype(str), list(unresolved_ids))
    before, after = anchor.argmax(1) == data["labels"], candidates[2].argmax(1) == data["labels"]
    unresolved_rescues = int((unresolved & ~before & after).sum())
    c2_c0 = 100 * (balanced["c2"] - balanced["c0"])
    c2_c1 = 100 * (balanced["c2"] - balanced["c1"])
    structure_balanced = {
        "c2_spatial_reassignment": upright_balanced_accuracy(data["labels"], diagnostic_q[0]),
        "c2_timestamp_reassignment": upright_balanced_accuracy(data["labels"], diagnostic_q[1]),
    }
    fold_c2_c0 = np.asarray(fold_balanced["c2"]) - np.asarray(fold_balanced["c0"])
    fold_c2_c1 = np.asarray(fold_balanced["c2"]) - np.asarray(fold_balanced["c1"])
    gates = protocol["stage1_gates"]
    checks = {
        "c2_minus_c0_upright_balanced_accuracy": c2_c0
        >= gates["c2_minus_c0_upright_balanced_accuracy_points"],
        "c2_minus_c0_nonnegative_folds": int((fold_c2_c0 >= 0).sum())
        >= gates["c2_minus_c0_nonnegative_folds"],
        "c2_minus_c1_upright_balanced_accuracy": c2_c1
        >= gates["c2_minus_c1_upright_balanced_accuracy_points"],
        "c2_minus_c1_positive_folds": int((fold_c2_c1 > 0).sum())
        >= gates["c2_minus_c1_positive_folds"],
        "minimum_c2_arftr_rescues": transitions["c2"]["rescues"]
        >= gates["minimum_c2_arftr_rescues"],
        "minimum_c2_rescues_per_fold": min(item["rescues"] for item in fold_transitions["c2"])
        >= gates["minimum_c2_rescues_per_fold"],
        "minimum_c2_rescues_in_historical_unresolved_304": unresolved_rescues
        >= gates["minimum_c2_rescues_in_historical_unresolved_304"],
        "independent_integrity_audit": True,
    }
    result_dir.mkdir(parents=True)
    oof_path = result_dir / "oof_probabilities.npz"
    with oof_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=data["sample_ids"],
            labels=data["labels"],
            scenarios=data["scenarios"],
            folds=data["folds"],
            arms=np.asarray(ARMS),
            seeds=np.asarray(seeds),
            seed_motion_probabilities=q,
            mean_motion_probabilities=mean_q,
            observation_available=observed[0, 0],
            candidate_probabilities=candidates,
            intervention_eligible=eligible,
            arftr_probabilities=anchor,
        )
    summary = {
        "status": "CORRESPONDENCE_FIELD_STAGE1_SCREEN_COMPLETE",
        "phase": phase,
        "adaptive_repeated_development": True,
        "study_id": protocol["study_id"],
        "execution_lock_sha256": lock_hash,
        "seeds": list(seeds),
        "model_fits_total": len(seeds) * len(ARMS) * 5,
        "fit_receipt_sha256": receipt_hashes,
        "independent_audit_sha256": file_sha256(audit_path),
        "oof_probabilities_sha256": file_sha256(oof_path),
        "upright_balanced_accuracy": balanced,
        "fold_upright_balanced_accuracy": fold_balanced,
        "candidate_metrics": metrics,
        "arftr_metrics": probability_metrics(data["labels"], anchor),
        "transitions_vs_arftr": transitions,
        "fold_transitions_vs_arftr": fold_transitions,
        "primary_effects": {
            "c2_minus_c0_upright_balanced_accuracy_points": c2_c0,
            "c2_minus_c1_upright_balanced_accuracy_points": c2_c1,
            "c2_minus_c0_fold_points": (100 * fold_c2_c0).tolist(),
            "c2_minus_c1_fold_points": (100 * fold_c2_c1).tolist(),
            "c2_rescues_in_historical_unresolved_304": unresolved_rescues,
        },
        "fixed_structure_diagnostics": {
            "upright_balanced_accuracy": structure_balanced,
            "spatial_reassignment_minus_c0_points": 100
            * (structure_balanced["c2_spatial_reassignment"] - balanced["c0"]),
            "timestamp_reassignment_minus_c1_points": 100
            * (structure_balanced["c2_timestamp_reassignment"] - balanced["c1"]),
            "label_blind_probability_audit": audit["fixed_structure_diagnostics"],
            "interpretation_rule": (
                "If C2's advantage survives either corresponding structure-destroying "
                "diagnostic unchanged, do not attribute that advantage to the destroyed structure."
            ),
        },
        "stage1_checks": {name: bool(value) for name, value in checks.items()},
        "stage1_gate_passed": bool(all(checks.values())),
        "stage2_status": "AUTHORIZED_TO_PLAN_ONLY" if all(checks.values()) else "FORBIDDEN",
        "limitations": (
            "This is an adaptive repeated-development representation screen. The fixed candidate is "
            "diagnostic and is not a deployable intervention policy. Historical unresolved membership "
            "is post-hoc evaluation only."
        ),
    }
    immutable_json(summary_path, summary)
    validate_lock(run, protocol, store.sample_ids, lock_hash)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "preflight", "fit", "summarize"), required=True)
    parser.add_argument("--phase", choices=("initial", "confirmation"), default="initial")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    run = args.run.resolve()
    run.relative_to(ROOT.resolve())
    if args.stage == "plan":
        result = metadata_plan(protocol, run)
    else:
        store = FieldStore(ROOT / protocol["inputs"]["field_cache"])
        lock_hash = prepare_lock(run, protocol, store.sample_ids)
        validate_lock(run, protocol, store.sample_ids, lock_hash)
        if args.stage == "preflight":
            result = preflight(run, protocol, store, lock_hash)
        elif args.stage == "fit":
            result = fit_phase(run, protocol, args.phase)
        else:
            result = summarize(run, protocol, args.phase)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
