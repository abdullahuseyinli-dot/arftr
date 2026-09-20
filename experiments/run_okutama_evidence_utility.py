"""Fit the locked four-arm aligned-evidence utility screen after cache completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Evidence utility training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.actor_memory_base import file_sha256
from hac.evidence_decomposition import (
    ARM_NAMES,
    READER_INPUT_DIM,
    EvidenceUtilityReader,
    apply_standardizer,
    decode_factor_logits,
    fit_standardizer,
    weighted_factor_nll,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_evidence_utility_protocol.json"
TASK_DATA = ROOT / ".runs/research_20260908/source_swap_v1/data/memory_data.npz"
TASK_DATA_SHA256 = "0bfadd5f080b162ab4ccfce76affbde4779d9fb61757124029b159adec4079e9"
DEFAULT_CACHE = ROOT / ".runs/research_20260913/evidence_utility_cache_v2"
DEFAULT_RUN = ROOT / ".runs/research_20260913/evidence_utility_screen_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("resource-gate", "prepare", "initial"), default="prepare"
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype.str}|{list(value.shape)}".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    reader, observations, metrics = (
        protocol["reader"],
        protocol["observations"],
        protocol["metrics"],
    )
    if (
        protocol.get("schema_version") != 1
        or protocol.get("study_id") != "HAC_ALIGNED_EVIDENCE_ACTION_BRIDGE_V1"
        or tuple(protocol["decomposition"]["arms"]) != ARM_NAMES
        or protocol["population"]["rows"] != 4977
        or tuple(protocol["population"]["folds"]) != tuple(range(5))
        or protocol["population"].get("task_data_sha256") != TASK_DATA_SHA256
        or observations["shard_size_max"] != 128
        or protocol["decomposition"]["input_dimension"] != READER_INPUT_DIM
        or reader["nominal_parameters"] != 737858
        or reader["batch_size"] != 64
        or reader["steps"] != 400
        or reader["seed"] != 42
        or reader["initial_fits"] != 20
        or reader["loss"] != "mean(class_weight[label] * stable_factor_NLL_per_row)"
        or "fixed classes [0,1,2]" not in metrics["macro_f1"]
        or protocol["evaluation_boundary"]["ARFTR and diagnostic support"].split()[0] != "read"
    ):
        raise RuntimeError("Evidence utility execution protocol changed")


def load_task_metadata() -> dict[str, np.ndarray]:
    """Task-stage allowlist; deliberately excludes support/purity/boundary members."""

    if file_sha256(TASK_DATA) != TASK_DATA_SHA256:
        raise RuntimeError("Pinned task metadata bytes changed")
    with np.load(TASK_DATA, allow_pickle=False) as saved:
        result = {
            name: saved[name]
            for name in (
                "sample_ids",
                "labels",
                "scenarios",
                "folds",
                "recordings",
                "tracks",
                "frames",
            )
        }
    rows = 4977
    if (
        any(value.shape != (rows,) for value in result.values())
        or len(np.unique(result["sample_ids"])) != rows
        or not np.array_equal(np.unique(result["labels"]), np.arange(3))
        or not np.array_equal(np.unique(result["folds"]), np.arange(5))
        or len(np.unique(result["scenarios"])) != 11
        or any(
            set(result["scenarios"][result["folds"] == fold])
            & set(result["scenarios"][result["folds"] != fold])
            for fold in range(5)
        )
    ):
        raise RuntimeError("Task metadata population/split contract changed")
    return result


class EvidenceStore:
    def __init__(self, cache_root: Path, expected_ids: np.ndarray) -> None:
        directories = [path for path in cache_root.glob("shard-*") if path.is_dir()]
        records = []
        for directory in directories:
            request, summary = (
                read_json(directory / "request.json"),
                read_json(directory / "summary.json"),
            )
            feature_path = directory / "features.npz"
            if (
                request.get("mode") != "shard"
                or summary.get("status") != "EVIDENCE_UTILITY_OBSERVATION_SHARD_COMPLETE"
                or summary.get("task_labels_read") != 0
                or summary.get("ARFTR_outputs_read") != 0
                or summary.get("artifacts_before_summary", {}).get("features.npz", {}).get("sha256")
                != file_sha256(feature_path)
            ):
                raise RuntimeError(f"Incomplete or impure observation shard: {directory}")
            records.append((int(request["start"]), directory, request, summary))
        records.sort()
        if not records or records[0][0] != 0:
            raise RuntimeError("Evidence shards do not start at row zero")
        expected_start = 0
        common_source_files = None
        common_model_snapshot = None
        common_context_audit = None
        arrays: dict[str, list[np.ndarray]] = {
            name: [] for name in ("sample_ids", "context", "center_bins", "contrasts", "coverage")
        }
        self.receipts = []
        for start, directory, request, summary in records:
            if start != expected_start:
                raise RuntimeError("Evidence shards overlap or leave a gap")
            with np.load(directory / "features.npz", allow_pickle=False) as saved:
                expected_members = {
                    "sample_ids",
                    "arm_names",
                    "context",
                    "center_bins",
                    "contrasts",
                    "coverage",
                    "transform_matrix",
                    "transform_valid",
                    "match_count",
                    "weighted_residual",
                    "common_target_count",
                    "roi_target_count",
                }
                if (
                    set(saved.files) != expected_members
                    or tuple(saved["arm_names"].tolist()) != ARM_NAMES
                ):
                    raise RuntimeError(
                        "Full shard contains duplicated arm inputs or changed arm order"
                    )
                values = {name: saved[name] for name in arrays}
            count = len(values["sample_ids"])
            expected_shard_ids = expected_ids[expected_start : expected_start + count]
            if (
                not np.array_equal(values["sample_ids"].astype(str), expected_shard_ids.astype(str))
                or request.get("sample_ids") != expected_shard_ids.tolist()
                or request.get("sample_ids_sha256") != array_sha256(expected_shard_ids)
                or summary.get("reader_input_shape") != [count, 4, READER_INPUT_DIM]
                or summary.get("B0_contrast_exact_zero") is not True
            ):
                raise RuntimeError("Shard sample/input identity changed")
            if common_source_files is None:
                common_source_files = request.get("source_files")
                common_model_snapshot = request.get("model_snapshot")
                common_context_audit = summary.get("context_audit")
            elif (
                request.get("source_files") != common_source_files
                or request.get("model_snapshot") != common_model_snapshot
                or summary.get("context_audit") != common_context_audit
            ):
                raise RuntimeError("Observation source/code/context ancestry differs across shards")
            for relative, record in request["source_files"].items():
                source = ROOT / relative
                if (
                    not source.is_file()
                    or source.stat().st_size != record["bytes"]
                    or file_sha256(source) != record["sha256"]
                ):
                    raise RuntimeError(f"Locked observation source changed: {relative}")
            for name in ("request.json", "source_receipts.json", "features.npz"):
                artifact = summary.get("artifacts_before_summary", {}).get(name)
                if artifact is None or file_sha256(directory / name) != artifact.get("sha256"):
                    raise RuntimeError(f"Shard artifact changed: {directory}/{name}")
            expected_start += count
            if request.get("rows") != count or summary.get("rows") != count:
                raise RuntimeError("Shard row count differs from its receipt")
            for name, value in values.items():
                arrays[name].append(value)
            self.receipts.append(
                {
                    "directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
                    "features_sha256": file_sha256(directory / "features.npz"),
                    "request_sha256": file_sha256(directory / "request.json"),
                    "summary_sha256": file_sha256(directory / "summary.json"),
                }
            )
        if expected_start != len(expected_ids):
            raise RuntimeError("Evidence shards do not cover the full task cohort")
        for name, parts in arrays.items():
            setattr(self, name, np.concatenate(parts))
        if (
            not np.array_equal(self.sample_ids.astype(str), expected_ids.astype(str))
            or self.context.shape != (4977, 2304)
            or self.center_bins.shape != (4977, 6, 768)
            or self.contrasts.shape != (4977, 4, 6, 768)
            or self.coverage.shape != (4977, 6)
            or any(
                not np.isfinite(value).all()
                for value in (self.context, self.center_bins, self.contrasts, self.coverage)
            )
            or np.any(self.contrasts[:, 0] != 0)
            or np.any((self.coverage < 0) | (self.coverage > 1))
        ):
            raise RuntimeError("Assembled evidence cache violates identity/shape/value contracts")

    def arm(self, arm: int) -> np.ndarray:
        if arm not in range(len(ARM_NAMES)):
            raise ValueError("Unknown evidence arm")
        rows = len(self.context)
        return np.concatenate(
            (
                self.context.astype(np.float32, copy=False),
                self.center_bins.reshape(rows, -1).astype(np.float32, copy=False),
                self.contrasts[:, arm].reshape(rows, -1).astype(np.float32, copy=False),
                self.coverage.astype(np.float32, copy=False),
            ),
            axis=1,
        )


def batch_schedule(train_rows: np.ndarray, *, steps: int, batch_size: int, seed: int) -> np.ndarray:
    train = np.asarray(train_rows, dtype=np.int64)
    if not np.array_equal(train, np.unique(train)) or len(train) < batch_size:
        raise ValueError("Training rows must be sorted, unique and at least one batch")
    rng = np.random.default_rng(seed)
    needed, chunks = steps * batch_size, []
    while sum(len(chunk) for chunk in chunks) < needed:
        chunks.append(rng.permutation(train))
    return np.concatenate(chunks)[:needed].reshape(steps, batch_size)


def class_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    count = np.bincount(labels, minlength=3)
    if count.shape != (3,) or (count == 0).any() or not np.isin(labels, range(3)).all():
        raise RuntimeError("Outer training partition is missing a class")
    return (len(labels) / (3 * count)).astype(np.float32)


def seed_training(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(array_sha256(array).encode())
    return digest.hexdigest()


def fit_reader(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    held_features: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Fit from training rows only; held labels are structurally unaccepted."""

    train_features, held_features = np.asarray(train_features), np.asarray(held_features)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if (
        train_features.ndim != 2
        or train_features.shape[1] != READER_INPUT_DIM
        or held_features.ndim != 2
        or held_features.shape[1] != READER_INPUT_DIM
        or train_labels.shape != (len(train_features),)
    ):
        raise ValueError("Reader fit arrays are malformed")
    mean, scale = fit_standardizer(train_features)
    train_x = apply_standardizer(train_features, mean, scale)
    held_x = apply_standardizer(held_features, mean, scale)
    weights = class_weights(train_labels)
    schedule = batch_schedule(
        np.arange(len(train_x)), steps=steps, batch_size=batch_size, seed=seed
    )
    seed_training(seed)
    model = EvidenceUtilityReader().to(device)
    if model.trainable_parameters != 737858:
        raise RuntimeError("Evidence reader parameter count changed")
    initial_state_sha256 = model_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weight_tensor = torch.as_tensor(weights, device=device)
    history = []
    started = time.perf_counter()
    model.train()
    for step, rows in enumerate(schedule):
        logits = model(torch.as_tensor(train_x[rows], device=device))
        loss = weighted_factor_nll(
            logits, torch.as_tensor(train_labels[rows], device=device), weight_tensor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Evidence reader optimization became nonfinite")
        if step in {0, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm.detach().cpu()),
                }
            )
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(held_x), 256):
            parts.append(
                decode_factor_logits(
                    model(torch.as_tensor(held_x[start : start + 256], device=device))
                )
                .cpu()
                .numpy()
            )
    probabilities = np.concatenate(parts).astype(np.float64)
    if (
        probabilities.shape != (len(held_x), 3)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-6)
    ):
        raise RuntimeError("Reader prediction simplex changed")
    checkpoint = {"scaler_mean": mean, "scaler_scale": scale, "class_weight": weights}
    checkpoint.update(
        {
            f"state__{name}": value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        }
    )
    return (
        checkpoint,
        probabilities,
        {
            "history": history,
            "seconds": time.perf_counter() - started,
            "train_rows": len(train_x),
            "held_row_count": len(held_x),
            "optimizer_updates": steps,
            "schedule_sha256": array_sha256(schedule),
            "initial_state_sha256": initial_state_sha256,
        },
    )


def resource_gate() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Evidence utility screen requires CUDA")
    seed_training(42)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = EvidenceUtilityReader().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    features = torch.randn(64, READER_INPUT_DIM, device=device)
    labels = torch.arange(64, device=device).remainder(3).long()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(400):
        logits = model(features)
        loss = weighted_factor_nll(logits, labels, torch.ones(3, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "status": "EVIDENCE_UTILITY_RESOURCE_GATE_PASS",
        "device": torch.cuda.get_device_name(),
        "parameters": model.trainable_parameters,
        "batch_size": 64,
        "optimizer_updates": 400,
        "optimizer_seconds": seconds,
        "linear_projection_20_fit_seconds": 20 * seconds,
        "loss": float(loss.detach().cpu()),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if not np.isfinite(result["loss"]):
        raise RuntimeError("Evidence utility resource gate produced a nonfinite loss")
    return result


def prepare(
    cache_root: Path, run: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], EvidenceStore, str]:
    protocol = read_json(PROTOCOL)
    validate_protocol(protocol)
    data = load_task_metadata()
    cache_root = cache_root.resolve()
    cache_root.relative_to(ROOT.resolve())
    store = EvidenceStore(cache_root, data["sample_ids"])
    run = run.resolve()
    run.relative_to(ROOT.resolve())
    if run.exists() and any(run.iterdir()) and not (run / "execution_lock.json").is_file():
        raise RuntimeError("Refusing a nonempty run without an execution lock")
    run.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "EVIDENCE_UTILITY_INITIAL_EXECUTION_LOCKED",
        "study_id": protocol["study_id"],
        "run_directory": str(run),
        "rows": 4977,
        "arms": list(ARM_NAMES),
        "fits": 20,
        "sample_ids_sha256": array_sha256(data["sample_ids"]),
        "split_sha256": array_sha256(
            np.column_stack((data["scenarios"], data["folds"].astype(str)))
        ),
        "task_members_read": [
            "sample_ids",
            "labels",
            "scenarios",
            "folds",
            "recordings",
            "tracks",
            "frames",
        ],
        "support_members_read": [],
        "ARFTR_outputs_read": [],
        "input_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (
                PROTOCOL,
                Path(__file__),
                ROOT / "experiments/audit_okutama_evidence_utility.py",
                ROOT / "experiments/audit_okutama_evidence_utility_cache.py",
                ROOT / "src/hac/evidence_decomposition.py",
                TASK_DATA,
            )
        },
        "observation_shards": store.receipts,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }
    lock_path = run / "execution_lock.json"
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise RuntimeError("Evidence execution lock changed")
    else:
        write_json_exclusive(lock_path, lock)
    return protocol, data, store, file_sha256(lock_path)


def run_initial(cache_root: Path, run: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Evidence utility screen requires CUDA")
    protocol, data, store, lock_sha = prepare(cache_root, run)
    config = protocol["reader"]
    inventory = []
    total_started = time.perf_counter()
    for arm_index, arm in enumerate(ARM_NAMES):
        features = store.arm(arm_index)
        for fold in range(5):
            train, held = (
                np.flatnonzero(data["folds"] != fold),
                np.flatnonzero(data["folds"] == fold),
            )
            directory = run / "models" / arm / f"fold-{fold}" / "seed-42"
            if directory.exists():
                raise FileExistsError(f"Fit output already exists: {directory}")
            directory.mkdir(parents=True)
            checkpoint, probabilities, receipt = fit_reader(
                features[train],
                data["labels"][train],
                features[held],
                steps=config["steps"],
                batch_size=config["batch_size"],
                seed=config["seed"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                gradient_clip_norm=config["gradient_clip_norm"],
                device=torch.device("cuda"),
            )
            with (directory / "checkpoint.npz").open("xb") as stream:
                np.savez_compressed(stream, **checkpoint)
            with (directory / "predictions.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    sample_ids=data["sample_ids"][held],
                    row_indices=held,
                    probabilities=probabilities,
                )
            fit_receipt = {
                "status": "EVIDENCE_UTILITY_FIT_COMPLETE_METRICS_EMBARGOED",
                "arm": arm,
                "fold": fold,
                "seed": 42,
                "execution_lock_sha256": lock_sha,
                "train_ids_sha256": array_sha256(data["sample_ids"][train]),
                "held_ids_sha256": array_sha256(data["sample_ids"][held]),
                "held_rows": held.tolist(),
                "held_sample_ids": data["sample_ids"][held].tolist(),
                "held_labels_passed_to_fit": 0,
                **receipt,
                "checkpoint_sha256": file_sha256(directory / "checkpoint.npz"),
                "predictions_sha256": file_sha256(directory / "predictions.npz"),
            }
            write_json_exclusive(directory / "receipt.json", fit_receipt)
            inventory.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "directory": str(directory.relative_to(run)).replace("\\", "/"),
                    "receipt_sha256": file_sha256(directory / "receipt.json"),
                }
            )
        del features
    summary = {
        "status": "EVIDENCE_UTILITY_INITIAL_20_FITS_COMPLETE_METRICS_EMBARGOED",
        "fits": len(inventory),
        "inventory": inventory,
        "execution_lock_sha256": lock_sha,
        "seconds": time.perf_counter() - total_started,
        "held_labels_passed_to_fit": 0,
        "ARFTR_outputs_read": 0,
    }
    if len(inventory) != 20:
        raise RuntimeError("Initial screen fit inventory is incomplete")
    write_json_exclusive(run / "fit_inventory.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "resource-gate":
        print(json.dumps(resource_gate(), indent=2), flush=True)
    elif args.mode == "prepare":
        protocol, data, store, lock_sha = prepare(args.cache_root, args.run_dir)
        print(
            json.dumps(
                {
                    "status": "EVIDENCE_UTILITY_RUN_PREPARED",
                    "rows": len(data["labels"]),
                    "arms": list(ARM_NAMES),
                    "fits": protocol["reader"]["initial_fits"],
                    "execution_lock_sha256": lock_sha,
                    "shards": len(store.receipts),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(json.dumps(run_initial(args.cache_root, args.run_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
