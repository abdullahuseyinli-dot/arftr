"""Train the matched center-only/all-cached-frame FSAR comparison."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("FSAR requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.nn import functional as F

from hac.actor_memory_base import canonical_hash, file_sha256, probability_metrics
from hac.cached_frame_supervision import (
    CENTER_INDEX,
    SOURCE_PATHS,
    load_cached_frame_supervision,
    validate_cached_frame_supervision_receipt,
)
from hac.frame_supervised_residual import (
    FrameSupervisedAnchorResidual,
    anchored_probabilities,
    class_weights,
    frame_residual_loss,
    seed_frame_training,
)
from hac.matr import geometric_residual
from hac.matr_artifacts import (
    load_cached_study_data,
    selected_fold_artifacts,
    selected_input_paths,
)
from hac.source_swap_data import immutable_json, read_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".runs/research_20260908/source_swap_v1"
SEAR = ROOT / ".runs/research_20260908/sear_matrix_v1"
FRAME_DATA = ROOT / ".runs/research_20260912/cached_frame_supervision_v2/data"
PROTOCOL = ROOT / "experiments/okutama_cached_frame_supervision_protocol.json"
DEFAULT_RUN = ROOT / ".runs/research_20260912/frame_supervised_anchor_residual_v2"
TRAINED_ARMS = ("f2_center_only_supervision", "f3_all_cached_frame_supervision")
ARMS = (
    "f0_exact_m4",
    "f1_fixed_geometric_m4_a3",
    *TRAINED_ARMS,
    "f4_center_only_standalone",
    "f5_all_cached_frame_standalone",
)
SEEDS = (42, 43, 44)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _save_or_validate_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[name], value) for name, value in arrays.items()
            ):
                raise RuntimeError(f"Existing FSAR artifact differs from reconstructed arrays: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_protocol(protocol: dict[str, Any], data: dict[str, np.ndarray]) -> None:
    architecture, training = protocol["architecture"], protocol["training"]
    statistics = protocol["statistics"]
    if (
        protocol.get("study_id") != "okutama_cached_frame_supervised_anchor_residual_v2"
        or protocol.get("primary_arm") != "f3_all_cached_frame_supervision"
        or set(protocol.get("arms", {})) != set(ARMS)
        or tuple(training.get("outer_folds", ())) != tuple(range(5))
        or tuple(training.get("outer_seeds", ())) != SEEDS
        or training.get("model_fits") != len(TRAINED_ARMS) * 5 * len(SEEDS)
        or training.get("epochs") != 8
        or training.get("batch_size") != 2048
        or training.get("center_loss_weight") != 1.0
        or training.get("auxiliary_loss_weight") != 0.5
        or training.get("anchor_kl_weight") != 0.1
        or training.get("required_device") != "cuda"
        or training.get("cublas_workspace_config") != ":4096:8"
        or architecture.get("input_dim") != 768
        or architecture.get("classes") != 3
        or architecture.get("residual_scale") != 0.25
        or architecture.get("a3_anchor_weight") != 0.10
        or protocol["frame_supervision"].get("center_index") != CENTER_INDEX
        or statistics.get("exact_scenario_swaps") != 2 ** len(np.unique(data["scenarios"]))
        or statistics.get("one_sided_alpha") != 0.05
        or statistics.get("primary_contrasts")
        != [
            ["f3_all_cached_frame_supervision", "f0_exact_m4"],
            ["f3_all_cached_frame_supervision", "f2_center_only_supervision"],
        ]
    ):
        raise RuntimeError("FSAR v2 protocol contract changed")


def prepare(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], str]:
    protocol = read_json(PROTOCOL)
    data = load_cached_study_data(SOURCE)
    frames, frame_receipt = validate_cached_frame_supervision_receipt(ROOT, FRAME_DATA)
    _validate_protocol(protocol, data)
    for name in ("sample_ids", "labels", "scenarios", "folds"):
        if not np.array_equal(frames[name], data[name]):
            raise RuntimeError(f"Frame-supervision identity differs from source data: {name}")
    for fold in range(5):
        selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    paths = set(selected_input_paths(ROOT, SOURCE, SEAR, data))
    paths.update(
        {
            PROTOCOL,
            Path(__file__),
            ROOT / "src/hac/frame_supervised_residual.py",
            ROOT / "src/hac/cached_frame_supervision.py",
            ROOT / "src/hac/matr.py",
            ROOT / "src/hac/matr_artifacts.py",
            ROOT / "src/hac/actor_memory_base.py",
            ROOT / "src/hac/source_swap_data.py",
            ROOT / "experiments/audit_okutama_frame_supervision.py",
            FRAME_DATA / "frame_supervision.npz",
            FRAME_DATA / "receipt.json",
            *(ROOT / value for value in SOURCE_PATHS.values()),
            *(Path(record["path"]) for record in frame_receipt["source_code"].values()),
        }
    )
    lock = {
        "status": "FSAR_EXECUTION_LOCKED_BEFORE_MODEL_FITTING",
        "study_id": protocol["study_id"],
        "scope": protocol["scope"],
        "primary_arm": protocol["primary_arm"],
        "rows": len(data["labels"]),
        "model_fits": protocol["training"]["model_fits"],
        "run_directory": str(run.resolve()),
        "sample_ids_sha256": canonical_hash(data["sample_ids"].tolist()),
        "input_files": [
            {
                "path": _relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(paths, key=str)
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }
    run.mkdir(parents=True, exist_ok=True)
    immutable_json(run / "execution_lock.json", lock)
    return protocol, data, frames, file_sha256(run / "execution_lock.json")


def _validate_lock(run: Path, data: dict[str, np.ndarray], lock_hash: str) -> None:
    path = run / "execution_lock.json"
    if file_sha256(path) != lock_hash:
        raise RuntimeError("FSAR execution lock changed")
    lock = read_json(path)
    if lock["sample_ids_sha256"] != canonical_hash(data["sample_ids"].tolist()):
        raise RuntimeError("FSAR sample identity changed")
    if Path(lock["run_directory"]).resolve() != run.resolve():
        raise RuntimeError("FSAR execution lock moved to a different run directory")
    expected_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    if lock["environment"] != expected_environment:
        raise RuntimeError("FSAR locked runtime environment changed")
    for record in lock["input_files"]:
        input_path = (ROOT / record["path"]).resolve()
        if (
            not input_path.is_file()
            or input_path.stat().st_size != record["size_bytes"]
            or file_sha256(input_path) != record["sha256"]
        ):
            raise RuntimeError(f"Locked FSAR input changed: {record['path']}")


class FrameFeatureStore:
    def __init__(self) -> None:
        self.values = (
            np.load(ROOT / SOURCE_PATHS["short_feature"], mmap_mode="r", allow_pickle=False),
            np.load(ROOT / SOURCE_PATHS["long_feature"], mmap_mode="r", allow_pickle=False),
        )
        if any(value.shape != (4977, 16, 1, 768) for value in self.values):
            raise RuntimeError("Frozen DINO frame feature shapes changed")
        for value in self.values:
            for start in range(0, len(value), 256):
                if not np.isfinite(value[start : start + 256]).all():
                    raise RuntimeError("Frozen DINO frame cache contains non-finite values")

    def observations(
        self, streams: np.ndarray, rows: np.ndarray, slots: np.ndarray
    ) -> np.ndarray:
        result = np.empty((len(rows), 768), dtype=np.float32)
        for stream in (0, 1):
            mask = streams == stream
            if mask.any():
                result[mask] = self.values[stream][rows[mask], slots[mask], 0].astype(
                    np.float32
                )
        if not np.isfinite(result).all():
            raise RuntimeError("Frozen DINO frame observations are non-finite")
        return result

    def centers(self, rows: np.ndarray) -> np.ndarray:
        result = self.values[0][rows, CENTER_INDEX, 0].astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Frozen DINO center observations are non-finite")
        return result


def _candidate_observations(
    frames: dict[str, np.ndarray], train_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    allowed = np.zeros(len(frames["labels"]), dtype=bool)
    allowed[train_rows] = True
    streams, rows, slots, labels, weights = [], [], [], [], []
    for stream, name in enumerate(("short", "long")):
        mask = frames[f"{name}_valid"] & allowed[:, None]
        flat = np.flatnonzero(mask)
        row, slot = np.divmod(flat, 16)
        streams.append(np.full(len(row), stream, dtype=np.int8))
        rows.append(row.astype(np.int64))
        slots.append(slot.astype(np.int8))
        labels.append(frames[f"{name}_frame_labels"][row, slot].astype(np.int64))
        weights.append(frames["physical_weight"][stream, row, slot].astype(np.float32))
    result = tuple(np.concatenate(values) for values in (streams, rows, slots, labels, weights))
    if (
        len(result[0]) == 0
        or np.any(result[3] < 0)
        or np.any(result[4] <= 0)
        or not set(result[1].tolist()).issubset(set(train_rows.tolist()))
    ):
        raise RuntimeError("Candidate auxiliary observation population is invalid")
    return result  # type: ignore[return-value]


def _cycled_permutations(length: int, total: int, rng: np.random.Generator) -> np.ndarray:
    if length < 1 or total < 1:
        raise ValueError("Cycled permutations need positive lengths")
    values = []
    while sum(len(value) for value in values) < total:
        values.append(rng.permutation(length))
    return np.concatenate(values)[:total]


def _fixed_anchor(m4: np.ndarray, a3: np.ndarray, a3_weight: float) -> np.ndarray:
    if a3_weight != 0.10:
        raise RuntimeError("FSAR fixed-anchor weight changed")
    return geometric_residual(m4, a3, np.full(len(m4), a3_weight), epsilon=1e-12)


def _fit_paths(run: Path, arm: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    directory = run / "models" / arm / f"fold-{fold}" / f"seed-{seed}"
    return directory / "predictions.npz", directory / "checkpoint.npz", directory / "receipt.json"


def _expected_parameter_count(architecture: dict[str, Any]) -> int:
    input_dim, width, classes = (
        architecture["input_dim"],
        architecture["width"],
        architecture["classes"],
    )
    return 2 * input_dim + input_dim * width + width + width * classes + classes


def _validate_fit_receipt_identity(
    receipt: dict[str, Any],
    *,
    arm: str,
    fold: int,
    seed: int,
    lock_hash: str,
    train: np.ndarray,
    held: np.ndarray,
    data: dict[str, np.ndarray],
    expected_steps: int,
    expected_parameters: int,
) -> None:
    if (
        receipt.get("status") != "FSAR_FIT_COMPLETE_OUTER_METRICS_EMBARGOED"
        or receipt.get("execution_lock_sha256") != lock_hash
        or receipt.get("arm") != arm
        or receipt.get("fold") != fold
        or receipt.get("seed") != seed
        or receipt.get("train_rows") != len(train)
        or receipt.get("held_rows") != len(held)
        or receipt.get("outer_held_labels_read") != 0
        or receipt.get("optimizer_steps") != expected_steps
        or receipt.get("parameters") != expected_parameters
        or receipt.get("train_sample_ids_sha256")
        != canonical_hash(data["sample_ids"][train].tolist())
        or receipt.get("train_labels_sha256") != canonical_hash(data["labels"][train].tolist())
        or receipt.get("held_sample_ids_sha256")
        != canonical_hash(data["sample_ids"][held].tolist())
    ):
        raise RuntimeError("FSAR completed fit request identity changed")


def _validate_probability_array(values: np.ndarray, rows: int, name: str) -> None:
    if (
        values.shape != (rows, 3)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0)
    ):
        raise RuntimeError(f"FSAR probability artifact changed: {name}")


def _validate_checkpoint_arrays(path: Path, architecture: dict[str, Any]) -> None:
    width, input_dim, classes = (
        architecture["width"],
        architecture["input_dim"],
        architecture["classes"],
    )
    expected = {
        "encoder__0__weight": (input_dim,),
        "encoder__0__bias": (input_dim,),
        "encoder__1__weight": (width, input_dim),
        "encoder__1__bias": (width,),
        "local_classifier__weight": (classes, width),
        "local_classifier__bias": (classes,),
    }
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected) or any(
            saved[name].shape != shape or not np.isfinite(saved[name]).all()
            for name, shape in expected.items()
        ):
            raise RuntimeError("FSAR checkpoint structure or values changed")


def _checkpoint(model: FrameSupervisedAnchorResidual) -> dict[str, np.ndarray]:
    return {
        name.replace(".", "__"): value.detach().cpu().numpy()
        for name, value in model.state_dict().items()
    }


def resource_gate(protocol: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSAR requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_frame_training(20260912)
    architecture = protocol["architecture"]
    model = FrameSupervisedAnchorResidual(
        input_dim=architecture["input_dim"],
        width=architecture["width"],
        classes=architecture["classes"],
        dropout=architecture["dropout"],
        residual_scale=architecture["residual_scale"],
        parameter_limit=architecture["parameter_limit"],
    ).to(device)
    if model.trainable_parameters != _expected_parameter_count(architecture):
        raise RuntimeError("FSAR resource-gate parameter count changed")
    batch = min(protocol["training"]["batch_size"], 2048)
    center = torch.randn(batch, 768, device=device)
    auxiliary = torch.randn(batch, 768, device=device)
    anchor = torch.softmax(torch.randn(batch, 3, device=device), 1)
    labels = torch.arange(batch, device=device) % 3
    loss, _ = frame_residual_loss(
        model,
        center_features=center,
        anchor_probabilities=anchor,
        center_labels=labels,
        auxiliary_features=auxiliary,
        auxiliary_labels=labels,
        auxiliary_weights=torch.ones(batch, device=device),
        class_weight=torch.ones(3, device=device),
        center_loss_weight=protocol["training"]["center_loss_weight"],
        auxiliary_loss_weight=protocol["training"]["auxiliary_loss_weight"],
        anchor_kl_weight=protocol["training"]["anchor_kl_weight"],
    )
    loss.backward()
    if not torch.isfinite(loss) or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("FSAR resource gate found a missing or non-finite gradient")
    receipt = {
        "status": "FSAR_SYNTHETIC_RESOURCE_GATE_PASSED",
        "execution_lock_sha256": lock_hash,
        "labels_read": 0,
        "optimizer_steps": 0,
        "batch_size": batch,
        "parameters": model.trainable_parameters,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device": torch.cuda.get_device_name(device),
    }
    del model, center, auxiliary, anchor, labels, loss
    torch.cuda.empty_cache()
    return receipt


def fit_one(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
    store: FrameFeatureStore,
    lock_hash: str,
    arm: str,
    fold: int,
    seed: int,
    *,
    recover_incomplete: bool = False,
) -> dict[str, Any]:
    _validate_lock(run, data, lock_hash)
    if arm not in TRAINED_ARMS or fold not in range(5) or seed not in SEEDS:
        raise ValueError("FSAR fit request is outside the locked worklist")
    selected = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
    train, held = selected.train_rows, selected.held_rows
    auxiliary = _candidate_observations(frames, train)
    config, architecture = protocol["training"], protocol["architecture"]
    batch_size = config["batch_size"]
    steps = math.ceil(len(auxiliary[0]) / batch_size)
    expected_steps = steps * config["epochs"]
    expected_parameters = _expected_parameter_count(architecture)
    prediction_path, checkpoint_path, receipt_path = _fit_paths(run, arm, fold, seed)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        _validate_fit_receipt_identity(
            receipt,
            arm=arm,
            fold=fold,
            seed=seed,
            lock_hash=lock_hash,
            train=train,
            held=held,
            data=data,
            expected_steps=expected_steps,
            expected_parameters=expected_parameters,
        )
        if (
            file_sha256(prediction_path) != receipt["predictions_sha256"]
            or file_sha256(checkpoint_path) != receipt["checkpoint_sha256"]
        ):
            raise RuntimeError("Completed FSAR fit changed")
        output = np.load(prediction_path, allow_pickle=False)
        if (
            not np.array_equal(output["held_rows"], held)
            or not np.array_equal(output["sample_ids"], data["sample_ids"][held])
        ):
            raise RuntimeError("Completed FSAR fit row identity changed")
        for name in ("anchor_probabilities", "probabilities", "standalone_probabilities"):
            _validate_probability_array(output[name], len(held), name)
        output.close()
        _validate_checkpoint_arrays(checkpoint_path, architecture)
        return receipt
    directory = receipt_path.parent
    if directory.exists():
        raise RuntimeError("Partial final FSAR fit directory retained; use a fresh run directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + ".incomplete")
    if staging.exists():
        if not recover_incomplete:
            raise RuntimeError(
                f"Incomplete FSAR fit retained at {staging}; rerun with --recover-incomplete"
            )
        if staging.resolve().parent != directory.resolve().parent or staging.name != directory.name + ".incomplete":
            raise RuntimeError("Refusing to recover an unexpected FSAR staging path")
        shutil.rmtree(staging)
    staging.mkdir()
    stage_prediction = staging / "predictions.npz"
    stage_checkpoint = staging / "checkpoint.npz"
    stage_receipt = staging / "receipt.json"
    seed_index = SEEDS.index(seed)
    inner_anchor = _fixed_anchor(
        selected.m4_inner["probabilities"],
        selected.a3_inner["probabilities"],
        architecture["a3_anchor_weight"],
    )
    outer_anchor = _fixed_anchor(
        selected.m4_outer["probabilities"][seed_index],
        selected.a3_outer["probabilities"][seed_index],
        architecture["a3_anchor_weight"],
    )
    center_features = store.centers(train)
    held_features = store.centers(held)
    device = torch.device("cuda")
    seed_frame_training(seed)
    model = FrameSupervisedAnchorResidual(
        input_dim=architecture["input_dim"],
        width=architecture["width"],
        classes=architecture["classes"],
        dropout=architecture["dropout"],
        residual_scale=architecture["residual_scale"],
        parameter_limit=architecture["parameter_limit"],
    ).to(device)
    if model.trainable_parameters != expected_parameters:
        raise RuntimeError("FSAR fit parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    weight = torch.as_tensor(class_weights(data["labels"], train), device=device)
    history = []
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        rng = np.random.default_rng(seed * 1000 + epoch)
        total = steps * batch_size
        center_order = _cycled_permutations(len(train), total, rng)
        if arm == "f3_all_cached_frame_supervision":
            auxiliary_order = _cycled_permutations(len(auxiliary[0]), total, rng)
        else:
            auxiliary_order = _cycled_permutations(len(train), total, rng)
        sums = {"loss": 0.0, "center": 0.0, "auxiliary": 0.0, "anchor_kl": 0.0}
        model.train()
        for step in range(steps):
            left, right = step * batch_size, (step + 1) * batch_size
            center_position = center_order[left:right]
            center_x = torch.as_tensor(center_features[center_position], device=device)
            center_y = torch.as_tensor(data["labels"][train[center_position]], device=device)
            anchor = torch.as_tensor(inner_anchor[center_position], dtype=torch.float32, device=device)
            aux_position = auxiliary_order[left:right]
            if arm == "f3_all_cached_frame_supervision":
                aux_x_np = store.observations(
                    auxiliary[0][aux_position], auxiliary[1][aux_position], auxiliary[2][aux_position]
                )
                aux_y_np = auxiliary[3][aux_position]
                aux_w_np = auxiliary[4][aux_position]
            else:
                aux_x_np = center_features[aux_position]
                aux_y_np = data["labels"][train[aux_position]]
                aux_w_np = np.ones(len(aux_position), dtype=np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = frame_residual_loss(
                model,
                center_features=center_x,
                anchor_probabilities=anchor,
                center_labels=center_y,
                auxiliary_features=torch.as_tensor(aux_x_np, device=device),
                auxiliary_labels=torch.as_tensor(aux_y_np, device=device),
                auxiliary_weights=torch.as_tensor(aux_w_np, device=device),
                class_weight=weight,
                center_loss_weight=config["center_loss_weight"],
                auxiliary_loss_weight=config["auxiliary_loss_weight"],
                anchor_kl_weight=config["anchor_kl_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("FSAR training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("FSAR optimizer produced non-finite parameters")
            sums["loss"] += float(loss.detach())
            for name, value in parts.items():
                sums[name] += float(value.detach())
        history.append({"epoch": epoch + 1, **{name: value / steps for name, value in sums.items()}})
        print(
            json.dumps(
                {
                    "event": "fsar_epoch",
                    "arm": arm,
                    "fold": fold,
                    "seed": seed,
                    **history[-1],
                }
            ),
            flush=True,
        )
    model.eval()
    local_chunks = []
    with torch.inference_mode():
        for left in range(0, len(held), batch_size):
            x = torch.as_tensor(held_features[left : left + batch_size], device=device)
            local_chunks.append(model.local_logits(x).cpu().numpy())
    local_logits = np.concatenate(local_chunks).astype(np.float64)
    standalone = F.softmax(torch.as_tensor(local_logits), dim=1).numpy()
    probabilities = anchored_probabilities(
        outer_anchor, local_logits, residual_scale=architecture["residual_scale"]
    )
    _validate_probability_array(standalone, len(held), "standalone")
    _validate_probability_array(probabilities, len(held), "anchored")
    _save_or_validate_npz(stage_checkpoint, **_checkpoint(model))
    _save_or_validate_npz(
        stage_prediction,
        held_rows=held,
        sample_ids=data["sample_ids"][held],
        anchor_probabilities=outer_anchor,
        probabilities=probabilities,
        standalone_probabilities=standalone,
        local_logits=local_logits,
    )
    seconds = time.perf_counter() - started
    receipt = {
        "status": "FSAR_FIT_COMPLETE_OUTER_METRICS_EMBARGOED",
        "execution_lock_sha256": lock_hash,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "outer_held_labels_read": 0,
        "auxiliary_source": (
            "all_valid_short_and_long_cached_frames"
            if arm == "f3_all_cached_frame_supervision"
            else "short_center_frame_only_cycled"
        ),
        "candidate_auxiliary_observations_available": int(len(auxiliary[0])),
        "auxiliary_unique_observations_used": (
            int(len(auxiliary[0])) if arm == "f3_all_cached_frame_supervision" else int(len(train))
        ),
        "auxiliary_presentations_per_epoch": int(steps * batch_size),
        "auxiliary_physical_weight_sum": (
            float(auxiliary[4].sum())
            if arm == "f3_all_cached_frame_supervision"
            else float(len(train))
        ),
        "auxiliary_effective_class_physical_weight_sum": (
            float((auxiliary[4] * class_weights(data["labels"], train)[auxiliary[3]]).sum())
            if arm == "f3_all_cached_frame_supervision"
            else float(class_weights(data["labels"], train)[data["labels"][train]].sum())
        ),
        "optimizer_steps": int(steps * config["epochs"]),
        "epochs": config["epochs"],
        "seconds": seconds,
        "parameters": model.trainable_parameters,
        "history": history,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "train_labels_sha256": canonical_hash(data["labels"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "predictions_sha256": file_sha256(stage_prediction),
        "checkpoint_sha256": file_sha256(stage_checkpoint),
    }
    immutable_json(stage_receipt, receipt)
    _validate_lock(run, data, lock_hash)
    os.replace(staging, directory)
    if (
        file_sha256(prediction_path) != receipt["predictions_sha256"]
        or file_sha256(checkpoint_path) != receipt["checkpoint_sha256"]
        or read_json(receipt_path) != receipt
    ):
        raise RuntimeError("Atomic FSAR fit commit failed validation")
    _validate_checkpoint_arrays(checkpoint_path, architecture)
    print(
        json.dumps(
            {
                "event": "fsar_fit_complete",
                "arm": arm,
                "fold": fold,
                "seed": seed,
                "seconds": seconds,
            }
        ),
        flush=True,
    )
    del model, optimizer
    torch.cuda.empty_cache()
    return receipt


def _transitions(labels: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> dict[str, int]:
    before, after = anchor.argmax(1) == labels, candidate.argmax(1) == labels
    rescues, harms = int((~before & after).sum()), int((before & ~after).sum())
    return {
        "rescues": rescues,
        "harms": harms,
        "net_corrections": rescues - harms,
        "prediction_changes": int((candidate.argmax(1) != anchor.argmax(1)).sum()),
    }


def _scenario_statistics(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scenarios: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.unique(scenarios)
    group_rows = {group: np.flatnonzero(scenarios == group) for group in groups}
    observed = probability_metrics(labels, candidate)["macro_f1"] - probability_metrics(
        labels, reference
    )["macro_f1"]
    swaps = []
    for assignment in itertools.product((False, True), repeat=len(groups)):
        left, right = candidate.copy(), reference.copy()
        for keep, group in zip(assignment, groups, strict=True):
            if not keep:
                rows = group_rows[group]
                left[rows], right[rows] = reference[rows], candidate[rows]
        swaps.append(
            probability_metrics(labels, left)["macro_f1"]
            - probability_metrics(labels, right)["macro_f1"]
        )
    rng = np.random.default_rng(seed)
    deltas = np.empty((resamples, 3))
    for index in range(resamples):
        sampled = rng.choice(groups, len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        cand, base = probability_metrics(labels[rows], candidate[rows]), probability_metrics(
            labels[rows], reference[rows]
        )
        deltas[index] = (
            cand["macro_f1"] - base["macro_f1"],
            cand["nll"] - base["nll"],
            cand["brier"] - base["brier"],
        )
    return {
        "observed_macro_f1_delta": observed,
        "scenario_swap_assignments": len(swaps),
        "scenario_swap_one_sided_p": float(np.mean(np.asarray(swaps) >= observed - 1e-15)),
        "macro_f1_delta_ci95": np.quantile(deltas[:, 0], [0.025, 0.975]).tolist(),
        "nll_delta_ci95": np.quantile(deltas[:, 1], [0.025, 0.975]).tolist(),
        "nll_delta_one_sided_upper95": float(np.quantile(deltas[:, 1], 0.95)),
        "brier_delta_ci95": np.quantile(deltas[:, 2], [0.025, 0.975]).tolist(),
        "brier_delta_one_sided_upper95": float(np.quantile(deltas[:, 2], 0.95)),
    }


def summarize(
    run: Path,
    protocol: dict[str, Any],
    data: dict[str, np.ndarray],
    lock_hash: str,
) -> dict[str, Any]:
    _validate_lock(run, data, lock_hash)
    rows = len(data["labels"])
    frame_data = load_cached_frame_supervision(FRAME_DATA)
    seeds = np.full((len(ARMS), 3, rows, 3), np.nan)
    fit_hashes = {}
    with np.load(SOURCE / "results/v0001/oof_probabilities.npz", allow_pickle=False) as official:
        if not np.array_equal(official["sample_ids"], data["sample_ids"]):
            raise RuntimeError("Official M4 OOF row identity changed")
        official_m4 = official["new_source_m4_seeds"].copy()
    seeds[0] = official_m4
    for fold in range(5):
        selected = selected_fold_artifacts(ROOT, SOURCE, SEAR, data, fold, seeds=SEEDS)
        held = selected.held_rows
        for seed_index, seed in enumerate(SEEDS):
            seeds[1, seed_index, held] = _fixed_anchor(
                selected.m4_outer["probabilities"][seed_index],
                selected.a3_outer["probabilities"][seed_index],
                protocol["architecture"]["a3_anchor_weight"],
            )
            for arm_index, arm in enumerate(TRAINED_ARMS, start=2):
                prediction_path, checkpoint_path, receipt_path = _fit_paths(
                    run, arm, fold, seed
                )
                receipt = read_json(receipt_path)
                train = selected.train_rows
                auxiliary_count = sum(
                    int(valid.sum())
                    for valid in (
                        frame_data["short_valid"][train],
                        frame_data["long_valid"][train],
                    )
                )
                expected_steps = (
                    math.ceil(auxiliary_count / protocol["training"]["batch_size"])
                    * protocol["training"]["epochs"]
                )
                _validate_fit_receipt_identity(
                    receipt,
                    arm=arm,
                    fold=fold,
                    seed=seed,
                    lock_hash=lock_hash,
                    train=train,
                    held=held,
                    data=data,
                    expected_steps=expected_steps,
                    expected_parameters=_expected_parameter_count(protocol["architecture"]),
                )
                if (
                    file_sha256(prediction_path) != receipt["predictions_sha256"]
                    or file_sha256(checkpoint_path) != receipt["checkpoint_sha256"]
                ):
                    raise RuntimeError("FSAR fit inventory failed metric embargo")
                _validate_checkpoint_arrays(checkpoint_path, protocol["architecture"])
                with np.load(prediction_path, allow_pickle=False) as output:
                    if not np.array_equal(output["held_rows"], held) or not np.array_equal(
                        output["sample_ids"], data["sample_ids"][held]
                    ):
                        raise RuntimeError("FSAR output row identity changed")
                    expected_anchor = seeds[1, seed_index, held]
                    local_logits = output["local_logits"].astype(np.float64)
                    expected_standalone = F.softmax(torch.as_tensor(local_logits), dim=1).numpy()
                    expected_probability = anchored_probabilities(
                        expected_anchor,
                        local_logits,
                        residual_scale=protocol["architecture"]["residual_scale"],
                    )
                    if (
                        not np.array_equal(output["anchor_probabilities"], expected_anchor)
                        or not np.allclose(
                            output["standalone_probabilities"], expected_standalone, atol=1e-14, rtol=0
                        )
                        or not np.allclose(
                            output["probabilities"], expected_probability, atol=1e-14, rtol=0
                        )
                    ):
                        raise RuntimeError("FSAR stored forward outputs do not replay")
                    seeds[arm_index, seed_index, held] = output["probabilities"]
                    seeds[arm_index + 2, seed_index, held] = output[
                        "standalone_probabilities"
                    ]
                fit_hashes[_relative(receipt_path)] = file_sha256(receipt_path)
    if len(fit_hashes) != 30 or not np.isfinite(seeds).all():
        raise RuntimeError("FSAR needs exactly30 complete fits and all outer outputs")
    averaged = seeds.mean(1)
    labels = data["labels"]
    metrics = {arm: probability_metrics(labels, averaged[index]) for index, arm in enumerate(ARMS)}
    seed_metrics = {
        arm: [probability_metrics(labels, seeds[index, seed]) for seed in range(3)]
        for index, arm in enumerate(ARMS)
    }
    transitions = {
        arm: _transitions(labels, averaged[index], averaged[0])
        for index, arm in enumerate(ARMS)
        if index
    }
    primary = protocol["primary_arm"]
    primary_index = ARMS.index(primary)
    statistics = {
        reference: _scenario_statistics(
            labels,
            averaged[primary_index],
            averaged[ARMS.index(reference)],
            data["scenarios"],
            resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
            seed=protocol["statistics"]["seed"] + index,
        )
        for index, reference in enumerate(("f0_exact_m4", "f2_center_only_supervision"))
    }
    if any(
        value["scenario_swap_assignments"] != protocol["statistics"]["exact_scenario_swaps"]
        for value in statistics.values()
    ):
        raise RuntimeError("FSAR exact scenario-swap inventory changed")
    scenario_metrics = {}
    improved = 0
    worst_accuracy_delta = 0.0
    for scenario in np.unique(data["scenarios"]):
        mask = data["scenarios"] == scenario
        scenario_metrics[str(scenario)] = {
            arm: probability_metrics(labels[mask], averaged[index, mask])
            for index, arm in enumerate(ARMS)
        }
        delta = (
            scenario_metrics[str(scenario)][primary]["accuracy"]
            - scenario_metrics[str(scenario)]["f0_exact_m4"]["accuracy"]
        )
        improved += int(delta > 0)
        worst_accuracy_delta = min(worst_accuracy_delta, delta)
    gates = protocol["screen_gates"]
    primary_metric, m4_metric = metrics[primary], metrics["f0_exact_m4"]
    control_metric = metrics["f2_center_only_supervision"]
    class_delta = np.asarray(primary_metric["per_class_f1"]) - np.asarray(
        m4_metric["per_class_f1"]
    )
    seed_sd = float(np.std([value["macro_f1"] for value in seed_metrics[primary]]))
    checks = {
        "minimum_primary_macro_f1": primary_metric["macro_f1"]
        >= gates["minimum_primary_macro_f1"],
        "minimum_gain_over_m4_points": 100
        * (primary_metric["macro_f1"] - m4_metric["macro_f1"])
        >= gates["minimum_gain_over_m4_points"],
        "minimum_gain_over_center_control_points": 100
        * (primary_metric["macro_f1"] - control_metric["macro_f1"])
        >= gates["minimum_gain_over_center_control_points"],
        "minimum_net_corrections_over_m4": transitions[primary]["net_corrections"]
        >= gates["minimum_net_corrections_over_m4"],
        "scenario_swap_vs_m4_one_sided_p_maximum": statistics["f0_exact_m4"][
            "scenario_swap_one_sided_p"
        ]
        <= gates["scenario_swap_vs_m4_one_sided_p_maximum"],
        "scenario_swap_vs_center_control_one_sided_p_maximum": statistics[
            "f2_center_only_supervision"
        ]["scenario_swap_one_sided_p"]
        <= gates["scenario_swap_vs_center_control_one_sided_p_maximum"],
        "maximum_nll_delta_over_m4_one_sided_upper95": statistics["f0_exact_m4"][
            "nll_delta_one_sided_upper95"
        ]
        <= gates["maximum_nll_delta_over_m4_one_sided_upper95"],
        "maximum_brier_delta_over_m4_one_sided_upper95": statistics["f0_exact_m4"][
            "brier_delta_one_sided_upper95"
        ]
        <= gates["maximum_brier_delta_over_m4_one_sided_upper95"],
        "maximum_worst_class_f1_loss_points": 100 * class_delta.min()
        >= -gates["maximum_worst_class_f1_loss_points"],
        "minimum_improved_scenarios": improved >= gates["minimum_improved_scenarios"],
        "maximum_scenario_accuracy_loss_points": 100 * worst_accuracy_delta
        >= -gates["maximum_scenario_accuracy_loss_points"],
        "maximum_seed_macro_f1_sd_points": 100 * seed_sd
        <= gates["maximum_seed_macro_f1_sd_points"],
        "exact_m4_replay": bool(
            np.array_equal(seeds[0], official_m4)
            and np.array_equal(averaged[0], official_m4.mean(0))
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result_dir = run / "results/v0001"
    oof_path = result_dir / "oof_predictions.npz"
    _save_or_validate_npz(
        oof_path,
        sample_ids=data["sample_ids"],
        labels=labels,
        scenarios=data["scenarios"],
        folds=data["folds"],
        arms=np.asarray(ARMS),
        seeds=np.asarray(SEEDS),
        seed_probabilities=seeds,
        mean_probabilities=averaged,
    )
    receipt = {
        "status": "FSAR_30_FIT_ADAPTIVE_SCREEN_COMPLETE",
        "complete": True,
        "study_id": protocol["study_id"],
        "scope": protocol["scope"],
        "rows": rows,
        "model_fits": 30,
        "primary_arm": primary,
        "run_directory": str(run.resolve()),
        "execution_lock_sha256": lock_hash,
        "fit_receipt_sha256": fit_hashes,
        "oof_predictions_sha256": file_sha256(oof_path),
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "transitions_vs_m4": transitions,
        "scenario_metrics": scenario_metrics,
        "primary_statistics": statistics,
        "primary_effects": {
            "macro_f1_gain_over_m4_points": 100
            * (primary_metric["macro_f1"] - m4_metric["macro_f1"]),
            "macro_f1_gain_over_center_control_points": 100
            * (primary_metric["macro_f1"] - control_metric["macro_f1"]),
            "per_class_f1_delta_over_m4_points": (100 * class_delta).tolist(),
            "improved_scenarios": improved,
            "worst_scenario_accuracy_delta_points": 100 * worst_accuracy_delta,
            "seed_macro_f1_sd_points": 100 * seed_sd,
        },
        "screen_checks": checks,
        "screen_gate_passed": bool(all(checks.values())),
        "confirmation_status": "REQUIRED_EXTERNAL_SCENARIOS_NOT_EVALUATED",
        "interpretation": "Adaptive screen only. The f3-f2 contrast isolates additional cached-frame supervision under an identical architecture and optimizer-step budget.",
    }
    _validate_lock(run, data, lock_hash)
    immutable_json(result_dir / "summary.json", receipt)
    _validate_lock(run, data, lock_hash)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(result_dir / "summary.json"),
                "primary_macro_f1": primary_metric["macro_f1"],
                "m4_macro_f1": m4_metric["macro_f1"],
                "center_control_macro_f1": control_metric["macro_f1"],
                "screen_gate_passed": receipt["screen_gate_passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resource-gate-only", action="store_true")
    parser.add_argument(
        "--recover-incomplete",
        action="store_true",
        help="discard only a precisely named incomplete fit staging directory and retry that fit",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    protocol, data, frames, lock_hash = prepare(run)
    _validate_lock(run, data, lock_hash)
    gate_path = run / "resource_gate.json"
    if gate_path.exists():
        gate = read_json(gate_path)
        if (
            gate.get("status") != "FSAR_SYNTHETIC_RESOURCE_GATE_PASSED"
            or gate.get("execution_lock_sha256") != lock_hash
            or gate.get("parameters") != _expected_parameter_count(protocol["architecture"])
            or gate.get("batch_size") != protocol["training"]["batch_size"]
        ):
            raise RuntimeError("Completed FSAR resource gate changed")
    else:
        gate = resource_gate(protocol, lock_hash)
        immutable_json(gate_path, gate)
    print(json.dumps(gate, indent=2), flush=True)
    if args.resource_gate_only:
        return
    store = FrameFeatureStore()
    for arm in TRAINED_ARMS:
        for fold in protocol["training"]["outer_folds"]:
            for seed in protocol["training"]["outer_seeds"]:
                fit_one(
                    run,
                    protocol,
                    data,
                    frames,
                    store,
                    lock_hash,
                    arm,
                    fold,
                    seed,
                    recover_incomplete=args.recover_incomplete,
                )
    summarize(run, protocol, data, lock_hash)


if __name__ == "__main__":
    main()
