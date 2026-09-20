"""Prepare, cross-fit, and replay the declared ordered local-motion experiment.

The default stage only materializes existing permitted token caches on CPU. Real
training requires ``--stage train`` and an explicit device. Historical P6 OOF
probabilities are not opened until all twenty declared workloads are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_okutama_video_probe import holm_adjust, metrics, paired_statistics
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import functional as F

from hac.ordered_local_motion import ARMS, OrderedLocalMotion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".runs/research_20260908/ordered_motion_v2"
DEFAULT_MEMORY = ROOT / ".runs/research_20260908/evidence_memory/data"
PROTOCOL = ROOT / "experiments/okutama_ordered_motion_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def checked(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    if (
        not path.is_file()
        or path.stat().st_size != entry["size_bytes"]
        or sha256(path) != entry["sha256"]
    ):
        raise RuntimeError(f"Artifact changed: {path}")
    return path


def select_clip_block(short: np.ndarray, long: np.ndarray, long_valid: np.ndarray) -> np.ndarray:
    """Use the entire exact short tensor for invalid long rows, never interpolation."""
    if short.shape != long.shape or long_valid.shape != (len(short),):
        raise ValueError("Short/long tokens and row validity are not aligned")
    if short.dtype != np.float16 or long.dtype != np.float16 or long_valid.dtype != np.bool_:
        raise ValueError("Expected float16 tokens and boolean long validity")
    if not np.isfinite(short).all() or not np.isfinite(long[long_valid]).all():
        raise ValueError("Valid source token is nonfinite")
    if np.any(long[~long_valid] != 0):
        raise ValueError("Invalid long rows must retain their original zero sentinel")
    result = long.copy()
    result[~long_valid] = short[~long_valid]
    return result


def prepare(output: Path, memory_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if (output / "data_summary.json").exists():
        _, _, _, summary = load_prepared(output)
        return summary
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    memory_summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    for name in ("data_lock.json", "memory_data.npz"):
        checked({"path": str(memory_dir / name), **memory_summary["artifacts"][name]})
    upstream = json.loads((memory_dir / "data_lock.json").read_text(encoding="utf-8"))
    source = upstream["source_receipts"]
    selected_names = (
        "vjepa21_real_clip",
        "vjepa21_real_clip_summary",
        "vjepa21_long16_real_clip_vjepa21_long16_real_clip.npy",
        "vjepa21_long16_real_clip_summary",
        "vjepa21_long16_real_clip_validity.npy",
        "short_frame_manifest",
        "long_frame_manifest",
    )
    selected = {name: receipt(checked(source[name])) for name in selected_names}
    with np.load(memory_dir / "memory_data.npz", allow_pickle=False) as archive:
        population = {
            key: archive[key].copy()
            for key in (
                "sample_ids",
                "labels",
                "scenarios",
                "folds",
                "frames",
                "recordings",
                "tracks",
                "long_valid",
                "quality",
            )
        }
    ids = population["sample_ids"]
    if len(ids) != protocol["primary_rows"] or ids.tolist() != upstream["center_sample_ids"]:
        raise RuntimeError("Prepared memory population is not the fixed permitted center cohort")
    if sorted(np.unique(population["folds"]).tolist()) != protocol["outer_folds"]:
        raise RuntimeError("Original outer folds changed")
    for name in ("vjepa21_real_clip_summary", "vjepa21_long16_real_clip_summary"):
        summary = json.loads(Path(selected[name]["path"]).read_text(encoding="utf-8"))
        if summary["sample_ids"] != ids.tolist():
            raise RuntimeError("Token cache sample identities changed")
    long_valid = population["long_valid"]
    validity = np.load(
        selected["vjepa21_long16_real_clip_validity.npy"]["path"], allow_pickle=False
    )
    if not np.array_equal(long_valid, validity):
        raise RuntimeError("Long token and memory fallback masks disagree")
    frame_arrays = {}
    for name in ("short", "long"):
        frame = pd.read_csv(
            selected[f"{name}_frame_manifest"]["path"],
            usecols=["sample_id", "time_index", "source_frame"],
        )
        if len(frame) != len(ids) * 16 or not np.array_equal(
            frame.sample_id.to_numpy().reshape(-1, 16),
            np.broadcast_to(ids[:, None], (len(ids), 16)),
        ):
            raise RuntimeError("Frame manifest center order changed")
        if not np.array_equal(
            frame.time_index.to_numpy().reshape(-1, 16),
            np.broadcast_to(np.arange(16), (len(ids), 16)),
        ):
            raise RuntimeError("Frame manifest slot order changed")
        frame_arrays[name] = frame.source_frame.to_numpy(np.int32).reshape(-1, 16)
        if not np.array_equal(frame_arrays[name][:, 8], population["frames"]):
            raise RuntimeError("Center timestamp mismatch")
    source_frames = np.where(long_valid[:, None], frame_arrays["long"], frame_arrays["short"])
    relative = (source_frames - population["frames"][:, None]) / 30.0
    population["times"] = relative.reshape(-1, 8, 2).mean(-1).astype(np.float32)
    population["source_frames"] = source_frames
    if np.any(source_frames < 0) or not np.all(np.diff(population["times"], axis=1) > 0):
        raise RuntimeError("Selected fallback produced unavailable or unordered frames")
    lock = {
        "status": "ORDERED_MOTION_LOCKED_BEFORE_MATERIALIZATION_OR_FITS",
        "protocol": protocol,
        "sources": {
            **selected,
            "memory_data": receipt(memory_dir / "memory_data.npz"),
            "memory_lock": receipt(memory_dir / "data_lock.json"),
            "protocol": receipt(PROTOCOL),
            "module": receipt(ROOT / "src/hac/ordered_local_motion.py"),
            "runner": receipt(Path(__file__)),
            "statistics": receipt(ROOT / "experiments/run_okutama_video_probe.py"),
        },
        "sample_ids_sha256": canonical_hash(ids.tolist()),
        "historical_oof_read": False,
        "protected_rows_read": 0,
    }
    lock_path = output / "execution_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError(
            "Existing ordered-motion lock differs; preserve it and use a fresh output directory"
        )
    write_json(lock_path, lock)
    short = np.load(selected["vjepa21_real_clip"]["path"], allow_pickle=False, mmap_mode="r")
    long = np.load(
        selected["vjepa21_long16_real_clip_vjepa21_long16_real_clip.npy"]["path"],
        allow_pickle=False,
        mmap_mode="r",
    )
    expected = tuple(protocol["input_contract"]["shape"])
    if short.shape != expected or long.shape != expected:
        raise RuntimeError("Frozen token shape changed")
    destination = np.lib.format.open_memmap(
        output / "tokens.npy", mode="w+", dtype=np.float16, shape=expected
    )
    for start in range(0, len(ids), 64):
        end = min(start + 64, len(ids))
        destination[start:end] = select_clip_block(
            short[start:end], long[start:end], long_valid[start:end]
        )
    destination.flush()
    del destination
    np.savez(output / "population.npz", **population)
    summary = {
        "status": "ORDERED_MOTION_DATA_READY_NO_FITS",
        "rows": len(ids),
        "scenarios": len(np.unique(population["scenarios"])),
        "long_valid_rows": int(long_valid.sum()),
        "exact_short_fallback_rows": int((~long_valid).sum()),
        "token_shape": expected,
        "token_dtype": "float16",
        "model_fits": 0,
        "historical_oof_read": False,
        "protected_rows_read": 0,
        "materialization_seconds": time.perf_counter() - started,
        "input_bytes_per_float32_batch": protocol["batch_size"] * int(np.prod(expected[1:])) * 4,
        "parameter_counts": {arm: make_model(protocol, arm).trainable_parameters for arm in ARMS},
        "artifacts": {
            name: receipt(output / name)
            for name in ("execution_lock.json", "tokens.npy", "population.npz")
        },
    }
    write_json(output / "data_summary.json", summary)
    return summary


def load_prepared(
    output: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    summary = json.loads((output / "data_summary.json").read_text(encoding="utf-8"))
    for entry in summary["artifacts"].values():
        checked(entry)
    lock = json.loads((output / "execution_lock.json").read_text(encoding="utf-8"))
    for entry in lock["sources"].values():
        checked(entry)
    tokens = np.load(output / "tokens.npy", allow_pickle=False, mmap_mode="r")
    with np.load(output / "population.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if canonical_hash(data["sample_ids"].tolist()) != lock["sample_ids_sha256"]:
        raise RuntimeError("Prepared population identity changed")
    return lock, tokens, data, summary


def make_model(protocol: dict[str, Any], arm: str) -> OrderedLocalMotion:
    return OrderedLocalMotion(
        768,
        arm,
        width=protocol["width"],
        rank=protocol["rank"],
        quality_dim=6,
        layers=protocol["layers"],
        heads=protocol["heads"],
        dropout=protocol["dropout"],
        lags=tuple(protocol["input_contract"]["lags"]),
        spatial_penalty=protocol["spatial_penalty"],
        parameter_limit=protocol["max_parameters"],
    )


def scaling(tokens: np.ndarray, quality: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    if len(train) < 1 or not np.array_equal(train, np.unique(train)):
        raise ValueError("Training rows must be nonempty, sorted and unique")
    total, squares, count = np.zeros(tokens.shape[-1]), np.zeros(tokens.shape[-1]), 0
    for start in range(0, len(train), 32):
        block = np.asarray(tokens[train[start : start + 32]], dtype=np.float64)
        total += block.sum((0, 1, 2))
        squares += np.square(block).sum((0, 1, 2))
        count += int(np.prod(block.shape[:-1]))
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    std = np.where(std < 1e-6, 1, std)
    q_mean = quality[train].mean(0, dtype=np.float64)
    q_std = quality[train].std(0, dtype=np.float64)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "quality_mean": q_mean.astype(np.float32),
        "quality_std": np.where(q_std < 1e-6, 1, q_std).astype(np.float32),
    }


def batch_inputs(
    tokens, data, rows, scaler, device, *, permutation: str | None = None, protocol=None
):
    values = (np.asarray(tokens[rows], dtype=np.float32) - scaler["mean"]) / scaler["std"]
    if permutation == "time":
        values = values[:, protocol["inference_diagnostics"]["temporal_permutation"]]
    elif permutation == "region":
        values = values[:, :, protocol["inference_diagnostics"]["region_permutation"]]
    q = (data["quality"][rows] - scaler["quality_mean"]) / scaler["quality_std"]
    return (
        torch.from_numpy(np.ascontiguousarray(values)).to(device),
        torch.from_numpy(data["times"][rows]).to(device),
        torch.from_numpy(np.ascontiguousarray(q)).to(device),
    )


def predict(model, tokens, data, rows, scaler, device, protocol, *, permutation=None):
    model.eval()
    output = np.empty((len(rows), 3), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), protocol["batch_size"]):
            selected = rows[start : start + protocol["batch_size"]]
            inputs = batch_inputs(
                tokens, data, selected, scaler, device, permutation=permutation, protocol=protocol
            )
            values = model(*inputs)["probabilities"].cpu().numpy().astype(np.float64)
            output[start : start + len(selected)] = values / values.sum(-1, keepdims=True)
    return output


def train_fit(output, tokens, data, protocol, request, train, held, device, *, epochs=None):
    """One isolated inner fit or fixed-epoch outer refit, with complete receipts."""
    request = {
        **request,
        "train_sample_ids_sha256": canonical_hash(data["sample_ids"][train].tolist()),
        "held_sample_ids_sha256": canonical_hash(data["sample_ids"][held].tolist()),
        "refit_epochs": epochs,
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    request_hash = canonical_hash(request)
    directory = output / "fits" / request_hash
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        if saved["request"] != request or saved["status"] != "ORDERED_FIT_COMPLETE":
            raise RuntimeError("Retained fit identity changed")
        for entry in saved["artifacts"].values():
            checked(entry)
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_ids"], data["sample_ids"][held]):
                raise RuntimeError("Retained fit prediction identity changed")
            return archive["probabilities"].copy(), saved
    if set(data["scenarios"][train]) & set(data["scenarios"][held]):
        raise RuntimeError("A training and prediction scenario overlaps")
    write_json(directory / "request.json", request)
    started = time.perf_counter()
    seed = int(request["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    scaler = scaling(tokens, data["quality"], train)
    model = make_model(protocol, request["arm"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=request["learning_rate"], weight_decay=request["weight_decay"]
    )
    counts = np.bincount(data["labels"][train], minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("A training fold lacks a target class")
    class_weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32, device=device)
    best_key, best_epoch, best_state, best_predictions = None, 0, None, None
    history = []
    for epoch in range(1, (epochs or protocol["max_epochs"]) + 1):
        model.train()
        shuffled = generator.permutation(train)
        total_loss, total_rows = 0.0, 0
        for start in range(0, len(shuffled), protocol["batch_size"]):
            selected = shuffled[start : start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            inputs = batch_inputs(tokens, data, selected, scaler, device)
            logits = model(*inputs)["logits"]
            labels = torch.from_numpy(data["labels"][selected]).to(device)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ordered-motion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
            total_rows += len(selected)
        history_row = {"epoch": epoch, "training_loss": total_loss / total_rows}
        print(
            json.dumps(
                {
                    "event": "ordered_training_epoch",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "arm": request["arm"],
                    "outer_fold": request.get("outer_fold"),
                    "stage": request.get("stage", "synthetic_test"),
                    "epoch": epoch,
                    "training_loss": history_row["training_loss"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if epochs is None:
            candidate = predict(model, tokens, data, held, scaler, device, protocol)
            score = metrics(data["labels"][held], candidate)
            key = (-score["macro_f1"], score["nll"])
            history_row["inner_validation"] = score
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                best_predictions = candidate
            history.append(history_row)
            if epoch - best_epoch >= protocol["early_stopping_patience"]:
                break
        else:
            history.append(history_row)
    if epochs is not None:
        best_epoch = epochs
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        best_predictions = predict(model, tokens, data, held, scaler, device, protocol)
    if best_state is None or best_predictions is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint = {
        "state_dict": best_state,
        "scaler": scaler,
        "request": request,
        "protocol": protocol,
    }
    torch.save(checkpoint, directory / "checkpoint.pt")
    arrays = {"sample_ids": data["sample_ids"][held], "probabilities": best_predictions}
    if epochs is not None:
        arrays["time_permuted_probabilities"] = predict(
            model, tokens, data, held, scaler, device, protocol, permutation="time"
        )
        arrays["region_permuted_probabilities"] = predict(
            model, tokens, data, held, scaler, device, protocol, permutation="region"
        )
    np.savez(directory / "predictions.npz", **arrays)
    saved = {
        "status": "ORDERED_FIT_COMPLETE",
        "request": request,
        "best_epoch": best_epoch,
        "train_rows": len(train),
        "held_rows": len(held),
        "history": history,
        "trainable_parameters": model.trainable_parameters,
        "seconds": time.perf_counter() - started,
        "outer_labels_used_for_training_or_selection": False,
        "historical_oof_read": False,
        "artifacts": {
            name: receipt(directory / name)
            for name in ("request.json", "checkpoint.pt", "predictions.npz")
        },
    }
    write_json(receipt_path, saved)
    print(
        json.dumps(
            {
                "event": "ordered_fit_complete",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "arm": request["arm"],
                "outer_fold": request.get("outer_fold"),
                "stage": request.get("stage", "synthetic_test"),
                "best_epoch": best_epoch,
                "seconds": saved["seconds"],
                "receipt": str(receipt_path),
            }
        ),
        flush=True,
    )
    del model, optimizer
    return best_predictions, saved


def workload(output, lock, tokens, data, arm, fold, device):
    protocol = lock["protocol"]
    train, held = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
    source_hash = sha256(output / "execution_lock.json")
    request = {"execution_lock_sha256": source_hash, "arm": arm, "outer_fold": int(fold)}
    directory = output / "workloads" / arm / f"fold-{fold}"
    path = directory / "receipt.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["request"] != request or saved["status"] != "ORDERED_WORKLOAD_COMPLETE":
            raise RuntimeError("Workload request changed")
        for entry in saved["artifacts"].values():
            checked(entry)
        return saved
    splitter = StratifiedGroupKFold(
        n_splits=protocol["inner_folds"], shuffle=True, random_state=protocol["split_seed"]
    )
    splits = [
        (train[a], train[b])
        for a, b in splitter.split(
            np.zeros(len(train)), data["labels"][train], data["scenarios"][train]
        )
    ]
    candidates = []
    fit_receipts = []
    for lr, wd in itertools.product(protocol["learning_rates"], protocol["weight_decays"]):
        inner_oof = np.full((len(data["labels"]), 3), np.nan)
        best_epochs = []
        for inner, (fit_rows, validation_rows) in enumerate(splits):
            fit_request = {
                **request,
                "stage": "inner",
                "inner_fold": inner,
                "learning_rate": lr,
                "weight_decay": wd,
                "seed": protocol["inner_seed"],
            }
            values, saved = train_fit(
                output, tokens, data, protocol, fit_request, fit_rows, validation_rows, device
            )
            inner_oof[validation_rows] = values
            best_epochs.append(saved["best_epoch"])
            fit_receipts.append(saved["artifacts"]["predictions.npz"])
        score = metrics(data["labels"][train], inner_oof[train])
        candidates.append(
            {
                "learning_rate": lr,
                "weight_decay": wd,
                "inner_metrics": score,
                "inner_best_epochs": best_epochs,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            -item["inner_metrics"]["macro_f1"],
            item["inner_metrics"]["nll"],
            item["learning_rate"],
            item["weight_decay"],
        ),
    )
    epochs = max(1, int(np.rint(np.median(selected["inner_best_epochs"]))))
    seed_predictions = []
    diagnostic_predictions = {
        "time_permuted_probabilities": [],
        "region_permuted_probabilities": [],
    }
    for seed in protocol["outer_seeds"]:
        fit_request = {
            **request,
            "stage": "outer_refit",
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "seed": seed,
        }
        values, saved = train_fit(
            output, tokens, data, protocol, fit_request, train, held, device, epochs=epochs
        )
        seed_predictions.append(values)
        fit_receipts.append(saved["artifacts"]["predictions.npz"])
        with np.load(checked(saved["artifacts"]["predictions.npz"]), allow_pickle=False) as archive:
            for name in diagnostic_predictions:
                diagnostic_predictions[name].append(archive[name].copy())
    directory.mkdir(parents=True, exist_ok=True)
    seed_predictions = np.stack(seed_predictions)
    np.savez(
        directory / "predictions.npz",
        sample_ids=data["sample_ids"][held],
        probabilities=seed_predictions.mean(0),
        seed_probabilities=seed_predictions,
        **{name: np.stack(values).mean(0) for name, values in diagnostic_predictions.items()},
    )
    saved = {
        "status": "ORDERED_WORKLOAD_COMPLETE",
        "request": request,
        "selected": selected,
        "outer_refit_epochs": epochs,
        "candidates": candidates,
        "unique_fits": len(fit_receipts),
        "fit_prediction_receipts": fit_receipts,
        "historical_oof_read": False,
        "artifacts": {"predictions.npz": receipt(directory / "predictions.npz")},
    }
    write_json(path, saved)
    print(json.dumps({"completed": f"{arm}/fold-{fold}", "selected_epochs": epochs}), flush=True)
    return saved


def aggregate(output, lock, data):
    protocol = lock["protocol"]
    predictions, seeds, perturbations, workload_receipts = {}, {}, {}, []
    expected_fits = 0
    for arm in protocol["arms"]:
        values = np.full((len(data["labels"]), 3), np.nan)
        seed_values = np.full((len(protocol["outer_seeds"]), len(data["labels"]), 3), np.nan)
        perturbed = {
            key: np.full_like(values, np.nan)
            for key in ("time_permuted_probabilities", "region_permuted_probabilities")
        }
        for fold in protocol["outer_folds"]:
            receipt_path = output / "workloads" / arm / f"fold-{fold}" / "receipt.json"
            if not receipt_path.exists():
                raise RuntimeError(
                    "All twenty workloads must finish before reading P6 or aggregating"
                )
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected_request = {
                "execution_lock_sha256": sha256(output / "execution_lock.json"),
                "arm": arm,
                "outer_fold": int(fold),
            }
            if (
                saved["status"] != "ORDERED_WORKLOAD_COMPLETE"
                or saved["request"] != expected_request
            ):
                raise RuntimeError("Incomplete or mismatched workload")
            for entry in saved["fit_prediction_receipts"]:
                checked(entry)
            with np.load(
                checked(saved["artifacts"]["predictions.npz"]), allow_pickle=False
            ) as archive:
                held = np.flatnonzero(data["folds"] == fold)
                if not np.array_equal(archive["sample_ids"], data["sample_ids"][held]):
                    raise RuntimeError("Workload holdout identities changed")
                values[held], seed_values[:, held] = (
                    archive["probabilities"],
                    archive["seed_probabilities"],
                )
                for key in perturbed:
                    perturbed[key][held] = archive[key]
            expected_fits += saved["unique_fits"]
            workload_receipts.append(receipt(receipt_path))
        predictions[arm], seeds[arm], perturbations[arm] = values, seed_values, perturbed
    if expected_fits != protocol["maximum_candidate_fits"]:
        raise RuntimeError("Declared 300-fit workload budget was not completed exactly")
    reference_spec = protocol["post_fit_system"]
    reference_path = ROOT / reference_spec["p6_path"]
    if sha256(reference_path) != reference_spec["p6_sha256"]:
        raise RuntimeError("Retained P6 reference changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        for key, target in (
            ("sample_ids", "sample_ids"),
            ("labels", "labels"),
            ("recording_ids", "scenarios"),
            ("folds", "folds"),
            ("long_valid", "long_valid"),
        ):
            if not np.array_equal(archive[key], data[target]):
                raise RuntimeError(f"P6 identity differs: {key}")
        p6 = archive[reference_spec["p6_key"]].copy()
    component_path = ROOT / reference_spec["diagnostic_components_path"]
    if sha256(component_path) != reference_spec["diagnostic_components_sha256"]:
        raise RuntimeError("Post-fit diagnostic component archive changed")
    with np.load(component_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"], data["sample_ids"]) or not np.array_equal(
            archive["p6_reference"], p6
        ):
            raise RuntimeError("Post-fit shared-error component identities changed")
        component_predictions = [
            archive[key].argmax(-1) for key in reference_spec["diagnostic_component_keys"]
        ]
        shared_failure = np.logical_and.reduce(
            [predicted != data["labels"] for predicted in component_predictions]
        )
    if int(shared_failure.sum()) != 508:
        raise RuntimeError("Retained P6 shared-failure partition changed")
    systems = {
        "p6": p6,
        **predictions,
        **{f"{arm}_blend": 0.5 * values + 0.5 * p6 for arm, values in predictions.items()},
    }
    models = {}
    for name, values in systems.items():
        correct, reference_correct = (
            values.argmax(1) == data["labels"],
            p6.argmax(1) == data["labels"],
        )
        models[name] = {
            "metrics": metrics(data["labels"], values),
            "rescues": int((correct & ~reference_correct).sum()),
            "harms": int((~correct & reference_correct).sum()),
            "shared_failure_repairs": int((correct & shared_failure).sum()),
            "per_scenario": {
                str(group): metrics(
                    data["labels"][data["scenarios"] == group], values[data["scenarios"] == group]
                )
                for group in np.unique(data["scenarios"])
            },
            "long_valid": metrics(data["labels"][data["long_valid"]], values[data["long_valid"]]),
            "short_fallback": metrics(
                data["labels"][~data["long_valid"]], values[~data["long_valid"]]
            ),
        }
    stats = {}
    for candidate, reference in protocol["statistics"]["comparisons"]:
        stats[f"{candidate}_minus_{reference}"] = paired_statistics(
            data["labels"],
            systems[candidate],
            systems[reference],
            data["scenarios"],
            bootstrap_resamples=protocol["statistics"]["scenario_bootstrap_resamples"],
            bootstrap_seed=protocol["statistics"]["seed"],
        )
    adjusted = holm_adjust(
        {name: result["one_sided_exact_swap_pvalue"] for name, result in stats.items()}
    )
    for name, value in adjusted.items():
        stats[name]["holm_adjusted_pvalue"] = value
    results = output / "results"
    results.mkdir(exist_ok=True)
    np.savez(
        results / "oof_probabilities.npz",
        sample_ids=data["sample_ids"],
        labels=data["labels"],
        scenarios=data["scenarios"],
        folds=data["folds"],
        **systems,
        **{f"{arm}_seeds": value for arm, value in seeds.items()},
    )
    summary = {
        "status": "ORDERED_MOTION_ADAPTIVE_CROSSFIT_COMPLETE",
        "models": models,
        "unique_fits": expected_fits,
        "workloads": workload_receipts,
        "seed_metrics": {
            arm: [metrics(data["labels"], value) for value in values]
            for arm, values in seeds.items()
        },
        "fixed_stress_diagnostics": {
            arm: {name: metrics(data["labels"], value) for name, value in variants.items()}
            for arm, variants in perturbations.items()
        },
        "statistics": stats,
        "error_correlations": {
            left: {
                right: float(
                    np.corrcoef(
                        systems[left].argmax(-1) != data["labels"],
                        systems[right].argmax(-1) != data["labels"],
                    )[0, 1]
                )
                for right in systems
            }
            for left in systems
        },
        "historical_reference_loaded_only_after_all_fits": True,
        "protected_rows_read": 0,
        "p6_blend_weights_selected_from_outer_results": False,
        "artifacts": {"oof_probabilities.npz": receipt(results / "oof_probabilities.npz")},
    }
    write_json(results / "summary.json", summary)
    return summary


def resource_pilot(output, lock, tokens, data, device):
    """Timing of untrained forward/backward only; no labels or optimizer steps."""
    protocol = lock["protocol"]
    selected = np.argsort(
        [hashlib.sha256(str(value).encode()).hexdigest() for value in data["sample_ids"]]
    )[:128]
    selected = np.sort(selected)
    scaler = {
        "mean": np.zeros(768, np.float32),
        "std": np.ones(768, np.float32),
        "quality_mean": np.zeros(6, np.float32),
        "quality_std": np.ones(6, np.float32),
    }
    results = {}
    for arm in ARMS:
        net = make_model(protocol, arm).to(device)
        durations = []
        for iteration in range(6):
            rows = selected[: protocol["batch_size"]]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            net.zero_grad(set_to_none=True)
            value = net(*batch_inputs(tokens, data, rows, scaler, device))["logits"]
            value.square().mean().backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if iteration >= 2:
                durations.append(time.perf_counter() - start)
        results[arm] = {
            "parameters": net.trainable_parameters,
            "batch_size": protocol["batch_size"],
            "median_forward_backward_seconds": float(np.median(durations)),
            "optimizer_steps": 0,
        }
        del net
    result = {
        "status": "ORDERED_RESOURCE_PILOT_ONLY",
        "device": device,
        "labels_read_by_pilot": False,
        "sample_ids_sha256": canonical_hash(data["sample_ids"][selected].tolist()),
        "arms": results,
    }
    write_json(output / f"resource_pilot_{device.replace(':', '_')}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "pilot", "train", "aggregate"), default="prepare"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(args.threads)
    if args.stage == "prepare":
        result = prepare(args.output_dir, args.memory_dir)
    else:
        lock, tokens, data, _ = load_prepared(args.output_dir)
        if args.stage == "pilot":
            result = resource_pilot(args.output_dir, lock, tokens, data, args.device)
        elif args.stage == "aggregate":
            result = aggregate(args.output_dir, lock, data)
        else:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            arms = [args.arm] if args.arm else lock["protocol"]["arms"]
            folds = [args.fold] if args.fold is not None else lock["protocol"]["outer_folds"]
            for fold in folds:
                for arm in arms:
                    workload(args.output_dir, lock, tokens, data, arm, fold, args.device)
            result = {
                "status": "REQUESTED_ORDERED_WORKLOADS_COMPLETE",
                "arms": arms,
                "folds": folds,
            }
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
