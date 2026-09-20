"""Extract frozen ViTPose measurements for the 128-center body-witness pilot.

This label-free stage is provisionally authorized after one completed blinded
review.  It cannot authorize task-model fitting or a final availability pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("Pose extraction requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.audit_okutama_body_witness import audit_target_sensitivity  # noqa: E402
from experiments.pilot_okutama_body_witness import (  # noqa: E402
    LANDMARKS,
    exact_fov_pose_tensor,
)
from hac.body_witness_data import (  # noqa: E402
    MASK_IDS,
    MASKED_SLOT_ORDER,
    BodyCropGeometry,
    common_acquisition_mask,
    corrupt_withheld_for_teacher,
)
from hac.vitpose_measurement import (  # noqa: E402
    COCO_LANDMARK_INDICES,
    measure_pose_heatmaps,
)
from hac.vitpose_parity import (  # noqa: E402
    CONVERTED_CHECKPOINT_SHA256,
    load_locked_vitpose,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_body_witness_protocol.json"
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
PARITY = ROOT / ".runs/research_20260913/vitpose_parity_v1"
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_pose_pilot_v1"
CORRECTED_MODEL = PARITY / "corrected_model"
PARITY_RECEIPT = PARITY / "vitpose_computational_parity_receipt.json"
REVIEW = PILOT / "blinded_review/reviewer_a_completed_v1.csv"
REVIEW_RECEIPT = PILOT / "reviewer_a_provisional_receipt_v1.json"
FLIP_PAIRS = torch.tensor(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
    dtype=torch.long,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_new_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _geometry(record: dict[str, Any]) -> BodyCropGeometry:
    value = record["geometry"]
    return BodyCropGeometry(
        native_box=tuple(value["native_box"]),
        crop_box=tuple(value["crop_box"]),
        source_size=tuple(value["source_size"]),
        extent=float(value["extent"]),
    )


def _load_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    interim = protocol["pilot"]["human_reference"]["interim_single_review"]
    parity = json.loads(PARITY_RECEIPT.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_RECEIPT.read_text(encoding="utf-8"))
    manifest = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((PILOT / "pilot_selection.json").read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != 4
        or "frozen_pose_extraction" not in interim["authorized_scope"]
        or "task_model_fitting" not in interim["forbidden_scope"]
        or not interim["second_review_still_required"]
        or parity["status"]
        != "PASS_COMPUTATIONAL_MISMATCH_CORRECTED_OFFICIAL_BYTE_IDENTITY_UNRESOLVED"
        or review["status"] != "SINGLE_REVIEW_PROVISIONAL_COMPLETE"
        or review["normalized_sha256"] != sha256_file(REVIEW)
        or len(manifest) != 128
        or [row["sample_id"] for row in manifest]
        != [row["sample_id"] for row in selection]
    ):
        raise RuntimeError("Provisional pose extraction contract or ancestry differs")
    model_hash = sha256_file(CORRECTED_MODEL / "model.safetensors")
    if model_hash != CONVERTED_CHECKPOINT_SHA256:
        raise RuntimeError("Corrected ViTPose checkpoint bytes changed")
    return protocol, manifest


def plan(run: Path) -> dict[str, Any]:
    protocol, manifest = _load_contract()
    run.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PROVISIONAL_SINGLE_REVIEW_POSE_EXTRACTION_PLANNED",
        "scope": "frozen pose measurement only; no labels, task fits, or final gate",
        "rows": len(manifest),
        "inputs_per_row": 6,
        "total_model_inputs": len(manifest) * 6,
        "protocol_sha256": sha256_file(PROTOCOL),
        "checkpoint_sha256": sha256_file(CORRECTED_MODEL / "model.safetensors"),
        "corrected_config_sha256": sha256_file(CORRECTED_MODEL / "config.json"),
        "parity_receipt_sha256": sha256_file(PARITY_RECEIPT),
        "review_sha256": sha256_file(REVIEW),
        "review_receipt_sha256": sha256_file(REVIEW_RECEIPT),
        "crop_manifest_sha256": sha256_file(PILOT / "crop_manifest.json"),
        "pose_inference_lock": protocol["verifier"]["pose_inference_lock"],
        "preflight_rows": 16,
        "batch_size": 16,
        "task_labels_read": 0,
        "task_optimizer_updates": 0,
        "second_review_required_before_task_fitting": True,
    }
    write_new_json(run / "pose_execution_plan.json", result)
    return result


@dataclass(frozen=True)
class InferenceTask:
    row: int
    extent: int
    visible_mask: int | None
    pixels: np.ndarray
    affine: np.ndarray
    geometry: BodyCropGeometry


def _tasks(manifest: list[dict[str, Any]]) -> Iterator[InferenceTask]:
    for row_index, row in enumerate(manifest):
        for extent_index, extent in enumerate(("1.0", "1.25")):
            record = row["crops"][extent]
            raw = np.asarray(Image.open(PILOT / record["file"]).convert("RGB"))
            geometry = _geometry(record)
            pixels, preprocessing = exact_fov_pose_tensor(raw)
            affine = np.asarray(preprocessing["source_to_model"], dtype=np.float64)
            yield InferenceTask(row_index, extent_index, None, pixels, affine, geometry)
            for mask_index, mask_id in enumerate(MASK_IDS):
                corrupted = corrupt_withheld_for_teacher(raw, geometry, mask_id)
                changed, changed_preprocessing = exact_fov_pose_tensor(corrupted)
                changed_affine = np.asarray(
                    changed_preprocessing["source_to_model"], dtype=np.float64
                )
                if not np.array_equal(affine, changed_affine):
                    raise RuntimeError("Target corruption changed pose preprocessing geometry")
                yield InferenceTask(
                    row_index,
                    extent_index,
                    mask_index,
                    changed,
                    changed_affine,
                    geometry,
                )


def _allocate(rows: int) -> dict[str, np.ndarray]:
    return {
        "observed_pose": np.full((rows, 2, 8, 193), np.nan, np.float32),
        "pose_peak_magnitudes": np.full((rows, 2, 8), np.nan, np.float32),
        "pose_peak_raw_xy": np.full((rows, 2, 8, 2), np.nan, np.float32),
        "original_witness": np.full((rows, 4, 8, 193), np.nan, np.float32),
        "corrupted_witness": np.full((rows, 4, 8, 193), np.nan, np.float32),
        "source_to_model": np.full((rows, 2, 3, 3), np.nan, np.float64),
        "crop_decode_valid": np.ones((rows, 2), dtype=np.bool_),
    }


def _store_measurement(
    arrays: dict[str, np.ndarray], task: InferenceTask, heatmaps: np.ndarray
) -> None:
    selected = heatmaps[COCO_LANDMARK_INDICES]
    if task.visible_mask is None:
        full = measure_pose_heatmaps(selected, task.affine, task.geometry)
        arrays["observed_pose"][task.row, task.extent] = full.descriptor
        arrays["pose_peak_magnitudes"][task.row, task.extent] = full.peak_magnitude
        arrays["pose_peak_raw_xy"][task.row, task.extent] = full.peak_raw_xy
        arrays["source_to_model"][task.row, task.extent] = task.affine
        for mask_index, mask_id in enumerate(MASK_IDS):
            slot = task.extent * 2 + mask_index
            witness = measure_pose_heatmaps(
                selected, task.affine, task.geometry, visible_mask_id=mask_id
            )
            arrays["original_witness"][task.row, slot] = witness.descriptor
    else:
        mask_id = MASK_IDS[task.visible_mask]
        slot = task.extent * 2 + task.visible_mask
        witness = measure_pose_heatmaps(
            selected, task.affine, task.geometry, visible_mask_id=mask_id
        )
        arrays["corrupted_witness"][task.row, slot] = witness.descriptor


def infer(
    manifest: list[dict[str, Any]], *, batch_size: int = 16
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260913)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260913)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = load_locked_vitpose(CORRECTED_MODEL).eval().to(device)
    flip_pairs = FLIP_PAIRS.to(device)
    arrays = _allocate(len(manifest))
    digest = hashlib.sha256()
    pending: list[InferenceTask] = []
    total_tasks = len(manifest) * 6
    completed = 0
    started = time.perf_counter()

    def process(tasks: list[InferenceTask]) -> None:
        nonlocal completed
        pixels = torch.from_numpy(np.stack([task.pixels for task in tasks])).to(device)
        with torch.inference_mode():
            direct = model(pixel_values=pixels).heatmaps
            flipped = model(
                pixel_values=torch.flip(pixels, dims=(-1,)), flip_pairs=flip_pairs
            ).heatmaps
            averaged = ((direct + flipped) * 0.5).cpu().numpy().astype(np.float32)
        for task, heatmaps in zip(tasks, averaged, strict=True):
            digest.update(
                f"{task.row}|{task.extent}|{task.visible_mask}".encode("ascii")
            )
            digest.update(heatmaps.tobytes(order="C"))
            _store_measurement(arrays, task, heatmaps)
            completed += 1
        print(json.dumps({"pose_inputs": completed, "total": total_tasks}), flush=True)

    for task in _tasks(manifest):
        pending.append(task)
        if len(pending) == batch_size:
            process(pending)
            pending = []
    if pending:
        process(pending)
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "crop_decode_valid"):
        raise RuntimeError("Pose extraction left missing or nonfinite measurements")
    elapsed = time.perf_counter() - started
    details = {
        "device": str(device),
        "rows": len(manifest),
        "model_inputs": completed,
        "batch_size": batch_size,
        "flip_averaged_for_every_input": True,
        "heatmap_stream_sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "seconds_per_center": elapsed / len(manifest),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, details


def preflight(run: Path) -> dict[str, Any]:
    _, manifest = _load_contract()
    execution = json.loads((run / "pose_execution_plan.json").read_text(encoding="utf-8"))
    if execution["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("Pose plan is stale")
    arrays, details = infer(manifest[:16], batch_size=execution["batch_size"])
    projection = details["elapsed_seconds"] * 128 / 16
    result = {
        "status": "POSE_PRETRAINED_16_CENTER_PREFLIGHT_PASS",
        "scope": "label-free measurement runtime/shape only; not pose quality or task performance",
        "details": details,
        "projected_full_seconds": projection,
        "projection_below_20_minutes": projection < 1200,
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "all_finite": all(
            np.isfinite(value).all()
            for key, value in arrays.items()
            if key != "crop_decode_valid"
        ),
        "task_labels_read": 0,
        "task_optimizer_updates": 0,
    }
    if not result["all_finite"] or not result["projection_below_20_minutes"]:
        raise RuntimeError(f"Pose preflight failed: {result}")
    write_new_json(run / "pose_preflight_receipt.json", result)
    return result


def extract(run: Path) -> dict[str, Any]:
    _, manifest = _load_contract()
    execution = json.loads((run / "pose_execution_plan.json").read_text(encoding="utf-8"))
    preflight_receipt = json.loads(
        (run / "pose_preflight_receipt.json").read_text(encoding="utf-8")
    )
    if (
        execution["protocol_sha256"] != sha256_file(PROTOCOL)
        or preflight_receipt["status"] != "POSE_PRETRAINED_16_CENTER_PREFLIGHT_PASS"
        or not preflight_receipt["projection_below_20_minutes"]
    ):
        raise RuntimeError("Pose extraction gates are stale or failed")
    arrays, details = infer(manifest, batch_size=execution["batch_size"])
    arrays["sample_ids"] = np.asarray([row["sample_id"] for row in manifest])
    arrays["masked_slot_order"] = np.asarray(
        [f"{extent}|{mask}" for extent, mask in MASKED_SLOT_ORDER]
    )
    output = run / "pose_measurements.npz"
    _write_new_npz(output, arrays)
    result = {
        "status": "PROVISIONAL_SINGLE_REVIEW_POSE_MEASUREMENTS_COMPLETE",
        "scope": "frozen pose measurements; no labels or task fitting",
        "details": details,
        "artifact": str(output.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_file(output),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "checkpoint_sha256": execution["checkpoint_sha256"],
        "config_sha256": execution["corrected_config_sha256"],
        "protocol_sha256": execution["protocol_sha256"],
        "review_sha256": execution["review_sha256"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "task_labels_read": 0,
        "arftr_outputs_read": 0,
        "task_optimizer_updates": 0,
        "second_review_required_before_task_fitting": True,
    }
    write_new_json(run / "pose_extraction_receipt.json", result)
    return result


def _review_arrays(manifest: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    with REVIEW.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in manifest]:
        raise RuntimeError("Review order differs from pose measurements")
    visible = np.asarray(
        [[row[f"{landmark}_visible"] == "1" for landmark in LANDMARKS] for row in rows],
        dtype=np.bool_,
    )
    coordinates = np.full((128, 8, 2), np.nan, np.float64)
    for row_index, row in enumerate(rows):
        for landmark_index, landmark in enumerate(LANDMARKS):
            if visible[row_index, landmark_index]:
                coordinates[row_index, landmark_index] = [
                    float(row[f"{landmark}_x_crop"]),
                    float(row[f"{landmark}_y_crop"]),
                ]
    return visible, coordinates


def summarize(run: Path) -> dict[str, Any]:
    protocol, manifest = _load_contract()
    extraction = json.loads((run / "pose_extraction_receipt.json").read_text(encoding="utf-8"))
    measurement_path = run / "pose_measurements.npz"
    if extraction["artifact_sha256"] != sha256_file(measurement_path):
        raise RuntimeError("Pose measurement artifact changed")
    with np.load(measurement_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    visible, coordinates = _review_arrays(manifest)
    visible_count = visible.sum(axis=1)
    category_present = np.stack(
        [visible[:, start : start + 2].any(axis=1) for start in range(0, 8, 2)], axis=1
    ).all(axis=1)
    anatomical_pass = (visible_count >= 6) & category_present

    predicted = arrays["pose_peak_raw_xy"][:, 0]
    errors = np.linalg.norm(predicted - coordinates, axis=2)
    actor_heights = np.asarray(
        [row["crops"]["1.0"]["geometry"]["native_box"][3] - row["crops"]["1.0"]["geometry"]["native_box"][1] for row in manifest]
    )
    normalized_errors = errors / actor_heights[:, None]
    accurate = visible & (normalized_errors <= 0.10)
    accurate_fraction = accurate.sum(axis=1) / np.maximum(visible_count, 1)
    whole_crop_pass = anatomical_pass & (accurate_fraction >= 0.8)
    scenarios = np.asarray([row["scenario"] for row in manifest])
    scenario_pass = {
        scenario: int(whole_crop_pass[scenarios == scenario].sum())
        for scenario in sorted(set(scenarios))
    }
    passing = int(whole_crop_pass.sum())
    final_threshold = protocol["pilot"]["availability_gate"]["passing_crops_min"]
    provisional_availability = {
        "status": (
            "PROVISIONAL_PASS_SECOND_REVIEW_REQUIRED"
            if passing >= final_threshold and min(scenario_pass.values()) >= 1
            else "PROVISIONAL_FAIL_SECOND_REVIEW_REQUIRED"
        ),
        "reviewer": "reviewer_a",
        "anatomical_visibility_pass": int(anatomical_pass.sum()),
        "whole_crop_pose_pass": passing,
        "required_whole_crop_pass": final_threshold,
        "scenario_whole_crop_pass": scenario_pass,
        "all_scenarios_have_pass": min(scenario_pass.values()) >= 1,
        "mean_visible_landmarks": float(visible_count.mean()),
        "visible_localization_fraction_mean": float(accurate_fraction.mean()),
        "second_review_required_for_final_status": True,
        "task_fitting_authorized": False,
    }
    write_new_json(run / "provisional_availability_receipt.json", provisional_availability)

    sensitivity = audit_target_sensitivity(
        arrays["original_witness"],
        arrays["corrupted_witness"],
        review_pass=anatomical_pass,
        threshold=protocol["pilot"]["target_sensitivity_gate"][
            "total_variation_strict_min"
        ],
    )
    sensitivity["status"] = f"PROVISIONAL_{sensitivity['status']}_SECOND_REVIEW_REQUIRED"
    sensitivity["task_fitting_authorized"] = False
    write_new_json(run / "provisional_target_sensitivity_receipt.json", sensitivity)

    observable = common_acquisition_mask(
        arrays["crop_decode_valid"], arrays["pose_peak_magnitudes"]
    )
    agreement = {
        "observable_available": int(observable.sum()),
        "human_whole_crop_pass": passing,
        "both": int((observable & whole_crop_pass).sum()),
        "observable_only": int((observable & ~whole_crop_pass).sum()),
        "human_only": int((~observable & whole_crop_pass).sum()),
        "neither": int((~observable & ~whole_crop_pass).sum()),
    }
    result = {
        "status": "PROVISIONAL_SINGLE_REVIEW_MEASUREMENT_SUMMARY_COMPLETE",
        "availability": provisional_availability,
        "target_sensitivity": {
            key: value for key, value in sensitivity.items() if key != "values"
        },
        "observable_vs_human": agreement,
        "review_sha256": sha256_file(REVIEW),
        "pose_measurements_sha256": sha256_file(measurement_path),
        "task_labels_read": 0,
        "arftr_outputs_read": 0,
        "task_optimizer_updates": 0,
        "task_fitting_authorized": False,
        "next": "obtain reviewer_b, resolve disagreements without pose access, then recompute final gate",
    }
    write_new_json(run / "provisional_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "preflight", "extract", "summarize"), required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "plan":
        result = plan(run)
    elif args.stage == "preflight":
        result = preflight(run)
    elif args.stage == "extract":
        result = extract(run)
    else:
        result = summarize(run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
