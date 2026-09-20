"""Prepare the blinded 128-center body-witness observation pilot.

Stages are deliberately separated. ``plan`` reads a pinned cohort container to
validate identity integrity but uses only immutable observation metadata for
selection;
``extract`` decodes exact verified native centers and writes label-free review
crops; ``runtime`` performs a random-weight compatibility smoke only; and
``summarize`` refuses to score until two completed blinded human reviews and a
pinned pretrained pose-output archive exist.  No stage fits a task classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.body_witness_data import (  # noqa: E402
    CROP_EXTENTS,
    PADDING_RGB,
    crop_raw_body,
    decode_exact_center,
    load_body_witness_cohort,
    load_verified_source_map,
    make_body_crop_geometry,
    resolve_center_source,
    select_measurement_pilot,
    verify_video_file,
)
from hac.okutama_native_video import EXPECTED_SCENARIOS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("okutama_body_witness_protocol.json")
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_new_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pilot = protocol["pilot"]
    observations = protocol["observations"]
    if (
        protocol.get("study_id") != "okutama_body_witness_v1"
        or pilot.get("rows") != 128
        or pilot.get("selection", {}).get("salt") != "hac-body-witness-v1|"
        or tuple(observations.get("crop_extents", ())) != CROP_EXTENTS
        or observations.get("spatial_masks", {}).get("excluded_band_fraction") != 0.05
    ):
        raise RuntimeError("Body-witness protocol differs from the predeclared observation contract")
    return protocol


def selection_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort = load_body_witness_cohort(ROOT)
    selected = select_measurement_pilot(cohort.observations)
    counts = Counter(row.scenario for row in selected)
    expected = {
        scenario: 12 if index < 7 else 11
        for index, scenario in enumerate(sorted(EXPECTED_SCENARIOS))
    }
    if len(selected) != 128 or counts != expected:
        raise RuntimeError("Fixed label-blind pilot quota changed")
    rows = [
        {
            "selection_index": index,
            "sample_id": row.sample_id,
            "recording": row.recording,
            "track": row.track,
            "scenario": row.scenario,
            "center_frame": row.center_frame,
        }
        for index, row in enumerate(selected)
    ]
    receipt = {
        "status": "LABEL_BLIND_SELECTION_COMPLETE",
        "rows": len(rows),
        "scenario_counts": dict(sorted(counts.items())),
        "selection_rule": "lexical scenario quota; SHA256(salt+sample_id); one recording-track first pass",
        "salt": "hac-body-witness-v1|",
        "sample_ids_sha256": hashlib.sha256(
            json.dumps([row["sample_id"] for row in rows], separators=(",", ":")).encode()
        ).hexdigest(),
        "training_label_rows_accessed_by_cohort_integrity_loader": 4977,
        "training_labels_used_by_selection": 0,
        "arftr_probability_rows_accessed_by_cohort_integrity_loader": 4977,
        "arftr_outputs_used_by_selection": 0,
        "error_membership_read_by_selection": 0,
        "cohort_provenance": cohort.provenance,
    }
    return rows, receipt


def review_fields() -> list[str]:
    result = ["sample_id", "reviewer_id", "review_complete", "notes"]
    for landmark in LANDMARKS:
        result.extend(
            (f"{landmark}_visible", f"{landmark}_x_crop", f"{landmark}_y_crop")
        )
    return result


def blank_review_rows(rows: list[dict[str, Any]], reviewer: str) -> list[dict[str, Any]]:
    return [
        {
            **{field: "" for field in review_fields()},
            "sample_id": row["sample_id"],
            "reviewer_id": reviewer,
            "review_complete": "",
        }
        for row in rows
    ]


def plan(run: Path) -> dict[str, Any]:
    protocol = read_protocol()
    run.mkdir(parents=True, exist_ok=True)
    rows, receipt = selection_rows()
    write_new_json(run / "pilot_selection.json", rows)
    write_new_json(run / "selection_receipt.json", receipt)
    review = run / "blinded_review"
    for reviewer in ("reviewer_a", "reviewer_b"):
        write_new_csv(
            review / f"{reviewer}.csv",
            blank_review_rows(rows, reviewer),
            review_fields(),
        )
    execution = {
        "status": "OBSERVATION_PILOT_PLANNED",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(PROTOCOL),
        },
        "source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                Path(__file__),
                ROOT / "src/hac/body_witness_data.py",
                ROOT / "src/hac/body_witness.py",
                ROOT / "experiments/audit_okutama_body_witness.py",
            )
        },
        "pilot_selection_sha256": sha256_file(run / "pilot_selection.json"),
        "selection_receipt_sha256": sha256_file(run / "selection_receipt.json"),
        "authorization": protocol.get("execution_authorization", {}),
        "commands": {
            "native_crop_extraction": (
                f'& "{ROOT / ".venv/Scripts/python.exe"}" experiments/pilot_okutama_body_witness.py '
                f'--stage extract --run "{run}"'
            ),
            "runtime_smoke": (
                f'& "{ROOT / ".venv/Scripts/python.exe"}" experiments/pilot_okutama_body_witness.py '
                f'--stage runtime --run "{run}"'
            ),
            "summary_after_human_review_and_pose": (
                f'& "{ROOT / ".venv/Scripts/python.exe"}" experiments/pilot_okutama_body_witness.py '
                f'--stage summarize --run "{run}"'
            ),
        },
        "next_gate": "exact native crop extraction; then two independent blinded reviews and pinned pose outputs",
        "task_models_fitted": 0,
    }
    write_new_json(run / "execution_plan.json", execution)
    return execution


def _review_index(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        sample = html.escape(row["sample_id"])
        cards.append(
            "<section><h2>{index:03d} {sample}</h2>"
            "<img src='images/{sample}__extent-1p0.png' alt='primary native crop'>"
            "<img src='images/{sample}__extent-1p25.png' alt='stability native crop'>"
            "</section>".format(index=row["selection_index"], sample=sample)
        )
    return """<!doctype html><meta charset='utf-8'><title>Blinded body review</title>
<style>body{font:14px system-ui;background:#eee}section{background:white;margin:1rem;padding:1rem}
img{margin-right:1rem;max-width:45%;min-width:260px;image-rendering:auto}h2{font-size:14px}</style>
<h1>Blinded body-landmark feasibility review</h1>
<p>Left: exact 1.0 supplied-box crop. Right: fixed 1.25 extent. Zoom the browser as needed.
Do not consult action labels, ARFTR predictions, errors, or pose estimates. Enter primary-crop pixel
coordinates in your assigned CSV; mark hidden/unlocalizable landmarks visible=0 and leave coordinates blank.</p>
""" + "\n".join(cards)


def extract(run: Path) -> dict[str, Any]:
    read_protocol()
    selection_path = run / "pilot_selection.json"
    plan_path = run / "execution_plan.json"
    receipt_path = run / "selection_receipt.json"
    if not selection_path.is_file() or not plan_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("Run the immutable plan stage before extraction")
    execution = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        execution["protocol"]["sha256"] != sha256_file(PROTOCOL)
        or execution.get("pilot_selection_sha256") != sha256_file(selection_path)
        or execution["selection_receipt_sha256"] != sha256_file(receipt_path)
        or any(
            sha256_file(ROOT / relative) != expected
            for relative, expected in execution["source_files"].items()
        )
    ):
        raise RuntimeError("Pilot plan ancestry changed after selection; extraction refused")
    rows = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected_digest = hashlib.sha256(
        json.dumps([row["sample_id"] for row in rows], separators=(",", ":")).encode()
    ).hexdigest()
    if len(rows) != 128 or selected_digest != selection_receipt["sample_ids_sha256"]:
        raise RuntimeError("Pilot selected IDs differ from their immutable receipt")
    cohort = load_body_witness_cohort(ROOT)
    by_id = {row.sample_id: row for row in cohort.observations}
    source_map = load_verified_source_map(ROOT)
    image_dir = run / "blinded_review/images"
    image_dir.mkdir(parents=True, exist_ok=False)
    verified = {}
    for recording in sorted({row["recording"] for row in rows}):
        representative = next(row for row in rows if row["recording"] == recording)
        request = resolve_center_source(by_id[representative["sample_id"]], source_map)
        verified[recording] = verify_video_file(request)
    started = time.perf_counter()
    audits, crop_rows = [], []
    for position, row in enumerate(rows):
        observation = by_id[row["sample_id"]]
        request = resolve_center_source(observation, source_map)
        rgb, audit = decode_exact_center(request, verified_video=verified[observation.recording])
        audits.append(audit)
        crop_record: dict[str, Any] = {
            "selection_index": position,
            "sample_id": observation.sample_id,
            "scenario": observation.scenario,
            "decode_valid": bool(rgb is not None),
            "source_request": {
                "native_index": request.native_index,
                "target_pts": request.target_pts,
                "native_seconds": request.native_seconds,
                "nominal_seconds": request.nominal_seconds,
                "native_box": request.native_box,
            },
            "crops": {},
        }
        if rgb is not None and request.native_box is not None:
            for extent in CROP_EXTENTS:
                geometry = make_body_crop_geometry(
                    request.native_box, extent=extent, source_size=request.video.source_size
                )
                raw = crop_raw_body(rgb, geometry)
                name = f"{observation.sample_id}__extent-{str(extent).replace('.', 'p')}.png"
                Image.fromarray(raw).save(image_dir / name, compress_level=6)
                crop_record["crops"][str(extent)] = {
                    "file": f"blinded_review/images/{name}",
                    "sha256": sha256_file(image_dir / name),
                    "raw_rgb_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "geometry": geometry.provenance(),
                }
        crop_rows.append(crop_record)
        if (position + 1) % 16 == 0:
            print(json.dumps({"decoded": position + 1, "total": len(rows)}), flush=True)
    elapsed = time.perf_counter() - started
    write_new_json(run / "crop_manifest.json", crop_rows)
    (run / "blinded_review/index.html").write_text(_review_index(rows), encoding="utf-8")
    result = {
        "status": "NATIVE_REVIEW_CROPS_COMPLETE",
        "centers": len(rows),
        "exact_decode_successes": sum(row["decode_valid"] for row in crop_rows),
        "exact_decode_failures": sum(not row["decode_valid"] for row in crop_rows),
        "crop_files": sum(len(row["crops"]) for row in crop_rows),
        "elapsed_seconds": elapsed,
        "seconds_per_center": elapsed / len(rows),
        "first_16_elapsed_projection_seconds": None,
        "labels_used": 0,
        "arftr_outputs_used": 0,
        "neighbor_fallbacks": 0,
        "video_receipts": verified,
        "decode_audits": audits,
        "crop_manifest_sha256": sha256_file(run / "crop_manifest.json"),
    }
    write_new_json(run / "extraction_receipt.json", result)
    return result


def exact_fov_pose_tensor(raw_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Letterbox the exact raw FOV to 192x256; never ask a processor to pad a box."""

    raw = np.asarray(raw_rgb)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3 or min(raw.shape[:2]) < 1:
        raise ValueError("Pose input must be a nonempty uint8 RGB crop")
    source_height, source_width = raw.shape[:2]
    target_width, target_height = 192, 256
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    resized = np.asarray(
        Image.fromarray(raw).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    )
    left, top = (target_width - resized_width) // 2, (target_height - resized_height) // 2
    canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
    canvas[:] = PADDING_RGB
    canvas[top : top + resized_height, left : left + resized_width] = resized
    normalized = canvas.astype(np.float32) / 255.0
    normalized = (normalized - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    tensor = normalized.transpose(2, 0, 1)
    affine = np.asarray(
        [[resized_width / source_width, 0, left], [0, resized_height / source_height, top], [0, 0, 1]],
        dtype=np.float64,
    )
    return tensor, {
        "method": "exact_fov_letterbox_then_bilinear_resize",
        "source_size": [source_width, source_height],
        "target_size": [target_width, target_height],
        "resized_size": [resized_width, resized_height],
        "padding_left_top": [left, top],
        "source_to_model": affine.tolist(),
        "processor_bbox_padding_disabled": True,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }


def runtime(run: Path) -> dict[str, Any]:
    """No-download, random-weight shape smoke; never a quality result."""

    read_protocol()
    import torch
    import transformers
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backbone = VitPoseBackboneConfig(
        image_size=(256, 192),
        patch_size=(16, 16),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_experts=1,
        out_indices=[12],
        layer_norm_eps=1e-6,
    )
    config = VitPoseConfig(
        backbone_config=backbone, use_simple_decoder=False, num_labels=17
    )
    model = VitPoseForPoseEstimation(config).eval().to(device)
    pixels = torch.zeros(1, 3, 256, 192, device=device)
    with torch.inference_mode():
        heatmaps = model(pixel_values=pixels).heatmaps
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    passed = tuple(heatmaps.shape) == (1, 17, 64, 48) and bool(torch.isfinite(heatmaps).all())
    result = {
        "status": "RUNTIME_SMOKE_PASS_PRETRAINED_CHECKPOINT_BLOCKED" if passed else "FAIL",
        "claim_scope": "random-weight construction/forward compatibility only; no pose quality output",
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "configuration": {
            "architecture": "ViTPose-B",
            "image_size": [256, 192],
            "classic_decoder": True,
            "use_simple_decoder": False,
            "out_indices": [12],
            "layer_norm_eps": 1e-6,
            "joints": 17,
        },
        "output_shape": list(heatmaps.shape),
        "finite": bool(torch.isfinite(heatmaps).all()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_construct_and_forward_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "installed_dependencies": {
            package: importlib.metadata.version(package)
            if importlib.util.find_spec(package.replace("-", "_")) is not None
            else None
            for package in ("transformers", "torch", "mmpose", "mmcv", "mmengine")
        },
        "pretrained_checkpoint": None,
        "checkpoint_sha256": None,
        "quality_outputs_authorized": False,
        "blocker": "exact official COCO classic-decoder checkpoint bytes and conversion/parity receipt absent",
        "preprocessing": "explicit exact-FOV adapter implemented; no implicit bbox padding",
    }
    write_new_json(run / "pose_runtime_preflight.json", result)
    del heatmaps, pixels, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _read_completed_review(path: Path, reviewer: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 128 or any(row["reviewer_id"] != reviewer for row in rows):
        raise RuntimeError(f"{reviewer} sheet identity/population changed")
    if any(str(row["review_complete"]).strip().lower() not in {"1", "true"} for row in rows):
        raise RuntimeError(f"{reviewer} sheet is not complete")
    return {row["sample_id"]: row for row in rows}


def summarize(run: Path) -> dict[str, Any]:
    """Fail closed until genuinely independent references and pose outputs exist."""

    read_protocol()
    required = {
        "reviewer_a": run / "blinded_review/reviewer_a.csv",
        "reviewer_b": run / "blinded_review/reviewer_b.csv",
    }
    reviews = {name: _read_completed_review(path, name) for name, path in required.items()}
    pose_path = run / "pose_outputs.npz"
    pose_receipt = run / "pose_checkpoint_receipt.json"
    if not pose_path.is_file() or not pose_receipt.is_file():
        raise RuntimeError(
            "Pinned pretrained pose outputs/checkpoint receipt are absent; availability cannot be scored"
        )
    # Parsing/scoring is intentionally not reached before both independent sheets
    # and pose artifacts exist. It will be added against their frozen schemas,
    # avoiding invention of human judgments or a checkpoint-specific convention.
    raise RuntimeError(
        f"Reviews ({sum(len(value) for value in reviews.values())} rows) exist, but pose schema "
        "must be independently audited before unblinding/scoring"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "extract", "runtime", "summarize"), required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = globals()[args.stage](args.run.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
