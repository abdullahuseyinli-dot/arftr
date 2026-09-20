"""Label-blind parity preflight for the source-causal body-parts hypothesis.

This preflight deliberately stops before image decoding, DINO execution, label
access, or classifier fitting.  It verifies the frozen paired source cache,
row/frame identity, array contracts, and the deterministic whole/upper/lower
crop geometry used by the next posture-only probe.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hac.absolute_body_evidence import make_crop_geometry, source_actor_boxes

CACHE = ROOT / ".runs/research_20260908/native4k_paired_full/features"
EXISTING = ROOT / ".runs/research_20260908/dense_tokens_full"
CROP_MANIFEST = ROOT / ".runs/research_20260913/body_witness_pilot_v1/crop_manifest.json"
SELECTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
DEFAULT_RUN = ROOT / ".runs/research_20260916/source_causal_body_parts_v1"
ROWS = 4977
SOURCE_ARMS = ("supplied720", "exact_native_downsample720", "native4k")
FEATURE_FILES = {
    "supplied720_grid3": EXISTING / "tokens_grid3.npy",
    "supplied720_grid12": EXISTING / "tokens_grid12.npy",
    "native4k_grid3": CACHE / "native4k_grid3.npy",
    "native4k_grid12": CACHE / "native4k_grid12.npy",
    "exact_native_downsample720_grid3": CACHE / "exact4k_downsample720_grid3.npy",
    "exact_native_downsample720_grid12": CACHE / "exact4k_downsample720_grid12.npy",
}
EXPECTED_SHAPES = {
    "supplied720_grid3": (ROWS, 8, 9, 768),
    "supplied720_grid12": (ROWS, 8, 144, 768),
    "native4k_grid3": (ROWS, 8, 9, 768),
    "native4k_grid12": (ROWS, 8, 144, 768),
    "exact_native_downsample720_grid3": (ROWS, 8, 9, 768),
    "exact_native_downsample720_grid12": (ROWS, 8, 144, 768),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_contract(path: Path, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing frozen feature artifact: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != expected_shape or array.dtype != np.dtype("float16"):
        raise RuntimeError(
            f"Feature contract changed for {path.name}: {array.shape}/{array.dtype}"
        )
    # A deterministic sample over the complete row population catches corrupt
    # chunks without materialising the 8.8-GB fine-grid arrays.
    sample_rows = np.unique(np.linspace(0, len(array) - 1, 128, dtype=np.int64))
    sample = np.asarray(array[sample_rows])
    if not np.isfinite(sample).all():
        raise RuntimeError(f"Non-finite sampled feature values: {path.name}")
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": int(path.stat().st_size),
        "sample_rows_checked": int(len(sample_rows)),
        "sample_finite": True,
    }


def _geometry_receipts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    crops = json.loads(CROP_MANIFEST.read_text(encoding="utf-8"))
    if len(selection) != 128 or len(crops) != 128:
        raise RuntimeError("Pinned body-witness population is not exactly 128")
    if [row["selection_index"] for row in selection] != list(range(128)):
        raise RuntimeError("Selection indices are not contiguous")
    crop_by_id = {row["sample_id"]: row for row in crops}
    if set(crop_by_id) != {row["sample_id"] for row in selection}:
        raise RuntimeError("Selection and crop-manifest identities differ")
    rows: list[dict[str, Any]] = []
    for selected in selection:
        row = crop_by_id[selected["sample_id"]]
        native_geometry = row["crops"]["1.0"]["geometry"]
        native_box = tuple(float(x) for x in native_geometry["native_box"])
        source_size = tuple(int(x) for x in native_geometry["source_size"])
        if source_size != (3840, 2160):
            raise RuntimeError("Pinned native crop source size changed")
        box720 = tuple(x / 3.0 for x in native_box)
        boxes = source_actor_boxes(box720, source_size)
        geometry: dict[str, Any] = {}
        for source_id, box in boxes.items():
            source_size_for_arm = (1280, 720) if source_id != "N" else source_size
            for region in ("whole", "upper", "lower"):
                item = make_crop_geometry(
                    box,
                    source_size=source_size_for_arm,
                    source_id=source_id,
                    region_id=region,
                )
                geometry[f"{source_id}_{region}"] = {
                    "integer_box": list(item.integer_box),
                    "continuous_box": list(item.continuous_box),
                    "raw_size": list(item.raw_size),
                }
        # Continuous source geometry is invariant under the exact 3x mapping;
        # integer rounding is intentionally allowed to differ by one pixel.
        for region in ("whole", "upper", "lower"):
            low = np.asarray(geometry[f"R_{region}"]["continuous_box"])
            high = np.asarray(geometry[f"N_{region}"]["continuous_box"])
            if not np.allclose(high, 3.0 * low, atol=1e-8, rtol=0.0):
                raise RuntimeError(f"Source geometry scale mismatch: {selected['sample_id']}/{region}")
        rows.append(
            {
                "selection_index": int(selected["selection_index"]),
                "sample_id": selected["sample_id"],
                "geometry_digest": hashlib.sha256(
                    json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "regions": ["whole", "upper", "lower"],
                "source_arms": list(SOURCE_ARMS),
                "geometry": geometry,
            }
        )
    return rows, {
        "rows": len(rows),
        "selection_sha256": sha256_file(SELECTION),
        "crop_manifest_sha256": sha256_file(CROP_MANIFEST),
        "all_source_geometry_scale_checks": True,
        "image_decode_count": 0,
    }


def run(run: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run = run.resolve()
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty preflight run: {run}")
    run.mkdir(parents=True, exist_ok=True)

    request = json.loads((CACHE / "request.json").read_text(encoding="utf-8"))
    source_summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (CACHE / "independent_post_extraction_verification.json").read_text(encoding="utf-8")
    )
    binding = json.loads((CACHE / "execution_source_binding.json").read_text(encoding="utf-8"))
    if request.get("centers") != ROWS or request.get("classifier_fits") != 0 or request.get("action_values_used") != 0:
        raise RuntimeError("Frozen source request is not label-blind/no-fit")
    if set(request.get("arms", [])) != {"native4k", "exact4k_downsample720"}:
        raise RuntimeError("Unexpected paired source arms in frozen cache")
    if source_summary.get("status") != "FULL_PAIRED_NATIVE4K_FEATURE_EXTRACTION_COMPLETE":
        raise RuntimeError("Paired source cache is not marked complete")
    if verification.get("rows") != ROWS or not verification.get("sample_id_order_equals_existing720"):
        raise RuntimeError("Independent source-cache identity verification failed")
    if verification.get("classifier_accuracy_accessed") or verification.get("feature_values_changed_by_verification"):
        raise RuntimeError("Source cache verification accessed task outputs or changed features")
    if binding.get("original_request_modified") or binding.get("extraction_restarted"):
        raise RuntimeError("Source encoder binding indicates an unapproved rewrite")

    paired_ids = np.load(CACHE / "sample_ids.npy", allow_pickle=False)
    existing_ids = np.load(EXISTING / "sample_ids.npy", allow_pickle=False)
    paired_frames = np.load(CACHE / "source_frames.npy", allow_pickle=False)
    existing_frames = np.load(EXISTING / "source_frames.npy", allow_pickle=False)
    native_frames = np.load(CACHE / "native_frames.npy", allow_pickle=False)
    completed = np.load(CACHE / "completed.npy", allow_pickle=False)
    if len(paired_ids) != ROWS or not np.array_equal(paired_ids, existing_ids):
        raise RuntimeError("Paired and supplied720 sample-id order differs")
    if paired_frames.shape != (ROWS, 16) or existing_frames.shape != (ROWS, 16):
        raise RuntimeError("Source frame arrays have changed shape")
    if not np.array_equal(paired_frames, existing_frames):
        raise RuntimeError("Paired source cache changed the selected timestamps")
    if native_frames.shape != (ROWS, 16) or not completed.all():
        raise RuntimeError("Native source completion/frame contract failed")

    arrays = {
        name: _array_contract(path, EXPECTED_SHAPES[name])
        for name, path in FEATURE_FILES.items()
    }
    geometry_rows, geometry_summary = _geometry_receipts()
    write_json(run / "parity_rows.json", geometry_rows)
    result = {
        "status": "SOURCE_CAUSAL_BODY_PARTS_PHASE_0_PREFLIGHT_PASS",
        "label_blind": True,
        "labels_read": False,
        "arftr_outputs_read": False,
        "support_annotations_read": False,
        "human_review_read": False,
        "classifier_fits": 0,
        "image_decodes": 0,
        "rows": ROWS,
        "source_arms": list(SOURCE_ARMS),
        "regions": ["whole", "upper", "lower"],
        "paired_cache_request": {
            "centers": int(request["centers"]),
            "classifier_fits": int(request["classifier_fits"]),
            "action_values_used": int(request["action_values_used"]),
            "native_centers": int(request["native_centers"]),
            "fallback_rows": int(request["exact720_whole_clip_fallbacks"]),
        },
        "identity_checks": {
            "sample_id_order_equals_existing720": True,
            "source_frame_order_equals_existing720": True,
            "native_frame_shape": list(native_frames.shape),
            "all_completion_flags_true": True,
        },
        "array_contracts": arrays,
        "geometry": geometry_summary,
        "task_fit_authorized": False,
        "next_gate": "authorize_phase_1_only_after_independent_replay_audit",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run / "preflight.json", result)
    artifacts = ["preflight.json", "parity_rows.json"]
    receipt = {
        "status": "SOURCE_CAUSAL_BODY_PARTS_PHASE_0_RECEIPT",
        "run": str(run),
        "artifacts": {name: sha256_file(run / name) for name in artifacts},
        "inputs": {
            "request.json": sha256_file(CACHE / "request.json"),
            "summary.json": sha256_file(CACHE / "summary.json"),
            "independent_post_extraction_verification.json": sha256_file(
                CACHE / "independent_post_extraction_verification.json"
            ),
            "execution_source_binding.json": sha256_file(CACHE / "execution_source_binding.json"),
            "sample_ids.npy": sha256_file(CACHE / "sample_ids.npy"),
            "source_frames.npy": sha256_file(CACHE / "source_frames.npy"),
            "native_frames.npy": sha256_file(CACHE / "native_frames.npy"),
            "completed.npy": sha256_file(CACHE / "completed.npy"),
            "supplied720_sample_ids.npy": sha256_file(EXISTING / "sample_ids.npy"),
            "supplied720_source_frames.npy": sha256_file(EXISTING / "source_frames.npy"),
            "selection.json": sha256_file(SELECTION),
            "crop_manifest.json": sha256_file(CROP_MANIFEST),
        },
        "feature_artifact_receipts": {
            name: {
                "bytes": int(path.stat().st_size),
                "sha256_from_frozen_summary": source_summary["artifacts"][path.name]["sha256"],
            }
            for name, path in FEATURE_FILES.items()
            if path.name in source_summary.get("artifacts", {})
        },
    }
    write_json(run / "receipt.json", receipt)
    return result


def audit(run: Path) -> dict[str, Any]:
    run = run.resolve()
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    failures = []
    for name, digest in receipt["artifacts"].items():
        if not (run / name).exists() or sha256_file(run / name) != digest:
            failures.append(name)
    result = json.loads((run / "preflight.json").read_text(encoding="utf-8"))
    checks = {
        "receipt_artifacts_exact": not failures,
        "label_blind": result.get("label_blind") is True and not result.get("labels_read"),
        "all_identity_checks": all(result.get("identity_checks", {}).values()),
        "all_array_contracts_sample_finite": all(
            item.get("sample_finite") is True for item in result.get("array_contracts", {}).values()
        ),
        "geometry_scale_checks": result.get("geometry", {}).get("all_source_geometry_scale_checks") is True,
        "task_fit_not_authorized": result.get("task_fit_authorized") is False,
    }
    audit_result = {
        "status": "SOURCE_CAUSAL_BODY_PARTS_PHASE_0_INDEPENDENT_REPLAY_AUDIT",
        "passed": bool(all(checks.values()) and not failures),
        "checks": checks,
        "hash_failures": failures,
    }
    write_json(run / "independent_audit.json", audit_result)
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "audit"), default="preflight")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.stage == "audit":
        print(json.dumps(audit(args.run), indent=2, sort_keys=True, allow_nan=False))
        return
    result = run(args.run)
    audit_result = audit(args.run)
    print(json.dumps({"preflight": result, "audit": audit_result}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
