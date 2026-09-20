"""Materialize label-blind transport evidence for the locked 128-center screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.center_completion_screen import build_gate_features_21  # noqa: E402
from hac.center_evidence_completion import (  # noqa: E402
    bilinear_transport,
    donor_confidence_logits,
    fit_visible_anchor_transport,
    transport_consensus,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/okutama_center_evidence_completion_protocol.json"
PLAN = ROOT / ".runs/research_20260913/center_evidence_completion_v1/plan_v8.json"
CACHE = ROOT / ".runs/research_20260913/center_evidence_completion_cache_v1"
CACHE_AUDIT = (
    ROOT
    / ".runs/research_20260913/center_evidence_completion_cache_v1_audit/audit_receipt.json"
)
SELECTION = ROOT / ".runs/research_20260913/body_witness_pilot_v1/pilot_selection.json"
DEFAULT_OUTPUT = (
    ROOT / ".runs/research_20260913/center_evidence_completion_transport_v1"
)
EXPECTED_PROTOCOL_SHA256 = (
    "f2e95656cfd179e72522d02103d2542e96b4124ccb76c5f4c0c1706783505244"
)
EXPECTED_PLAN_SHA256 = (
    "b7e148c8efe1e72fc0000ffd0713aecf16a3a4e21387d9f520a9dac6e5e6b63c"
)
EXPECTED_CACHE_SUMMARY_SHA256 = (
    "5eec07a433c189de949abeff70e1987fb8fbfb873cb4c0f308921b5833d4f201"
)
EXPECTED_CACHE_AUDIT_SHA256 = (
    "269eccee054f3313ed254445e5fb5a9cf71a581bf14dc1b4f1cd9306c3e6b1e2"
)
ROWS = 128
MASKS = 2
GRID = 27
DIMENSION = 768
DONORS = 4
OFFSETS = (-8, -4, 4, 7)
ARRAY_SPECS = {
    "true_transport_tokens.npy": (np.float16, (ROWS, MASKS, GRID, GRID, DIMENSION)),
    "true_transport_available.npy": (np.bool_, (ROWS, MASKS, GRID, GRID)),
    "true_gate_features.npy": (np.float32, (ROWS, MASKS, GRID, GRID, 21)),
    "wrong_transport_tokens.npy": (np.float16, (ROWS, MASKS, GRID, GRID, DIMENSION)),
    "wrong_transport_available.npy": (np.bool_, (ROWS, MASKS, GRID, GRID)),
    "wrong_gate_features.npy": (np.float32, (ROWS, MASKS, GRID, GRID, 21)),
    "repeated_transport_tokens.npy": (
        np.float16,
        (ROWS, MASKS, GRID, GRID, DIMENSION),
    ),
    "repeated_transport_available.npy": (np.bool_, (ROWS, MASKS, GRID, GRID)),
    "repeated_gate_features.npy": (np.float32, (ROWS, MASKS, GRID, GRID, 21)),
}


def sha256_file_stable(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _open_output_arrays(output: Path) -> dict[str, np.memmap]:
    arrays = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        arrays[name] = np.lib.format.open_memmap(
            output / name, mode="w+", dtype=dtype, shape=shape
        )
        arrays[name][...] = False if dtype is np.bool_ else 0
        arrays[name].flush()
    return arrays


def validate_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file_stable(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Protocol changed after the label-blind transport lock")
    if sha256_file_stable(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Plan changed after the label-blind transport lock")
    if sha256_file_stable(CACHE / "summary.json") != EXPECTED_CACHE_SUMMARY_SHA256:
        raise RuntimeError("Full-cache summary changed after independent replay")
    if sha256_file_stable(CACHE_AUDIT) != EXPECTED_CACHE_AUDIT_SHA256:
        raise RuntimeError("Full-cache audit changed after the transport lock")
    summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if (
        summary["status"] != "CENTER_EVIDENCE_COMPLETION_FULL_CACHE_COMPLETE"
        or summary["scientific_gate_evaluated"] is not False
        or summary["task_training_authorized"] is not False
        or summary["encoded_inputs"] != 1392
        or audit["status"] != "INDEPENDENT_FULL_CACHE_REPLAY_AUDIT_PASS"
        or len(selection) != ROWS
        or len({row["sample_id"] for row in selection}) != ROWS
    ):
        raise RuntimeError("Cache/audit authorization differs from the transport plan")
    for name, entry in summary["artifacts"].items():
        path = CACHE / name
        if (
            name not in {
                "teacher_tokens.npy",
                "masked_tokens.npy",
                "neighbor_tokens.npy",
                "wrong_neighbor_tokens.npy",
                "target_masks.npy",
                "visible_masks.npy",
                "neighbor_valid.npy",
                "wrong_neighbor_valid.npy",
                "wrong_slot_available.npy",
            }
            or list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
            != entry["shape"]
            or sha256_file_stable(path) != entry["sha256"]
        ):
            raise RuntimeError(f"Audited cache artifact changed: {name}")
    return summary, selection


def _local_statistics(matches, target_mask: np.ndarray, observed: np.ndarray):
    count = int(target_mask.sum())
    cosine = np.full(count, np.nan, dtype=np.float32)
    distance = np.full(count, np.inf, dtype=np.float32)
    if not len(matches.center_xy):
        return cosine, distance
    target_xy = np.argwhere(target_mask)[:, ::-1].astype(np.float64)
    for index in np.flatnonzero(observed):
        distances = np.linalg.norm(matches.center_xy - target_xy[index], axis=1)
        nearest = np.argsort(distances, kind="stable")[:4]
        cosine[index] = np.median(matches.cosine[nearest])
        distance[index] = distances[nearest[0]]
    return cosine, distance


def build_transport_row(
    center_tokens: np.ndarray,
    donors: np.ndarray,
    target_mask: np.ndarray,
    center_valid: np.ndarray,
    donor_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_count = int(target_mask.sum())
    donor_values = np.zeros((DONORS, target_count, DIMENSION), dtype=np.float32)
    donor_observed = np.zeros((DONORS, target_count), dtype=bool)
    donor_logits = np.full((DONORS, target_count), -np.inf, dtype=np.float64)
    residuals = np.full((target_count, DONORS), np.inf, dtype=np.float32)
    local_cosines = np.full((target_count, DONORS), np.nan, dtype=np.float32)
    nearest_distances = np.full((target_count, DONORS), np.inf, dtype=np.float32)
    receipts = []
    for donor_index, offset in enumerate(OFFSETS):
        transform, matches = fit_visible_anchor_transport(
            center_tokens.astype(np.float32),
            donors[donor_index].astype(np.float32),
            target_mask,
            center_valid=center_valid,
            neighbor_valid=donor_valid[donor_index],
        )
        values, observed = bilinear_transport(
            donors[donor_index].astype(np.float32),
            target_mask,
            transform,
            neighbor_valid=donor_valid[donor_index],
        )
        logits = donor_confidence_logits(matches, transform, target_mask, observed)
        cosine, distance = _local_statistics(matches, target_mask, observed)
        donor_values[donor_index] = values
        donor_observed[donor_index] = observed
        donor_logits[donor_index] = logits
        if transform.valid:
            residuals[observed, donor_index] = transform.weighted_residual
        local_cosines[:, donor_index] = cosine
        nearest_distances[:, donor_index] = distance
        receipts.append(
            {
                "offset_frames": offset,
                "match_count": transform.match_count,
                "valid": transform.valid,
                "reason": transform.reason,
                "weighted_residual": (
                    transform.weighted_residual
                    if math.isfinite(transform.weighted_residual)
                    else None
                ),
                "transported_targets": int(observed.sum()),
            }
        )
    consensus = transport_consensus(
        donor_values,
        donor_observed,
        donor_logits,
        np.zeros((target_count, DIMENSION), dtype=np.float32),
    )
    gate = build_gate_features_21(
        torch.from_numpy(np.transpose(donor_values, (1, 0, 2))),
        torch.from_numpy(donor_observed.T),
        torch.from_numpy(residuals),
        torch.from_numpy(local_cosines),
        torch.from_numpy(nearest_distances),
    ).numpy()
    return consensus.features, consensus.available, gate, receipts


def build_repeated_row(
    center_tokens: np.ndarray,
    target_mask: np.ndarray,
    center_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.argwhere(target_mask)
    visible_positions = np.argwhere(center_valid)
    if len(visible_positions) < 12:
        raise RuntimeError("Repeated-center control lacks 12 visible anchors")
    values = center_tokens[target_mask].astype(np.float32)
    observed = np.ones((len(values), DONORS), dtype=bool)
    residuals = np.zeros((len(values), DONORS), dtype=np.float32)
    cosines = np.ones((len(values), DONORS), dtype=np.float32)
    distances = np.empty((len(values), DONORS), dtype=np.float32)
    for index, position in enumerate(positions):
        nearest = np.linalg.norm(visible_positions - position, axis=1).min()
        distances[index] = nearest
    donor_values = np.repeat(values[:, None, :], DONORS, axis=1)
    gate = build_gate_features_21(
        torch.from_numpy(donor_values),
        torch.from_numpy(observed),
        torch.from_numpy(residuals),
        torch.from_numpy(cosines),
        torch.from_numpy(distances),
    ).numpy()
    return values, np.ones(len(values), dtype=bool), gate


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("Transport output must be new or empty")
    else:
        output.mkdir(parents=True)
    cache_summary, selection = validate_inputs()
    source_files = [
        Path(__file__),
        ROOT / "src/hac/center_evidence_completion.py",
        ROOT / "src/hac/center_completion_screen.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file_stable(path)
        for path in source_files
    }
    request = {
        "status": "CENTER_EVIDENCE_COMPLETION_TRANSPORT_REQUEST_LOCKED",
        "scope": "label-blind true/wrong/repeated transport materialization; no fits",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "cache_summary_sha256": EXPECTED_CACHE_SUMMARY_SHA256,
        "cache_audit_sha256": EXPECTED_CACHE_AUDIT_SHA256,
        "source_hashes": source_hashes,
        "sample_ids": [row["sample_id"] for row in selection],
        "conditions": ["true_track", "wrong_track", "repeated_masked_center"],
        "zero_access_contract": {
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "optimizer_updates": 0,
        },
    }
    write_json_exclusive(output / "request.json", request)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in source_files:
        shutil.copyfile(path, snapshot / path.name)
    arrays = _open_output_arrays(output)
    masked = np.load(CACHE / "masked_tokens.npy", mmap_mode="r", allow_pickle=False)
    true_donors = np.load(
        CACHE / "neighbor_tokens.npy", mmap_mode="r", allow_pickle=False
    )
    wrong_donors = np.load(
        CACHE / "wrong_neighbor_tokens.npy", mmap_mode="r", allow_pickle=False
    )
    target_masks = np.load(CACHE / "target_masks.npy", mmap_mode="r", allow_pickle=False)
    visible_masks = np.load(
        CACHE / "visible_masks.npy", mmap_mode="r", allow_pickle=False
    )
    true_valid = np.load(CACHE / "neighbor_valid.npy", mmap_mode="r", allow_pickle=False)
    wrong_valid = np.load(
        CACHE / "wrong_neighbor_valid.npy", mmap_mode="r", allow_pickle=False
    )
    diagnostics = []
    started = time.perf_counter()
    for center in range(ROWS):
        for mask_id in range(MASKS):
            target = target_masks[center, mask_id]
            visible = visible_masks[center, mask_id]
            y, x = np.nonzero(target)
            for condition, donors, valid, prefix in (
                ("true_track", true_donors, true_valid, "true"),
                ("wrong_track", wrong_donors, wrong_valid, "wrong"),
            ):
                features, available, gate, receipts = build_transport_row(
                    masked[center, mask_id],
                    donors[center],
                    target,
                    visible,
                    valid[center],
                )
                arrays[f"{prefix}_transport_tokens.npy"][center, mask_id, y, x] = (
                    features.astype(np.float16)
                )
                arrays[f"{prefix}_transport_available.npy"][center, mask_id, y, x] = (
                    available
                )
                arrays[f"{prefix}_gate_features.npy"][center, mask_id, y, x] = gate
                diagnostics.append(
                    {
                        "center_index": center,
                        "sample_id": selection[center]["sample_id"],
                        "mask_id": ("upper", "lower")[mask_id],
                        "condition": condition,
                        "target_count": int(target.sum()),
                        "available_targets": int(available.sum()),
                        "donors": receipts,
                    }
                )
            features, available, gate = build_repeated_row(
                masked[center, mask_id], target, visible
            )
            arrays["repeated_transport_tokens.npy"][center, mask_id, y, x] = (
                features.astype(np.float16)
            )
            arrays["repeated_transport_available.npy"][center, mask_id, y, x] = (
                available
            )
            arrays["repeated_gate_features.npy"][center, mask_id, y, x] = gate
        if (center + 1) % 16 == 0:
            for values in arrays.values():
                values.flush()
            print(json.dumps({"completed_centers": center + 1}), flush=True)
    elapsed = time.perf_counter() - started
    for values in arrays.values():
        values.flush()
    write_json_exclusive(output / "transport_receipts.json", {"rows": diagnostics})
    if any(
        sha256_file_stable(path)
        != source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")]
        for path in source_files
    ):
        raise RuntimeError("Transport source changed during execution")
    target_count = int(target_masks.sum())
    artifacts = {
        name: {
            "shape": list(shape),
            "dtype": str(arrays[name].dtype),
            "size_bytes": (output / name).stat().st_size,
            "sha256": sha256_file_stable(output / name),
        }
        for name, (_, shape) in ARRAY_SPECS.items()
    }
    summary = {
        "status": "CENTER_EVIDENCE_COMPLETION_TRANSPORT_CACHE_COMPLETE",
        "scientific_gate_evaluated": False,
        "task_training_authorized": False,
        "centers": ROWS,
        "mask_rows": ROWS * MASKS,
        "target_tokens": target_count,
        "availability": {
            "true_track_targets": int(arrays["true_transport_available.npy"].sum()),
            "wrong_track_targets": int(arrays["wrong_transport_available.npy"].sum()),
            "repeated_center_targets": int(
                arrays["repeated_transport_available.npy"].sum()
            ),
        },
        "valid_affine_fits": {
            condition: sum(
                donor["valid"]
                for row in diagnostics
                if row["condition"] == condition
                for donor in row["donors"]
            )
            for condition in ("true_track", "wrong_track")
        },
        "runtime_seconds": elapsed,
        "artifacts": artifacts,
        "cache_artifact_count": len(cache_summary["artifacts"]),
        "zero_access_counters": {
            "task_labels": 0,
            "arftr_probability_arrays": 0,
            "pose_outputs": 0,
            "support_categories": 0,
            "optimizer_updates": 0,
            "task_fits": 0,
            "router_fits": 0,
        },
        "request_sha256": sha256_file_stable(output / "request.json"),
        "next_authorization": "independent replay, then bounded fit-runtime smoke only",
    }
    write_json_exclusive(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
