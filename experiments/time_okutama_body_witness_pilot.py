"""Time the frozen 16-center label-blind decode/crop preprocessing subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.pilot_okutama_body_witness import exact_fov_pose_tensor  # noqa: E402
from hac.body_witness_data import (  # noqa: E402
    CROP_EXTENTS,
    crop_raw_body,
    decode_exact_center,
    load_body_witness_cohort,
    load_verified_source_map,
    make_body_crop_geometry,
    predictor_payload,
    resolve_center_source,
    verify_video_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def time_subset(run: Path) -> dict[str, Any]:
    rows = json.loads((run / "pilot_selection.json").read_text(encoding="utf-8"))
    chosen = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(("hac-body-witness-v1|" + row["sample_id"]).encode()).hexdigest(),
            row["sample_id"],
        ),
    )[:16]
    cohort = load_body_witness_cohort(ROOT)
    by_id = {row.sample_id: row for row in cohort.observations}
    source_map = load_verified_source_map(ROOT)
    verification_started = time.perf_counter()
    verified = {}
    for recording in sorted({row["recording"] for row in chosen}):
        representative = next(row for row in chosen if row["recording"] == recording)
        request = resolve_center_source(by_id[representative["sample_id"]], source_map)
        verified[recording] = verify_video_file(request)
    verification_seconds = time.perf_counter() - verification_started
    started = time.perf_counter()
    records = []
    for row in chosen:
        observation = by_id[row["sample_id"]]
        request = resolve_center_source(observation, source_map)
        rgb, audit = decode_exact_center(request, verified_video=verified[observation.recording])
        if rgb is None or request.native_box is None:
            raise RuntimeError("Frozen timing sample exact decode unexpectedly failed")
        payload_hashes = []
        for extent in CROP_EXTENTS:
            geometry = make_body_crop_geometry(
                request.native_box, extent=extent, source_size=request.video.source_size
            )
            raw = crop_raw_body(rgb, geometry)
            exact_fov_pose_tensor(raw)
            for mask_id in ("upper_visible", "lower_visible"):
                payload = predictor_payload(raw, geometry, mask_id, decode_succeeded=True)
                pose_tensor, _ = exact_fov_pose_tensor(payload["masked_rgb"])
                payload_hashes.append(hashlib.sha256(pose_tensor.tobytes()).hexdigest())
        records.append(
            {
                "sample_id": observation.sample_id,
                "decode_pts": audit["actual_pts"],
                "payload_hashes": payload_hashes,
            }
        )
    processing_seconds = time.perf_counter() - started
    return {
        "status": "LABEL_BLIND_FIRST_16_DECODE_CROP_TIMING_COMPLETE",
        "sample_ids": [row["sample_id"] for row in chosen],
        "centers": len(chosen),
        "video_hash_verification_seconds": verification_seconds,
        "exact_decode_crop_mask_preprocess_seconds": processing_seconds,
        "seconds_per_center_excluding_video_hashes": processing_seconds / len(chosen),
        "components_not_measured": [
            "pretrained_pose_inference",
            "DINO_masked_crop_inference",
            "human_review",
        ],
        "task_label_rows_accessed_by_cohort_integrity_loader": 4977,
        "task_labels_used_by_timing": 0,
        "arftr_probability_rows_accessed_by_cohort_integrity_loader": 4977,
        "arftr_outputs_used_by_timing": 0,
        "neighbor_fallbacks": 0,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_PILOT)
    args = parser.parse_args()
    result = time_subset(args.run.resolve())
    write_new_json(args.run.resolve() / "timing_receipt.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
