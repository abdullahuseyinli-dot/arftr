"""Derive a provisional inverse for the v5 reviewer's CSS coordinate defect.

This script is deliberately pose-blind.  It preserves the uploaded review,
uses only browser geometry and frozen crop sizes, and emits a diagnostic CSV.
Because click-time viewport telemetry was not recorded, its output is not a
valid final human reference and cannot authorize the availability gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_okutama_body_witness_review import LANDMARKS, review_fields

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
DEFAULT_OUTPUT = ROOT / ".runs/research_20260913/body_witness_review_geometry_v1"
DEFAULT_SOURCE = Path("C:/Users/DELL/Downloads/reviewer_a.csv")
NORMALIZED_REVIEW = PILOT / "blinded_review/reviewer_a_completed_v1.csv"
V5_REVIEWER = PILOT / "blinded_review/index_v5_interactive.html"
EXPECTED_SOURCE_SHA256 = "1543283e4eb304f7195853cd519d32f71bc47c269076f1394c0d634e7f7dfa78"
EXPECTED_NORMALIZED_SHA256 = "4636a82145f74783e464738f83c215e24193dde5204b1bf7531d493d634ce43d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v5_display_x_scale(
    crop_width: int,
    crop_height: int,
    *,
    element_width: float,
    capped_element_height: float,
) -> float:
    """Return the v5 saved-x / true-x factor for a left/top contained image."""

    values = (crop_width, crop_height, element_width, capped_element_height)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Crop and browser dimensions must be positive and finite")
    element_height = min(element_width * crop_height / crop_width, capped_element_height)
    content_width = min(element_width, element_height * crop_width / crop_height)
    return content_width / element_width


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def derive(
    source: Path,
    output: Path,
    *,
    viewport_inner_width: int,
    viewport_inner_height: int,
    device_pixel_ratio: float,
    element_width: float,
    capped_element_height: float,
) -> dict[str, Any]:
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Uploaded review differs from the audited source")
    if sha256_file(NORMALIZED_REVIEW) != EXPECTED_NORMALIZED_SHA256:
        raise RuntimeError("Normalized single review differs from its audited bytes")
    output.mkdir(parents=True, exist_ok=False)
    preserved = output / "reviewer_a_uploaded_original.csv"
    with preserved.open("xb") as stream:
        stream.write(source.read_bytes())

    with NORMALIZED_REVIEW.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != 128 or [row["sample_id"] for row in rows] != [
        row["sample_id"] for row in manifest
    ]:
        raise RuntimeError("Review and frozen manifest differ")

    derived_rows: list[dict[str, str]] = []
    factors: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    affected_rows = 0
    affected_visible_points = 0
    scales: list[float] = []
    for row, sample in zip(rows, manifest, strict=True):
        left, top, right, bottom = sample["crops"]["1.0"]["geometry"]["crop_box"]
        width, height = right - left, bottom - top
        scale = v5_display_x_scale(
            width,
            height,
            element_width=element_width,
            capped_element_height=capped_element_height,
        )
        scales.append(scale)
        affected = scale < 1 - 1e-12
        affected_rows += affected
        output_row = dict(row)
        visible_in_row = 0
        for landmark in LANDMARKS:
            if row[f"{landmark}_visible"] != "1":
                continue
            visible_in_row += 1
            x_field = f"{landmark}_x_crop"
            corrected_x = float(row[x_field]) / scale
            output_row[x_field] = f"{corrected_x:.6f}"
            if affected:
                affected_visible_points += 1
            if not 0 <= corrected_x < width:
                outside.append(
                    {
                        "sample_id": row["sample_id"],
                        "landmark": landmark,
                        "derived_x": corrected_x,
                        "crop_width": width,
                        "outside_by_pixels": max(-corrected_x, corrected_x - width),
                    }
                )
        derived_rows.append(output_row)
        factors.append(
            {
                "sample_id": row["sample_id"],
                "crop_width": width,
                "crop_height": height,
                "saved_x_over_true_x": f"{scale:.12f}",
                "height_cap_active": int(affected),
                "visible_points": visible_in_row,
            }
        )

    derived_csv = output / "reviewer_a_css_inverse_diagnostic_v1.csv"
    factor_csv = output / "v5_css_geometry_factors_v1.csv"
    _write_csv(derived_csv, review_fields(), derived_rows)
    _write_csv(
        factor_csv,
        [
            "sample_id",
            "crop_width",
            "crop_height",
            "saved_x_over_true_x",
            "height_cap_active",
            "visible_points",
        ],
        factors,
    )
    sorted_scales = sorted(scales)
    receipt = {
        "status": "PROVISIONAL_CSS_INVERSE_DERIVED_NOT_A_VALID_FINAL_REFERENCE",
        "defect": (
            "v5 canvas covered the full 640px element while max-height/object-fit placed "
            "narrower left-aligned content inside it; exported x was compressed"
        ),
        "browser_geometry_observed_after_review": {
            "viewport_inner_width": viewport_inner_width,
            "viewport_inner_height": viewport_inner_height,
            "device_pixel_ratio": device_pixel_ratio,
            "image_element_width": element_width,
            "capped_element_height": capped_element_height,
        },
        "unverified_assumption": (
            "The browser viewport, zoom, and v5 layout remained unchanged throughout the "
            "original clicks; click-time telemetry was not saved."
        ),
        "formula": (
            "true_x = saved_x / min(1, capped_element_height*crop_width/"
            "(crop_height*image_element_width)); y unchanged"
        ),
        "rows": len(rows),
        "height_capped_rows": affected_rows,
        "affected_visible_points": affected_visible_points,
        "x_scale_min": min(scales),
        "x_scale_median": (sorted_scales[63] + sorted_scales[64]) / 2,
        "x_scale_max": max(scales),
        "derived_out_of_bounds_points": outside,
        "derived_out_of_bounds_count": len(outside),
        "formal_reference_valid": False,
        "final_availability_gate_authorized": False,
        "task_model_fitting_authorized": False,
        "second_independent_review_required": True,
        "source_original_path_at_audit": str(source),
        "source_original_sha256": sha256_file(source),
        "preserved_original_path": str(preserved.relative_to(ROOT)).replace("\\", "/"),
        "preserved_original_sha256": sha256_file(preserved),
        "normalized_review_sha256": sha256_file(NORMALIZED_REVIEW),
        "v5_reviewer_sha256": sha256_file(V5_REVIEWER),
        "script_sha256": sha256_file(Path(__file__)),
        "derived_csv_sha256": sha256_file(derived_csv),
        "factor_csv_sha256": sha256_file(factor_csv),
        "pose_outputs_read": 0,
        "action_labels_read": 0,
        "arftr_outputs_read": 0,
    }
    _write_json(output / "css_inverse_receipt_v1.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--viewport-inner-width", type=int, default=1536)
    parser.add_argument("--viewport-inner-height", type=int, default=769)
    parser.add_argument("--device-pixel-ratio", type=float, default=2.5)
    parser.add_argument("--element-width", type=float, default=640.0)
    parser.add_argument("--capped-element-height", type=float, default=538.4375)
    args = parser.parse_args()
    result = derive(
        args.source.resolve(),
        args.output.resolve(),
        viewport_inner_width=args.viewport_inner_width,
        viewport_inner_height=args.viewport_inner_height,
        device_pixel_ratio=args.device_pixel_ratio,
        element_width=args.element_width,
        capped_element_height=args.capped_element_height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
