"""Validate and normalize one label-blind body-witness human review.

This audit reads only the frozen crop manifest, selection IDs, and reviewer
annotations.  It never reads action labels, ARFTR outputs, or pose estimates.
Single-review output is provisional and cannot authorize task-model fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_fields() -> list[str]:
    result = ["sample_id", "reviewer_id", "review_complete", "notes"]
    for landmark in LANDMARKS:
        result.extend(
            (f"{landmark}_visible", f"{landmark}_x_crop", f"{landmark}_y_crop")
        )
    return result


def _write_new_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields())
        writer.writeheader()
        writer.writerows(rows)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def audit_review(
    source: Path, run: Path, reviewer: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        supplied_fields = reader.fieldnames
    expected_fields = review_fields()
    if supplied_fields != expected_fields:
        raise RuntimeError("Reviewer CSV schema or column order differs")

    selection = json.loads((run / "pilot_selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "crop_manifest.json").read_text(encoding="utf-8"))
    expected_ids = [row["sample_id"] for row in selection]
    if (
        len(rows) != 128
        or [row["sample_id"] for row in rows] != expected_ids
        or len(set(expected_ids)) != 128
        or any(row["reviewer_id"] != reviewer for row in rows)
    ):
        raise RuntimeError("Reviewer identity or exact 128-row population differs")

    manifest_by_id = {row["sample_id"]: row for row in manifest}
    normalized: list[dict[str, str]] = []
    initially_complete = 0
    visible_counts: Counter[str] = Counter()
    rows_with_six_visible = 0
    for row in rows:
        initially_complete += row["review_complete"].strip().lower() in {"1", "true"}
        sample_id = row["sample_id"]
        crop_record = manifest_by_id[sample_id]["crops"]["1.0"]
        image_path = run / crop_record["file"]
        with Image.open(image_path) as image:
            width, height = image.size
        visible_total = 0
        for landmark in LANDMARKS:
            visible_field = f"{landmark}_visible"
            x_field, y_field = f"{landmark}_x_crop", f"{landmark}_y_crop"
            value = row[visible_field].strip()
            if value not in {"0", "1"}:
                raise RuntimeError(f"Unset visibility for {sample_id}/{landmark}")
            if value == "1":
                try:
                    x, y = float(row[x_field]), float(row[y_field])
                except ValueError as error:
                    raise RuntimeError(
                        f"Invalid visible coordinate for {sample_id}/{landmark}"
                    ) from error
                if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                    raise RuntimeError(
                        f"Out-of-crop coordinate for {sample_id}/{landmark}: {(x, y)} vs {(width, height)}"
                    )
                visible_total += 1
                visible_counts[landmark] += 1
            elif row[x_field].strip() or row[y_field].strip():
                raise RuntimeError(f"Hidden landmark has coordinates for {sample_id}/{landmark}")
        rows_with_six_visible += visible_total >= 6
        normalized.append({**row, "review_complete": "1"})

    result = {
        "status": "SINGLE_REVIEW_PROVISIONAL_COMPLETE",
        "scope": "human anatomical feasibility only; cannot authorize task fitting or final gate",
        "reviewer": reviewer,
        "rows": len(rows),
        "source_sha256": sha256_file(source),
        "source_rows_marked_complete": initially_complete,
        "completion_flags_derived_from_eight_valid_decisions": len(rows) - initially_complete,
        "normalized_rows_complete": len(normalized),
        "visible_landmark_counts": dict(visible_counts),
        "rows_with_at_least_six_visible": rows_with_six_visible,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(expected_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "action_labels_read": 0,
        "arftr_outputs_read": 0,
        "pose_outputs_read": 0,
        "second_independent_review_required_for_final_gate": True,
    }
    return normalized, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reviewer", choices=("reviewer_a", "reviewer_b"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    normalized, result = audit_review(
        args.input.resolve(), args.run.resolve(), args.reviewer
    )
    _write_new_csv(args.output_csv.resolve(), normalized)
    result["normalized_sha256"] = sha256_file(args.output_csv.resolve())
    _write_new_json(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
