"""Revalidate reusable native pilot evidence after prospective protocol amendments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, log_loss

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.body_witness_data import (  # noqa: E402
    ARFTR_PATH,
    ARFTR_SHA256,
    CROP_EXTENTS,
    canonical_digest,
    load_body_witness_cohort,
    load_verified_source_map,
    make_body_crop_geometry,
    resolve_center_source,
    select_measurement_pilot,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
PROTOCOL = ROOT / "experiments/okutama_body_witness_protocol.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _blank_review(path: Path, expected_ids: list[str]) -> bool:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (
        len(rows) == 128
        and [row["sample_id"] for row in rows] == expected_ids
        and all(not str(value).strip() for row in rows for key, value in row.items() if key not in {"reviewer_id", "sample_id"})
    )


def _broken_links(index: Path) -> list[str]:
    links = re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", index.read_text(encoding="utf-8"))
    return [link for link in links if not (index.parent / link).is_file()]


def validate(pilot: Path, screen: Path | None = None) -> dict[str, Any]:
    protocol_hash = sha256_file(PROTOCOL)
    old_plan = _json(pilot / "execution_plan.json")
    selection_rows = _json(pilot / "pilot_selection.json")
    selection_ids = [row["sample_id"] for row in selection_rows]
    cohort = load_body_witness_cohort(ROOT)
    expected_ids = [row.sample_id for row in select_measurement_pilot(cohort.observations)]
    if selection_ids != expected_ids or canonical_digest(selection_ids) != "34f99f6b73da733d8f170518e90a7e12c5fc434cd0051375b163b0c14a42fba4":
        raise RuntimeError("Pilot selection no longer reproduces")

    by_id = {row.sample_id: row for row in cohort.observations}
    source_map = load_verified_source_map(ROOT)
    manifest = _json(pilot / "crop_manifest.json")
    extraction = _json(pilot / "extraction_receipt.json")
    if len(manifest) != 128 or extraction["crop_files"] != 256:
        raise RuntimeError("Native crop manifest size differs")
    checked_crops = 0
    for row in manifest:
        request = resolve_center_source(by_id[row["sample_id"]], source_map)
        saved = row["source_request"]
        if (
            not row["decode_valid"]
            or request.target_pts != saved["target_pts"]
            or request.native_index != saved["native_index"]
            or list(request.native_box or ()) != saved["native_box"]
        ):
            raise RuntimeError("Native source request no longer reproduces")
        for extent in CROP_EXTENTS:
            record = row["crops"][str(extent)]
            path = pilot / record["file"]
            geometry = make_body_crop_geometry(
                request.native_box, extent=extent, source_size=request.video.source_size
            )
            if (
                canonical_digest(geometry.provenance()) != canonical_digest(record["geometry"])
                or sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError("Crop file or geometry no longer reproduces")
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != record["raw_rgb_sha256"]:
                raise RuntimeError("Decoded PNG pixels differ from the raw RGB receipt")
            checked_crops += 1

    anchor_path = ROOT / ARFTR_PATH
    if sha256_file(anchor_path) != ARFTR_SHA256:
        raise RuntimeError("Retained ARFTR bytes changed")
    with np.load(anchor_path, allow_pickle=False) as saved:
        arm = int(np.flatnonzero(saved["arms"] == "r5_arftr_full")[0])
        labels = saved["labels"]
        probabilities = saved["mean_probabilities"][arm]
    predictions = probabilities.argmax(1)
    anchor = {
        "sha256": ARFTR_SHA256,
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "errors": int(np.count_nonzero(labels != predictions)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "brier_sum_classes_mean_rows": float(
            np.square(probabilities - np.eye(3)[labels]).sum(1).mean()
        ),
    }
    expected_anchor = _json(PROTOCOL)["anchor_metrics"]
    if any(not np.isclose(anchor[key], expected_anchor[key], atol=1e-12, rtol=0) for key in expected_anchor):
        raise RuntimeError("Retained ARFTR metrics differ from the protocol")

    corrected_screen = None
    if screen is not None:
        screen_plan = _json(screen / "screen_execution_plan.json")
        screen_preflight = _json(screen / "synthetic_preflight_receipt.json")
        if screen_plan["protocol"]["sha256"] != protocol_hash or screen_preflight["task_optimizer_updates"] != 0:
            raise RuntimeError("Corrected screen receipts are stale or contain task updates")
        active = screen_plan["architecture"]["active_trainable_parameters"]
        matched = np.asarray([active[arm] for arm in ("V1", "V2", "V3")])
        if matched.max() / matched.min() - 1 > 0.10:
            raise RuntimeError("Corrected screen still violates capacity matching")
        corrected_screen = {
            "run": str(screen.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": protocol_hash,
            "active_trainable_parameters": active,
            "matched_relative_range": float(matched.max() / matched.min() - 1),
            "synthetic_updates": int(sum(row["updates"] for row in screen_preflight["arms"])),
            "task_optimizer_updates": 0,
        }

    approved_pose_source = (
        ROOT
        / ".runs/research_20260913/vitpose_parity_v1/downloads/original/vitpose-b.pth"
    )
    approved_pose_hash = _json(PROTOCOL)["observations"]["source_model_lock"][
        "source_mirror_checkpoint_sha256"
    ]
    checkpoint_files = [
        path
        for path in (ROOT / ".runs/research_20260913").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    ]
    approved_pose_present = (
        approved_pose_source.is_file()
        and sha256_file(approved_pose_source) == approved_pose_hash
    )
    forbidden_outputs = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in checkpoint_files
        if path != approved_pose_source
    ]
    pose_outputs = list(pilot.glob("pose_outputs.*"))
    if forbidden_outputs or pose_outputs or not approved_pose_present:
        raise RuntimeError("Unexpected task checkpoint or pretrained pose output exists")

    original_broken = _broken_links(pilot / "blinded_review/index.html")
    corrected_broken = _broken_links(pilot / "blinded_review/index_v2.html")
    interactive_path = pilot / "blinded_review/index_v5_interactive.html"
    interactive = interactive_path.read_text(encoding="utf-8")
    interactive_missing = [
        record["file"].removeprefix("blinded_review/")
        for row in manifest
        for record in row["crops"].values()
        if record["file"].removeprefix("blinded_review/") not in interactive
        or not (interactive_path.parent / record["file"].removeprefix("blinded_review/")).is_file()
    ]
    if len(original_broken) != 128 or corrected_broken or interactive_missing:
        raise RuntimeError("Review-index supersession differs from the documented state")

    reviews_blank = all(
        _blank_review(pilot / f"blinded_review/{reviewer}.csv", selection_ids)
        for reviewer in ("reviewer_a", "reviewer_b")
    )
    if not reviews_blank:
        raise RuntimeError("Reviewer sheets are no longer the uncompleted blinded templates")

    return {
        "status": "PASS_NATIVE_ARTIFACTS_REUSABLE_POSE_AND_TASK_GATES_BLOCKED",
        "scope": "Revalidates label-blind selection/native crops and corrected synthetic architecture only; not pose quality, task performance, or deployable integration.",
        "selection": {"rows": 128, "sample_ids_sha256": canonical_digest(selection_ids)},
        "native_extraction": {
            "exact_pts_rows": 128,
            "verified_crop_files": checked_crops,
            "neighbor_fallbacks": extraction["neighbor_fallbacks"],
            "crop_manifest_sha256": sha256_file(pilot / "crop_manifest.json"),
        },
        "retained_arftr": anchor,
        "human_review": {"sheets_blank": True, "rows_per_sheet": 128},
        "review_index": {
            "original_superseded_broken_links": len(original_broken),
            "index_v2_broken_links": len(corrected_broken),
            "index_v3_interactive_status": "superseded_syntax_failure_preserved",
            "index_v4_interactive_status": "superseded_passed_but_primary_crop_not_enlarged",
            "index_v5_interactive_missing_images": len(interactive_missing),
            "index_v5_interactive_export": "reviewer_a.csv or reviewer_b.csv",
        },
        "pose_checkpoint_input": {
            "source_mirror_present_and_hash_verified": approved_pose_present,
            "pilot_pose_outputs": len(pose_outputs),
        },
        "historical_pilot_plan": {
            "protocol_sha256": old_plan["protocol"]["sha256"],
            "current_protocol_sha256": protocol_hash,
            "is_superseded": old_plan["protocol"]["sha256"] != protocol_hash,
        },
        "corrected_screen": corrected_screen,
        "blocked": [
            "two independent blinded human reviews",
            "official-publisher checkpoint-byte identity and pilot preprocessing/flip lock",
            "pose availability and target-sensitivity gates",
            "task fitting authority",
            "nested integration under the current structurally infeasible recipe",
        ],
        "audit_code_sha256": sha256_file(Path(__file__)),
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--screen", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.pilot.resolve(), args.screen.resolve() if args.screen is not None else None
    )
    write_new_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
