"""Compare raw and CSS-inverted reviewer coordinates against frozen pose peaks.

The inverse is a post-review diagnostic under an unverified stable-viewport
assumption.  Its metrics are never a final availability result or authority to
train a task model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / ".runs/research_20260913/body_witness_pilot_v1"
POSE_RUN = ROOT / ".runs/research_20260913/body_witness_pose_pilot_v1"
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_review_geometry_v1"
RAW_REVIEW = PILOT / "blinded_review/reviewer_a_completed_v1.csv"
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


def _read_review(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    visible = np.asarray(
        [[row[f"{name}_visible"] == "1" for name in LANDMARKS] for row in rows],
        dtype=np.bool_,
    )
    coordinates = np.full((len(rows), len(LANDMARKS), 2), np.nan, np.float64)
    for row_index, row in enumerate(rows):
        for landmark_index, name in enumerate(LANDMARKS):
            if visible[row_index, landmark_index]:
                coordinates[row_index, landmark_index] = [
                    float(row[f"{name}_x_crop"]),
                    float(row[f"{name}_y_crop"]),
                ]
    return rows, visible, coordinates


def _metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    visible: np.ndarray,
    anatomical: np.ndarray,
    actor_heights: np.ndarray,
    scenarios: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    delta = (predicted - reference) / actor_heights[:, None, None]
    errors = np.linalg.norm(delta, axis=2)
    accurate = visible & (errors <= 0.10)
    visible_count = visible.sum(1)
    required = np.ceil(0.8 * visible_count).astype(np.int64)
    accurate_count = accurate.sum(1)
    whole = anatomical & (accurate_count >= required)
    visible_delta = delta[visible]
    result = {
        "visible_joints_accurate_at_0p10": int(accurate.sum()),
        "visible_joints_total": int(visible.sum()),
        "visible_joint_accuracy": float(accurate.sum() / visible.sum()),
        "whole_crop_passes": int(whole.sum()),
        "median_normalized_error": float(np.median(errors[visible])),
        "median_signed_x_residual": float(np.median(visible_delta[:, 0])),
        "median_signed_y_residual": float(np.median(visible_delta[:, 1])),
        "scenario_whole_crop_passes": {
            str(scenario): int(whole[scenarios == scenario].sum())
            for scenario in sorted(set(scenarios.tolist()))
        },
    }
    return result, errors, whole


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _contact_sheet(
    path: Path,
    manifest: list[dict[str, Any]],
    visible: np.ndarray,
    reference: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    visible_count = visible.sum(1)
    row_error = np.divide(
        np.where(visible, errors, 0).sum(1),
        visible_count,
        out=np.full(len(visible), np.inf, dtype=np.float64),
        where=visible_count > 0,
    )
    finite_order = np.argsort(row_error[np.isfinite(row_error)])
    finite_rows = np.flatnonzero(np.isfinite(row_error))[finite_order]
    selected = finite_rows[
        np.linspace(0, len(finite_rows) - 1, 16).round().astype(np.int64)
    ]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for axis, row in zip(axes.ravel(), selected, strict=True):
        image = np.asarray(Image.open(PILOT / manifest[row]["crops"]["1.0"]["file"]))
        axis.imshow(image)
        axis.scatter(
            reference[row, visible[row], 0],
            reference[row, visible[row], 1],
            c="#00e676",
            s=28,
            marker="o",
            label="CSS-inverted human",
        )
        axis.scatter(
            predicted[row, :, 0],
            predicted[row, :, 1],
            c="#ff1744",
            s=35,
            marker="x",
            label="pose",
        )
        axis.set_title(f"{row:03d} mean error={row_error[row]:.2f}H", fontsize=9)
        axis.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Diagnostic only: CSS-inverted Reviewer A (green) vs ViTPose (red)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def analyze(run: Path) -> dict[str, Any]:
    receipt = json.loads((run / "css_inverse_receipt_v1.json").read_text(encoding="utf-8"))
    derived_path = run / "reviewer_a_css_inverse_diagnostic_v1.csv"
    pose_path = POSE_RUN / "pose_measurements.npz"
    if (
        receipt["formal_reference_valid"]
        or receipt["derived_csv_sha256"] != sha256_file(derived_path)
        or sha256_file(pose_path)
        != "949db8988c9290c8ac2d4ead5cab24c94ef7cb6145a1482075e8c98e55c7c15b"
    ):
        raise RuntimeError("CSS inverse or pose artifact ancestry differs")
    manifest = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    raw_rows, raw_visible, raw_coordinates = _read_review(RAW_REVIEW)
    derived_rows, derived_visible, derived_coordinates = _read_review(derived_path)
    expected_ids = [row["sample_id"] for row in manifest]
    if (
        [row["sample_id"] for row in raw_rows] != expected_ids
        or [row["sample_id"] for row in derived_rows] != expected_ids
        or not np.array_equal(raw_visible, derived_visible)
    ):
        raise RuntimeError("Manifest or visibility changed during CSS inversion")
    with np.load(pose_path, allow_pickle=False) as saved:
        predicted = saved["pose_peak_raw_xy"][:, 0].astype(np.float64)
    actor_heights = np.asarray(
        [
            row["crops"]["1.0"]["geometry"]["native_box"][3]
            - row["crops"]["1.0"]["geometry"]["native_box"][1]
            for row in manifest
        ],
        dtype=np.float64,
    )
    category_present = np.stack(
        [raw_visible[:, start : start + 2].any(1) for start in range(0, 8, 2)], axis=1
    ).all(1)
    anatomical = (raw_visible.sum(1) >= 6) & category_present
    scenarios = np.asarray([row["scenario"] for row in manifest])
    raw_metrics, _, raw_whole = _metrics(
        predicted,
        raw_coordinates,
        raw_visible,
        anatomical,
        actor_heights,
        scenarios,
    )
    derived_metrics, derived_errors, derived_whole = _metrics(
        predicted,
        derived_coordinates,
        derived_visible,
        anatomical,
        actor_heights,
        scenarios,
    )
    row_output = []
    for index, sample in enumerate(manifest):
        count = int(raw_visible[index].sum())
        row_output.append(
            {
                "sample_id": sample["sample_id"],
                "scenario": sample["scenario"],
                "visible_landmarks": count,
                "anatomical_pass": int(anatomical[index]),
                "required_localized": int(np.ceil(0.8 * count)),
                "css_inverse_localized": int(
                    (raw_visible[index] & (derived_errors[index] <= 0.10)).sum()
                ),
                "raw_whole_crop_pass": int(raw_whole[index]),
                "css_inverse_whole_crop_pass": int(derived_whole[index]),
            }
        )
    row_path = run / "css_inverse_pose_row_diagnostics_v1.csv"
    image_path = run / "css_inverse_pose_overlay_v1.png"
    _write_csv(row_path, row_output)
    _contact_sheet(
        image_path,
        manifest,
        derived_visible,
        derived_coordinates,
        predicted,
        derived_errors,
    )
    result = {
        "status": "PROVISIONAL_CSS_INVERSE_POSE_DIAGNOSTIC_NOT_A_GATE_RESULT",
        "interpretation": (
            "The CSS inverse removes the horizontal residual signature and falsifies the "
            "claim that 7/128 demonstrated pose failure.  It cannot replace a valid blinded "
            "review because original click-time viewport telemetry is absent."
        ),
        "raw_corrupted_reference_metrics": raw_metrics,
        "css_inverse_diagnostic_metrics": derived_metrics,
        "anatomical_visibility_passes_unchanged": int(anatomical.sum()),
        "availability_gate_required": 103,
        "anatomical_ceiling_under_reviewer_a": int(anatomical.sum()),
        "css_inverse_whole_crop_shortfall": 103 - int(derived_whole.sum()),
        "anatomical_ceiling_shortfall": 103 - int(anatomical.sum()),
        "css_inverse_out_of_bounds_points": receipt["derived_out_of_bounds_count"],
        "formal_reference_valid": False,
        "final_availability_gate_authorized": False,
        "task_model_fitting_authorized": False,
        "second_independent_review_required": True,
        "css_inverse_receipt_sha256": sha256_file(run / "css_inverse_receipt_v1.json"),
        "pose_measurements_sha256": sha256_file(pose_path),
        "row_diagnostics_sha256": sha256_file(row_path),
        "overlay_sha256": sha256_file(image_path),
        "script_sha256": sha256_file(Path(__file__)),
        "action_labels_read": 0,
        "arftr_outputs_read": 0,
        "task_optimizer_updates": 0,
    }
    _write_json(run / "css_inverse_pose_diagnostic_v1.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
