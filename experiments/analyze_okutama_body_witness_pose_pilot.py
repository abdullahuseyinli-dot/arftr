"""Label-free error forensics for provisional body-witness pose measurements."""

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
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_pose_pilot_v1"
REVIEW = PILOT / "blinded_review/reviewer_a_completed_v1.csv"
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
LEFT_RIGHT_SWAP = np.asarray([1, 0, 3, 2, 5, 4, 7, 6])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _whole_crop_counts(
    normalized_errors: np.ndarray, visible: np.ndarray, anatomical: np.ndarray
) -> dict[str, Any]:
    visible_count = visible.sum(1)
    accurate = visible & (normalized_errors <= 0.10)
    fractions = accurate.sum(1) / np.maximum(visible_count, 1)
    whole = anatomical & (fractions >= 0.8)
    return {
        "visible_joints_accurate_at_0p10": int(accurate.sum()),
        "visible_joints_total": int(visible.sum()),
        "visible_joint_accuracy": float(accurate.sum() / visible.sum()),
        "mean_row_visible_accuracy": float(fractions.mean()),
        "whole_crop_passes": int(whole.sum()),
    }


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((PILOT / "crop_manifest.json").read_text(encoding="utf-8"))
    with REVIEW.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    with np.load(run / "pose_measurements.npz", allow_pickle=False) as saved:
        sample_ids = saved["sample_ids"]
        peak_raw = saved["pose_peak_raw_xy"]
        peak_magnitude = saved["pose_peak_magnitudes"]
    if sample_ids.tolist() != [row["sample_id"] for row in manifest] or sample_ids.tolist() != [
        row["sample_id"] for row in reviews
    ]:
        raise RuntimeError("Pose/review/manifest order differs")

    visible = np.asarray(
        [[row[f"{name}_visible"] == "1" for name in LANDMARKS] for row in reviews],
        dtype=np.bool_,
    )
    human_primary = np.full((128, 8, 2), np.nan, np.float64)
    for row_index, row in enumerate(reviews):
        for landmark_index, name in enumerate(LANDMARKS):
            if visible[row_index, landmark_index]:
                human_primary[row_index, landmark_index] = [
                    float(row[f"{name}_x_crop"]),
                    float(row[f"{name}_y_crop"]),
                ]

    crop_origins = np.asarray(
        [
            [manifest[row]["crops"][extent]["geometry"]["crop_box"][:2] for extent in ("1.0", "1.25")]
            for row in range(128)
        ],
        dtype=np.float64,
    )
    human_global = human_primary + crop_origins[:, 0, None]
    predicted_global = peak_raw + crop_origins[:, :, None]
    actor_heights = np.asarray(
        [
            manifest[row]["crops"]["1.0"]["geometry"]["native_box"][3]
            - manifest[row]["crops"]["1.0"]["geometry"]["native_box"][1]
            for row in range(128)
        ],
        dtype=np.float64,
    )
    category_present = np.stack(
        [visible[:, start : start + 2].any(1) for start in range(0, 8, 2)], axis=1
    ).all(1)
    anatomical = (visible.sum(1) >= 6) & category_present

    variants: dict[str, np.ndarray] = {}
    for extent, name in ((0, "extent1p0"), (1, "extent1p25")):
        identity = np.linalg.norm(predicted_global[:, extent] - human_global, axis=2)
        swapped = np.linalg.norm(
            predicted_global[:, extent, LEFT_RIGHT_SWAP] - human_global, axis=2
        )
        variants[f"{name}_identity"] = identity / actor_heights[:, None]
        variants[f"{name}_left_right_swapped"] = swapped / actor_heights[:, None]
        pair_oracle = np.empty_like(identity)
        for start in range(0, 8, 2):
            predictions = predicted_global[:, extent, start : start + 2]
            for target in range(start, start + 2):
                alternatives = np.linalg.norm(
                    predictions - human_global[:, target, None], axis=2
                )
                pair_oracle[:, target] = alternatives.min(1)
        variants[f"{name}_pair_identity_oracle"] = pair_oracle / actor_heights[:, None]

    metrics: dict[str, Any] = {}
    for name, errors in variants.items():
        values = errors[visible]
        metrics[name] = {
            **_whole_crop_counts(errors, visible, anatomical),
            "median_normalized_error": float(np.median(values)),
            "p90_normalized_error": float(np.quantile(values, 0.9)),
            "accuracy_by_threshold": {
                str(threshold): float(np.mean(values <= threshold))
                for threshold in (0.10, 0.20, 0.30, 0.50, 1.00)
            },
        }

    primary_errors = variants["extent1p0_identity"]
    per_landmark = {}
    for index, name in enumerate(LANDMARKS):
        values = primary_errors[:, index][visible[:, index]]
        correct = values <= 0.10
        confidence = peak_magnitude[:, 0, index][visible[:, index]]
        per_landmark[name] = {
            "visible": len(values),
            "accuracy_at_0p10": float(correct.mean()),
            "median_normalized_error": float(np.median(values)),
            "median_peak_magnitude": float(np.median(confidence)),
            "median_peak_correct": float(np.median(confidence[correct])) if correct.any() else None,
            "median_peak_incorrect": float(np.median(confidence[~correct])) if (~correct).any() else None,
        }

    result = {
        "status": "PROVISIONAL_SINGLE_REVIEW_POSE_FORENSICS_COMPLETE",
        "scope": "label-free diagnostic only; variants are not replacement gate definitions",
        "review_sha256": sha256_file(REVIEW),
        "pose_measurements_sha256": sha256_file(run / "pose_measurements.npz"),
        "anatomical_visibility_passes": int(anatomical.sum()),
        "variants": metrics,
        "per_landmark_primary": per_landmark,
        "interpretation_guard": (
            "Left/right swaps, wider-crop predictions, and pair-oracle matching test rival "
            "explanations only; the locked primary gate remains extent1.0 identity."
        ),
        "action_labels_read": 0,
        "arftr_outputs_read": 0,
        "task_optimizer_updates": 0,
    }
    write_new_json(run / "pose_error_forensics.json", result)
    _contact_sheet(
        run / "pose_overlay_contact_sheet.png",
        manifest,
        visible,
        human_primary,
        peak_raw[:, 0],
        primary_errors,
    )
    return result


def _contact_sheet(
    output: Path,
    manifest: list[dict[str, Any]],
    visible: np.ndarray,
    human: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
) -> None:
    visible_count = visible.sum(axis=1)
    row_error = np.divide(
        np.where(visible, errors, 0).sum(axis=1),
        visible_count,
        out=np.full(len(visible), np.inf, dtype=np.float64),
        where=visible_count > 0,
    )
    finite_rows = np.flatnonzero(np.isfinite(row_error))
    order = finite_rows[np.argsort(row_error[finite_rows])]
    selected = order[np.linspace(0, len(order) - 1, 16).round().astype(int)]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for axis, row in zip(axes.ravel(), selected, strict=True):
        image = np.asarray(Image.open(PILOT / manifest[row]["crops"]["1.0"]["file"]))
        axis.imshow(image)
        axis.scatter(human[row, visible[row], 0], human[row, visible[row], 1], c="#00e676", s=28, marker="o", label="human")
        axis.scatter(predicted[row, :, 0], predicted[row, :, 1], c="#ff1744", s=35, marker="x", label="pose")
        axis.set_title(f"{row:03d} mean error={row_error[row]:.2f}H", fontsize=9)
        axis.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Reviewer A (green) vs locked ViTPose peaks (red); identity assignment")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
