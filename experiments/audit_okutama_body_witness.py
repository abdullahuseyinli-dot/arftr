"""Independent structural audits for the body-witness pilot and candidate.

No task model is fitted here.  The audit deliberately separates predictor-input
invariance (a dependency claim) from target sensitivity (a measurement claim).
The latter uses total variation over categorical spatial mass; mean per-bin L1
would make the locked 0.05 threshold mathematically impossible with 193 bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.body_witness import make_pair_only_candidate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical_total_variation(
    original: np.ndarray,
    corrupted: np.ndarray,
) -> np.ndarray:
    """Mean-over-landmarks TV, retaining every crop/view/mask denominator.

    Arrays are ``[..., landmarks, bins]`` and each landmark distribution must
    sum to one.  Returned values are in ``[0,1]`` with shape ``[...]``.
    """

    original = np.asarray(original, dtype=np.float64)
    corrupted = np.asarray(corrupted, dtype=np.float64)
    if (
        original.shape != corrupted.shape
        or original.ndim < 2
        or original.shape[-1] < 2
        or not np.isfinite(original).all()
        or not np.isfinite(corrupted).all()
        or (original < 0).any()
        or (corrupted < 0).any()
        or not np.allclose(original.sum(-1), 1.0, atol=5e-6, rtol=0)
        or not np.allclose(corrupted.sum(-1), 1.0, atol=5e-6, rtol=0)
    ):
        raise ValueError("TV inputs must be matching normalized categorical witnesses")
    return 0.5 * np.abs(original - corrupted).sum(-1).mean(-1)


def audit_dependency_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Verify hidden-pixel corruption cannot alter any predictor input."""

    pairs = (
        ("visible_pixels_original", "visible_pixels_corrupted"),
        ("visible_features_original", "visible_features_corrupted"),
        ("crop_transforms_original", "crop_transforms_corrupted"),
        ("predictor_validity_original", "predictor_validity_corrupted"),
        ("mask_ids_original", "mask_ids_corrupted"),
        ("predictor_logits_original", "predictor_logits_corrupted"),
    )
    missing = [name for pair in pairs for name in pair if name not in arrays]
    if missing:
        raise RuntimeError(f"Dependency audit is missing arrays: {missing}")
    checks = {}
    for left_name, right_name in pairs:
        left, right = arrays[left_name], arrays[right_name]
        checks[left_name.removesuffix("_original")] = {
            "shape_equal": left.shape == right.shape,
            "dtype_equal": left.dtype == right.dtype,
            "bit_exact": bool(np.array_equal(left, right)),
            "max_absolute_difference": (
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                if left.shape == right.shape and left.size
                else None
            ),
        }
    passed = all(item["shape_equal"] and item["dtype_equal"] and item["bit_exact"] for item in checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "interpretation": "predictor dependency exclusion only; not evidence that target is sensitive",
        "checks": checks,
    }


def audit_target_sensitivity(
    original_witness: np.ndarray,
    corrupted_witness: np.ndarray,
    *,
    review_pass: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Apply the locked pilot sensitivity gate to four witnesses per crop."""

    review_pass = np.asarray(review_pass)
    if review_pass.dtype != np.bool_ or review_pass.ndim != 1:
        raise ValueError("review_pass must be a boolean crop-level vector")
    tv = categorical_total_variation(original_witness, corrupted_witness)
    if tv.ndim != 2 or tv.shape[0] != len(review_pass):
        raise ValueError("Expected [crops, spatial_witnesses, landmarks, bins]")
    eligible = np.broadcast_to(review_pass[:, None], tv.shape)
    numerator = int(np.count_nonzero(eligible & (tv > threshold)))
    denominator = int(np.count_nonzero(eligible))
    fraction = numerator / denominator if denominator else 0.0
    return {
        "status": "PASS" if denominator and fraction >= 0.5 else "FAIL",
        "metric": "mean_landmark_categorical_total_variation",
        "threshold_strictly_greater_than": threshold,
        "required_fraction": 0.5,
        "sensitive_witnesses": numerator,
        "eligible_witnesses": denominator,
        "fraction": fraction,
        "all_spatial_witnesses_retained_in_denominator": True,
        "values": tv.tolist(),
    }


def audit_pilot_target_sensitivity(
    arrays: dict[str, np.ndarray], selection_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail-closed production wrapper around the generic TV calculation."""

    required = {"sample_ids", "original_witness", "corrupted_witness", "review_pass"}
    if not required.issubset(arrays):
        raise RuntimeError(f"Sensitivity archive lacks {sorted(required - set(arrays))}")
    expected_ids = np.asarray([row["sample_id"] for row in selection_rows])
    supplied_ids = np.asarray(arrays["sample_ids"])
    if (
        len(selection_rows) != 128
        or supplied_ids.dtype.kind != "U"
        or supplied_ids.shape != (128,)
        or not np.array_equal(supplied_ids, expected_ids)
        or arrays["original_witness"].shape != (128, 4, 8, 193)
        or arrays["corrupted_witness"].shape != (128, 4, 8, 193)
        or arrays["review_pass"].shape != (128,)
        or arrays["review_pass"].dtype != np.bool_
    ):
        raise RuntimeError("Pilot sensitivity inputs violate the exact 128x4x8x193 contract")
    result = audit_target_sensitivity(
        arrays["original_witness"],
        arrays["corrupted_witness"],
        review_pass=arrays["review_pass"],
    )
    result["production_shape_and_sample_order_verified"] = True
    result["sample_ids_sha256"] = hashlib.sha256(
        json.dumps(supplied_ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    return result


def audit_candidate(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"arftr_probabilities", "sitting_probability", "acquired_measurement_valid"}
        if not required.issubset(saved.files):
            raise RuntimeError(f"Candidate archive lacks {sorted(required - set(saved.files))}")
        anchor = saved["arftr_probabilities"]
        result = make_pair_only_candidate(
            anchor,
            saved["sitting_probability"],
            acquired_measurement_valid=saved["acquired_measurement_valid"],
        )
        if "candidate_probabilities" in saved.files and not np.array_equal(
            result.probabilities, saved["candidate_probabilities"]
        ):
            raise RuntimeError("Independent pair-only candidate replay differs")
    return {
        "status": "PASS",
        "rows": len(anchor),
        "eligible_rows": int(result.eligible.sum()),
        "exact_retain_rows": int((~result.eligible).sum()),
        "locomotion_column_bit_exact": bool(np.array_equal(result.probabilities[:, 2], anchor[:, 2])),
        "candidate_sha256": sha256_file(path),
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dependency", "sensitivity", "candidate"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if args.stage == "dependency":
        result = audit_dependency_arrays(arrays)
    elif args.stage == "sensitivity":
        selection = args.input.parent / "pilot_selection.json"
        if not selection.is_file():
            raise RuntimeError("Production sensitivity audit requires the pinned pilot selection")
        result = audit_pilot_target_sensitivity(
            arrays, json.loads(selection.read_text(encoding="utf-8"))
        )
    else:
        result = audit_candidate(args.input)
    write_new_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
