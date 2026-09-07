"""Label-blind camera-compensated actor-correspondence pilot primitives."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

SIZE_BANDS = ("small", "medium", "large")


def stable_hash(domain: str, value: str) -> str:
    return hashlib.sha256((domain + value).encode()).hexdigest()


def native_size_band(height: float) -> str:
    if not math.isfinite(height) or height <= 0:
        raise ValueError("CCAC requires a positive finite native actor height")
    if height < 32:
        return "small"
    if height <= 64:
        return "medium"
    return "large"


def select_pilot_candidates(
    candidates: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select a fixed scenario/size/validity-balanced cohort without task labels."""
    domain = str(selection["hash_domain"])
    total = int(selection["selected_clips"])
    targets = {name: int(selection["size_targets"][name]) for name in SIZE_BANDS}
    invalid_target = int(selection["long_incomplete_target"])
    if sum(targets.values()) != total or not 0 <= invalid_target <= total:
        raise ValueError("CCAC selection targets do not sum to the requested cohort")
    if not candidates or len({row["sample_id"] for row in candidates}) != len(candidates):
        raise ValueError("CCAC candidates must have unique sample identities")
    scenarios = sorted({str(row["scenario"]) for row in candidates})
    if total < len(scenarios):
        raise ValueError("CCAC cohort is too small for its scenario contract")
    base, extras = divmod(total, len(scenarios))
    extra_scenarios = set(
        sorted(
            scenarios,
            key=lambda value: (stable_hash(domain, "scenario:" + value), value),
        )[:extras]
    )
    scenario_targets = {scenario: base + int(scenario in extra_scenarios) for scenario in scenarios}
    availability: dict[tuple[str, str, bool], int] = {}
    for row in candidates:
        key = (str(row["scenario"]), str(row["size_band"]), bool(row["long_valid"]))
        availability[key] = availability.get(key, 0) + 1
    remaining = scenario_targets.copy()
    cell_targets = {(scenario, band): 0 for scenario in scenarios for band in SIZE_BANDS}

    def allocate_band(band: str, target: int) -> None:
        order = sorted(
            scenarios,
            key=lambda value: (
                stable_hash(domain, f"band:{band}:{value}"),
                value,
            ),
        )
        allocated = 0
        while allocated < target:
            progressed = False
            for scenario in order:
                if allocated >= target:
                    break
                available = availability.get((scenario, band, True), 0) + availability.get(
                    (scenario, band, False), 0
                )
                if remaining[scenario] and cell_targets[(scenario, band)] < available:
                    cell_targets[(scenario, band)] += 1
                    remaining[scenario] -= 1
                    allocated += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"CCAC cannot satisfy the {band} size target")

    allocate_band("large", targets["large"])
    allocate_band("small", targets["small"])
    for scenario in scenarios:
        medium_available = availability.get((scenario, "medium", True), 0) + availability.get(
            (scenario, "medium", False), 0
        )
        if remaining[scenario] > medium_available:
            raise RuntimeError("CCAC medium-size residual allocation is infeasible")
        cell_targets[(scenario, "medium")] = remaining[scenario]
        remaining[scenario] = 0
    if any(
        sum(cell_targets[(scenario, band)] for scenario in scenarios) != targets[band]
        for band in SIZE_BANDS
    ):
        raise RuntimeError("CCAC size allocation changed")

    cells = [key for key, count in cell_targets.items() if count]
    invalid_by_cell: dict[tuple[str, str], int] = {}
    for scenario, band in cells:
        complete = availability.get((scenario, band, True), 0)
        invalid_by_cell[(scenario, band)] = max(0, cell_targets[(scenario, band)] - complete)
    remaining_invalid = invalid_target - sum(invalid_by_cell.values())
    if remaining_invalid < 0:
        raise RuntimeError("CCAC cells force too many incomplete clips")
    cell_order = sorted(
        cells,
        key=lambda value: (
            stable_hash(domain, f"invalid:{value[0]}:{value[1]}"),
            value,
        ),
    )
    while remaining_invalid:
        progressed = False
        for cell in cell_order:
            if not remaining_invalid:
                break
            scenario, band = cell
            incomplete_available = availability.get((scenario, band, False), 0)
            if (
                invalid_by_cell[cell] < cell_targets[cell]
                and invalid_by_cell[cell] < incomplete_available
            ):
                invalid_by_cell[cell] += 1
                remaining_invalid -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("CCAC cannot satisfy its incomplete-clip target")

    selected: list[dict[str, Any]] = []
    track_use: dict[str, int] = {}
    visit_order = sorted(
        cells,
        key=lambda value: (
            stable_hash(domain, f"cell:{value[0]}:{value[1]}"),
            value,
        ),
    )
    for scenario, band in visit_order:
        cell = (scenario, band)
        for long_valid, count in (
            (False, invalid_by_cell[cell]),
            (True, cell_targets[cell] - invalid_by_cell[cell]),
        ):
            pool = [
                row
                for row in candidates
                if str(row["scenario"]) == scenario
                and str(row["size_band"]) == band
                and bool(row["long_valid"]) is long_valid
            ]
            for _ in range(count):
                if not pool:
                    raise RuntimeError("CCAC selection cell exhausted")
                pool.sort(
                    key=lambda row: (
                        track_use.get(str(row["track_key"]), 0),
                        stable_hash(domain, "sample:" + str(row["sample_id"])),
                        str(row["sample_id"]),
                    )
                )
                chosen = dict(pool.pop(0))
                chosen["selection_rank"] = len(selected)
                selected.append(chosen)
                key = str(chosen["track_key"])
                track_use[key] = track_use.get(key, 0) + 1
    if (
        len(selected) != total
        or len({row["sample_id"] for row in selected}) != total
        or sum(not bool(row["long_valid"]) for row in selected) != invalid_target
    ):
        raise RuntimeError("CCAC selected-cohort contract changed")
    for scenario, target in scenario_targets.items():
        if sum(row["scenario"] == scenario for row in selected) != target:
            raise RuntimeError("CCAC scenario allocation changed")
    for band, target in targets.items():
        if sum(row["size_band"] == band for row in selected) != target:
            raise RuntimeError("CCAC selected size balance changed")
    return selected


def configure_opencv(*, threads: int, opencl: bool) -> dict[str, Any]:
    cv2.setNumThreads(int(threads))
    cv2.ocl.setUseOpenCL(bool(opencl))
    return {
        "version": cv2.__version__,
        "threads": int(cv2.getNumThreads()),
        "opencl": bool(cv2.ocl.useOpenCL()),
    }


@dataclass(frozen=True)
class FBTracks:
    source: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    forward_backward_error: np.ndarray


@dataclass(frozen=True)
class CameraEstimate:
    transform: np.ndarray
    selected: str
    usable: bool
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PairAnalysis:
    metrics: dict[str, Any]
    background_source: np.ndarray
    background_target: np.ndarray
    background_cells: np.ndarray
    actor_source: np.ndarray
    actor_target: np.ndarray
    actor_forward_backward_error: np.ndarray
    camera_transform: np.ndarray


def _lk_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "winSize": tuple(int(value) for value in spec["window"]),
        "maxLevel": int(spec["maximum_pyramid_level"]),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(spec["iterations"]),
            float(spec["epsilon"]),
        ),
        "minEigThreshold": float(spec["minimum_eigenvalue"]),
    }


def track_points_forward_backward(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    spec: dict[str, Any],
    *,
    maximum_error: float,
) -> FBTracks:
    """Track fixed source points and reject failures using backward consistency."""
    source = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    count = len(source)
    target = np.full((count, 2), np.nan, dtype=np.float32)
    fb_error = np.full(count, np.inf, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    if count == 0:
        return FBTracks(source, target, valid, fb_error)
    if (
        previous.dtype != np.uint8
        or current.dtype != np.uint8
        or previous.ndim != 2
        or previous.shape != current.shape
    ):
        raise ValueError("CCAC LK inputs must be aligned uint8 grayscale frames")
    height, width = previous.shape
    source_ok = (
        np.isfinite(source).all(axis=1)
        & (source[:, 0] >= 0)
        & (source[:, 0] < width)
        & (source[:, 1] >= 0)
        & (source[:, 1] < height)
    )
    source_indices = np.flatnonzero(source_ok)
    if not len(source_indices):
        return FBTracks(source, target, valid, fb_error)
    query = source[source_indices].reshape(-1, 1, 2)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, query, None, **_lk_spec(spec)
    )
    if forward is None or forward_status is None:
        return FBTracks(source, target, valid, fb_error)
    forward_values = forward.reshape(-1, 2)
    forward_finite = np.isfinite(forward_values).all(axis=1)
    forward_inside = (
        (forward_values[:, 0] >= 0)
        & (forward_values[:, 0] < width)
        & (forward_values[:, 1] >= 0)
        & (forward_values[:, 1] < height)
    )
    forward_ok = forward_status.reshape(-1).astype(bool) & forward_finite & forward_inside
    forward_local_indices = np.flatnonzero(forward_ok)
    if not len(forward_local_indices):
        return FBTracks(source, target, valid, fb_error)
    reverse_indices = source_indices[forward_local_indices]
    reverse_query = forward_values[forward_local_indices].reshape(-1, 1, 2)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, reverse_query, None, **_lk_spec(spec)
    )
    if backward is None or backward_status is None:
        return FBTracks(source, target, valid, fb_error)
    target[reverse_indices] = forward_values[forward_local_indices].astype(np.float32)
    returned = backward.reshape(-1, 2)
    reverse_finite = np.isfinite(returned).all(axis=1)
    reverse_status = backward_status.reshape(-1).astype(bool)
    reverse_error = np.linalg.norm(returned - source[reverse_indices], axis=1).astype(np.float32)
    fb_error[reverse_indices] = reverse_error
    valid[reverse_indices] = (
        reverse_status
        & reverse_finite
        & np.isfinite(reverse_error)
        & (reverse_error <= np.float32(maximum_error))
    )
    return FBTracks(source, target, valid, fb_error)


def _clip_box(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, ...]:
    height, width = shape
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(math.floor(x0)))),
        max(0, min(height, int(math.floor(y0)))),
        max(0, min(width, int(math.ceil(x1)))),
        max(0, min(height, int(math.ceil(y1)))),
    )


def expand_box(
    box: tuple[float, float, float, float], amount: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return x0 - amount, y0 - amount, x1 + amount, y1 + amount


def _points_in_box(points: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)


def points_outside_boxes(
    points: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    *,
    expansion: float,
) -> np.ndarray:
    outside = np.ones(len(points), dtype=bool)
    for box in boxes:
        outside &= ~_points_in_box(points, expand_box(box, expansion))
    return outside


def actor_seed_points(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    spec: dict[str, Any],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    fraction = float(spec["seed_region_erode_fraction"])
    dx, dy = fraction * (x1 - x0), fraction * (y1 - y0)
    left, top, right, bottom = _clip_box((x0 + dx, y0 + dy, x1 - dx, y1 - dy), image.shape)
    if right - left < 3 or bottom - top < 3:
        return np.empty((0, 2), dtype=np.float32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    detector = spec["corner_detector"]
    points = cv2.goodFeaturesToTrack(
        image,
        maxCorners=int(spec["maximum_corners"]),
        qualityLevel=float(detector["quality_level"]),
        minDistance=float(detector["minimum_distance_pixels"]),
        mask=mask,
        blockSize=int(detector["block_size"]),
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def background_seed_points(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    columns, rows = (int(value) for value in spec["spatial_grid"])
    maximum = int(spec["maximum_corners_total"])
    per_cell = int(math.ceil(maximum / (columns * rows)))
    exclusion = float(spec["known_actor_expansion_pixels"])
    base_mask = np.full(image.shape, 255, dtype=np.uint8)
    for box in boxes:
        left, top, right, bottom = _clip_box(expand_box(box, exclusion), image.shape)
        base_mask[top:bottom, left:right] = 0
    detector = spec["corner_detector"]
    all_points: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    height, width = image.shape
    for row in range(rows):
        for column in range(columns):
            left, right = column * width // columns, (column + 1) * width // columns
            top, bottom = row * height // rows, (row + 1) * height // rows
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask[top:bottom, left:right] = base_mask[top:bottom, left:right]
            points = cv2.goodFeaturesToTrack(
                image,
                maxCorners=per_cell,
                qualityLevel=float(detector["quality_level"]),
                minDistance=float(detector["minimum_distance_pixels"]),
                mask=mask,
                blockSize=int(detector["block_size"]),
            )
            if points is None:
                continue
            values = points.reshape(-1, 2).astype(np.float32)
            all_points.append(values)
            cell_ids.append(np.full(len(values), row * columns + column, dtype=np.int16))
    if not all_points:
        return np.empty((0, 2), np.float32), np.empty(0, np.int16)
    points = np.concatenate(all_points)[:maximum]
    cells = np.concatenate(cell_ids)[:maximum]
    return points, cells


def project_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values))))
    projected = homogeneous @ np.asarray(transform, dtype=np.float64).T
    denominator = projected[:, 2]
    result = np.full((len(values), 2), np.nan, dtype=np.float64)
    usable = np.isfinite(projected).all(axis=1) & (np.abs(denominator) > 1e-12)
    result[usable] = projected[usable, :2] / denominator[usable, None]
    return result


def _residual_summary(
    transform: np.ndarray, source: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    error = np.linalg.norm(target - project_points(transform, source), axis=1)
    if not len(error) or not np.isfinite(error).all():
        return {"median": float("inf"), "p90": float("inf"), "mean": float("inf")}
    return {
        "median": float(np.median(error)),
        "p90": float(np.quantile(error, 0.9)),
        "mean": float(error.mean()),
    }


def _three_way_split(
    source: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spatially stratified fit, model-selection, and final-audit masks."""
    selection = np.zeros(len(source), dtype=bool)
    audit = np.zeros(len(source), dtype=bool)
    for cell in np.unique(cells):
        indices = np.flatnonzero(cells == cell)
        order = indices[np.lexsort((source[indices, 0], source[indices, 1]))]
        selection[order[0::5]] = True
        audit[order[1::5]] = True
    return ~(selection | audit), selection, audit


def _homography_has_safe_image_domain(transform: np.ndarray, image_shape: tuple[int, int]) -> bool:
    """Reject projective poles, reflections, and extreme image-domain warps."""
    height, width = image_shape
    locations = np.asarray(
        (
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
            (width / 2.0, height / 2.0),
        ),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((locations, np.ones(len(locations)))) @ transform.T
    denominators = homogeneous[:, 2]
    if (
        not np.isfinite(homogeneous).all()
        or np.any(np.abs(denominators) <= 1e-9)
        or not (np.all(denominators > 0) or np.all(denominators < 0))
    ):
        return False
    projected = homogeneous[:, :2] / denominators[:, None]
    if (
        np.any(projected[:, 0] < -width)
        or np.any(projected[:, 0] > 2 * width)
        or np.any(projected[:, 1] < -height)
        or np.any(projected[:, 1] > 2 * height)
    ):
        return False
    a, b, c = transform[0]
    d, e, f = transform[1]
    g, h, _ = transform[2]
    jacobian_determinants: list[float] = []
    for (x, y), denominator in zip(locations, denominators, strict=True):
        numerator_x = a * x + b * y + c
        numerator_y = d * x + e * y + f
        denominator_sq = denominator * denominator
        jacobian = np.asarray(
            (
                (
                    (a * denominator - g * numerator_x) / denominator_sq,
                    (b * denominator - h * numerator_x) / denominator_sq,
                ),
                (
                    (d * denominator - g * numerator_y) / denominator_sq,
                    (e * denominator - h * numerator_y) / denominator_sq,
                ),
            )
        )
        jacobian_determinants.append(float(np.linalg.det(jacobian)))
    return bool(
        np.isfinite(jacobian_determinants).all()
        and np.all(np.asarray(jacobian_determinants) > 1e-12)
    )


def estimate_camera(
    source: np.ndarray,
    target: np.ndarray,
    cells: np.ndarray,
    image_shape: tuple[int, int],
    *,
    seed: int,
    spec: dict[str, Any],
    gate: dict[str, Any],
) -> CameraEstimate:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    cells = np.asarray(cells).reshape(-1)
    if len(source) != len(target) or len(source) != len(cells):
        raise ValueError("CCAC camera correspondences are not aligned")
    fit, selection, audit = _three_way_split(source, cells)
    identity = np.eye(3, dtype=np.float64)
    models: dict[str, dict[str, Any]] = {
        "H0": {
            "valid": bool(selection.any() and audit.any()),
            "transform": identity,
            "inliers": 0,
            "selection": _residual_summary(identity, source[selection], target[selection]),
            "audit": _residual_summary(identity, source[audit], target[audit]),
        }
    }
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    similarity = None
    similarity_inliers = None
    if fit.sum() >= 3:
        similarity, similarity_inliers = cv2.estimateAffinePartial2D(
            source[fit].astype(np.float32),
            target[fit].astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(spec["ransac_reprojection_pixels"]),
            maxIters=int(spec["ransac_maximum_iterations"]),
            confidence=float(spec["ransac_confidence"]),
            refineIters=10,
        )
    similarity_transform = identity.copy()
    similarity_valid = similarity is not None and np.isfinite(similarity).all()
    if similarity_valid:
        similarity_transform[:2] = similarity
        linear = similarity_transform[:2, :2]
        scale = math.sqrt(abs(float(np.linalg.det(linear))))
        rotation = abs(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
        translation = float(np.linalg.norm(similarity_transform[:2, 2]))
        limits = spec["similarity_plausibility"]
        similarity_valid = bool(
            float(limits["minimum_scale"]) <= scale <= float(limits["maximum_scale"])
            and rotation <= float(limits["maximum_absolute_rotation_degrees"])
            and translation <= float(limits["maximum_translation_pixels"])
        )
    models["H1"] = {
        "valid": bool(similarity_valid),
        "transform": similarity_transform,
        "inliers": int(similarity_inliers.sum()) if similarity_inliers is not None else 0,
        "selection": _residual_summary(similarity_transform, source[selection], target[selection]),
        "audit": _residual_summary(similarity_transform, source[audit], target[audit]),
    }
    cv2.setRNGSeed((int(seed) + 1) & 0x7FFFFFFF)
    homography = None
    homography_inliers = None
    if fit.sum() >= 4:
        homography, homography_inliers = cv2.findHomography(
            source[fit].astype(np.float32),
            target[fit].astype(np.float32),
            cv2.RANSAC,
            float(spec["ransac_reprojection_pixels"]),
            maxIters=int(spec["ransac_maximum_iterations"]),
            confidence=float(spec["ransac_confidence"]),
        )
    homography_valid = homography is not None and np.isfinite(homography).all()
    if homography_valid:
        homography = np.asarray(homography, dtype=np.float64)
        if abs(homography[2, 2]) > 1e-12:
            homography /= homography[2, 2]
        condition = float(np.linalg.cond(homography))
        homography_valid = bool(
            math.isfinite(condition)
            and condition <= float(spec["homography_maximum_condition_number"])
            and _homography_has_safe_image_domain(homography, image_shape)
        )
    homography_transform = (
        np.asarray(homography, dtype=np.float64) if homography_valid else identity
    )
    models["H2"] = {
        "valid": bool(homography_valid),
        "transform": homography_transform,
        "inliers": int(homography_inliers.sum()) if homography_inliers is not None else 0,
        "selection": _residual_summary(homography_transform, source[selection], target[selection]),
        "audit": _residual_summary(homography_transform, source[audit], target[audit]),
    }
    selected = "H0"
    absolute = float(spec["upgrade_minimum_absolute_pixels"])
    relative = float(spec["upgrade_minimum_relative_fraction"])
    for candidate in ("H1", "H2"):
        current_error = float(models[selected]["selection"]["median"])
        candidate_error = float(models[candidate]["selection"]["median"])
        if (
            models[candidate]["valid"]
            and math.isfinite(current_error)
            and current_error - candidate_error >= absolute
            and candidate_error <= (1.0 - relative) * current_error
        ):
            selected = candidate
    retained = models[selected]
    retained_count = len(source)
    occupied = len(np.unique(cells))
    selection_count = int(selection.sum())
    audit_count = int(audit.sum())
    initial = float(models["H0"]["audit"]["median"])
    retained_audit = retained["audit"]
    reduction = (
        0.0
        if not math.isfinite(initial) or initial <= 1e-12
        else float(1.0 - float(retained_audit["median"]) / initial)
    )
    large_motion_ok = initial <= float(gate["large_motion_threshold_pixels"]) or reduction >= float(
        gate["large_motion_minimum_reduction_fraction"]
    )
    usable = bool(
        retained_count >= int(gate["minimum_retained_correspondences"])
        and occupied >= int(gate["minimum_occupied_cells"])
        and selection_count >= int(gate["minimum_selection_correspondences"])
        and audit_count >= int(gate["minimum_audit_correspondences"])
        and float(retained_audit["median"]) <= float(gate["maximum_heldout_median_pixels"])
        and float(retained_audit["p90"]) <= float(gate["maximum_heldout_p90_pixels"])
        and large_motion_ok
    )
    camera_failures: list[str] = []
    if retained_count < int(gate["minimum_retained_correspondences"]):
        camera_failures.append("insufficient_background_correspondences")
    if occupied < int(gate["minimum_occupied_cells"]):
        camera_failures.append("insufficient_background_cells")
    if selection_count < int(gate["minimum_selection_correspondences"]):
        camera_failures.append("insufficient_selection_correspondences")
    if audit_count < int(gate["minimum_audit_correspondences"]):
        camera_failures.append("insufficient_audit_correspondences")
    if float(retained_audit["median"]) > float(gate["maximum_heldout_median_pixels"]):
        camera_failures.append("audit_median_above_threshold")
    if float(retained_audit["p90"]) > float(gate["maximum_heldout_p90_pixels"]):
        camera_failures.append("audit_p90_above_threshold")
    if not large_motion_ok:
        camera_failures.append("large_camera_motion_not_reduced")
    public_models = {}
    for name, values in models.items():
        public_models[name] = {
            "valid": bool(values["valid"]),
            "inliers": int(values["inliers"]),
            "selection_median_pixels": (
                float(values["selection"]["median"])
                if math.isfinite(float(values["selection"]["median"]))
                else None
            ),
            "selection_p90_pixels": (
                float(values["selection"]["p90"])
                if math.isfinite(float(values["selection"]["p90"]))
                else None
            ),
            "audit_median_pixels": (
                float(values["audit"]["median"])
                if math.isfinite(float(values["audit"]["median"]))
                else None
            ),
            "audit_p90_pixels": (
                float(values["audit"]["p90"])
                if math.isfinite(float(values["audit"]["p90"]))
                else None
            ),
        }
    return CameraEstimate(
        np.asarray(retained["transform"], dtype=np.float64),
        selected,
        usable,
        {
            "background_retained_points": retained_count,
            "background_fit_points": int(fit.sum()),
            "background_selection_points": selection_count,
            "background_audit_points": audit_count,
            "background_occupied_cells": occupied,
            "camera_selected": selected,
            "camera_usable": usable,
            "camera_failure_reasons": camera_failures,
            "camera_selection_median_pixels": (
                float(retained["selection"]["median"])
                if math.isfinite(float(retained["selection"]["median"]))
                else None
            ),
            "camera_audit_median_pixels": (
                float(retained_audit["median"])
                if math.isfinite(float(retained_audit["median"]))
                else None
            ),
            "camera_audit_p90_pixels": (
                float(retained_audit["p90"])
                if math.isfinite(float(retained_audit["p90"]))
                else None
            ),
            "camera_identity_median_pixels": initial if math.isfinite(initial) else None,
            "camera_reduction_fraction": reduction,
            "camera_models": public_models,
        },
    )


def analyze_actor_correspondence(
    previous: np.ndarray,
    current: np.ndarray,
    previous_box: tuple[float, float, float, float],
    current_box: tuple[float, float, float, float],
    camera: CameraEstimate,
    *,
    elapsed_seconds: float,
    spec: dict[str, Any],
    decomposition: dict[str, Any],
) -> tuple[dict[str, Any], FBTracks]:
    seeds = actor_seed_points(previous, previous_box, spec)
    primary = track_points_forward_backward(
        previous,
        current,
        seeds,
        spec["primary_lk"],
        maximum_error=float(spec["forward_backward_max_pixels"]),
    )
    sensitivity_spec = dict(spec["primary_lk"])
    sensitivity_spec["window"] = spec["sensitivity_lk_window"]
    sensitivity = track_points_forward_backward(
        previous,
        current,
        seeds,
        sensitivity_spec,
        maximum_error=float(spec["forward_backward_max_pixels"]),
    )
    current_height = current_box[3] - current_box[1]
    expansion = float(spec["endpoint_box_expansion_fraction"]) * current_height
    primary_valid = primary.valid & _points_in_box(
        primary.target, expand_box(current_box, expansion)
    )
    sensitivity_valid = sensitivity.valid & _points_in_box(
        sensitivity.target, expand_box(current_box, expansion)
    )
    source = primary.source[primary_valid]
    target = primary.target[primary_valid]
    scale = math.sqrt(
        max(previous_box[3] - previous_box[1], 1e-6) * max(current_box[3] - current_box[1], 1e-6)
    )
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("CCAC requires a positive finite elapsed time")
    denominator = scale * elapsed_seconds
    projected = project_points(camera.transform, source)
    projection_valid = np.isfinite(projected).all(axis=1)
    retained_valid = primary_valid.copy()
    retained_valid[primary_valid] &= projection_valid
    source = source[projection_valid]
    target = target[projection_valid]
    projected = projected[projection_valid]
    normalized = (target - projected) / denominator
    raw = (target - source) / denominator
    translation = np.median(normalized, axis=0) if len(normalized) else None
    residual = normalized - translation if translation is not None else np.empty((0, 2))
    residual_norm = np.linalg.norm(residual, axis=1)
    raw_translation = np.median(raw, axis=0) if len(raw) else None
    y0, y1 = previous_box[1], previous_box[3]
    region = np.floor(3 * (source[:, 1] - y0) / max(y1 - y0, 1e-6)).astype(int)
    region = np.clip(region, 0, 2)
    regions = len(np.unique(region)) if len(region) else 0
    common = primary_valid & sensitivity_valid
    disagreement = np.linalg.norm(primary.target[common] - sensitivity.target[common], axis=1)
    translation_usable = bool(
        camera.usable and len(source) >= int(decomposition["translation_minimum_points"])
    )
    articulation_usable = bool(
        translation_usable
        and len(source) >= int(decomposition["articulation_minimum_points"])
        and regions >= int(decomposition["articulation_minimum_vertical_regions"])
    )
    fb_values = primary.forward_backward_error[retained_valid]
    translation_failure = None
    if not camera.usable:
        translation_failure = "camera_unusable"
    elif len(source) < int(decomposition["translation_minimum_points"]):
        translation_failure = "insufficient_actor_correspondences"
    articulation_failure = translation_failure
    if translation_failure is None:
        if len(source) < int(decomposition["articulation_minimum_points"]):
            articulation_failure = "insufficient_actor_correspondences"
        elif regions < int(decomposition["articulation_minimum_vertical_regions"]):
            articulation_failure = "insufficient_actor_vertical_regions"
    metrics = {
        "actor_seed_points": len(seeds),
        "actor_primary_retained_points": len(source),
        "actor_sensitivity_retained_points": int(sensitivity_valid.sum()),
        "actor_vertical_regions": regions,
        "actor_primary_fb_median_pixels": float(np.median(fb_values)) if len(fb_values) else None,
        "actor_window_disagreement_median_pixels": (
            float(np.median(disagreement)) if len(disagreement) else None
        ),
        "actor_scale_pixels": float(scale),
        "translation_usable": translation_usable,
        "translation_failure_reason": translation_failure,
        "articulation_usable": articulation_usable,
        "articulation_failure_reason": articulation_failure,
        "raw_translation_x_height_per_second": (
            float(raw_translation[0]) if raw_translation is not None else None
        ),
        "raw_translation_y_height_per_second": (
            float(raw_translation[1]) if raw_translation is not None else None
        ),
        "compensated_translation_x_height_per_second": (
            float(translation[0]) if translation_usable and translation is not None else None
        ),
        "compensated_translation_y_height_per_second": (
            float(translation[1]) if translation_usable and translation is not None else None
        ),
        "compensated_translation_norm_height_per_second": (
            float(np.linalg.norm(translation))
            if translation_usable and translation is not None
            else None
        ),
        "articulation_median_height_per_second": (
            float(np.median(residual_norm)) if articulation_usable else None
        ),
        "articulation_p90_height_per_second": (
            float(np.quantile(residual_norm, 0.9)) if articulation_usable else None
        ),
    }
    retained = FBTracks(
        primary.source, primary.target, retained_valid, primary.forward_backward_error
    )
    return metrics, retained


def analyze_pair(
    previous: np.ndarray,
    current: np.ndarray,
    previous_box: tuple[float, float, float, float],
    current_box: tuple[float, float, float, float],
    previous_known_boxes: list[tuple[float, float, float, float]],
    current_known_boxes: list[tuple[float, float, float, float]],
    *,
    elapsed_seconds: float,
    seed: int,
    background_spec: dict[str, Any],
    actor_spec: dict[str, Any],
    camera_spec: dict[str, Any],
    decomposition: dict[str, Any],
    gate: dict[str, Any],
) -> PairAnalysis:
    background_seeds, cells = background_seed_points(
        previous, previous_known_boxes, background_spec
    )
    tracked = track_points_forward_backward(
        previous,
        current,
        background_seeds,
        background_spec["lk"],
        maximum_error=float(background_spec["forward_backward_max_pixels"]),
    )
    endpoint_ok = points_outside_boxes(
        tracked.target,
        current_known_boxes,
        expansion=float(background_spec["known_actor_expansion_pixels"]),
    )
    retained = tracked.valid & endpoint_ok
    background_source = tracked.source[retained]
    background_target = tracked.target[retained]
    retained_cells = cells[retained]
    camera = estimate_camera(
        background_source,
        background_target,
        retained_cells,
        previous.shape,
        seed=seed,
        spec=camera_spec,
        gate=gate,
    )
    actor_metrics, actor_tracks = analyze_actor_correspondence(
        previous,
        current,
        previous_box,
        current_box,
        camera,
        elapsed_seconds=elapsed_seconds,
        spec=actor_spec,
        decomposition=decomposition,
    )
    metrics = {
        "background_seed_points": len(background_seeds),
        "background_fb_median_pixels": (
            float(np.median(tracked.forward_backward_error[retained])) if retained.any() else None
        ),
        **camera.metrics,
        **actor_metrics,
    }
    return PairAnalysis(
        metrics,
        background_source,
        background_target,
        retained_cells,
        actor_tracks.source[actor_tracks.valid],
        actor_tracks.target[actor_tracks.valid],
        actor_tracks.forward_backward_error[actor_tracks.valid],
        camera.transform,
    )


def center_seeded_survival(
    images: list[np.ndarray | None],
    boxes: list[tuple[float, float, float, float] | None],
    actor_spec: dict[str, Any],
    *,
    center_slot: int = 8,
) -> dict[str, Any]:
    if (
        len(images) != 16
        or len(boxes) != 16
        or images[center_slot] is None
        or boxes[center_slot] is None
    ):
        return {
            "center_seed_points": 0,
            "center_track_survival_fraction_75pct": 0.0,
            "center_track_mean_survived_links": 0.0,
        }
    seeds = actor_seed_points(images[center_slot], boxes[center_slot], actor_spec)
    count = len(seeds)
    if not count:
        return {
            "center_seed_points": 0,
            "center_track_survival_fraction_75pct": 0.0,
            "center_track_mean_survived_links": 0.0,
        }
    links = np.zeros(count, dtype=np.int16)
    lk = actor_spec["primary_lk"]
    maximum_error = float(actor_spec["forward_backward_max_pixels"])
    for direction in (1, -1):
        positions = seeds.copy()
        alive = np.ones(count, dtype=bool)
        source_index = center_slot
        while 0 <= source_index + direction < len(images):
            target_index = source_index + direction
            previous, current = images[source_index], images[target_index]
            target_box = boxes[target_index]
            if previous is None or current is None or target_box is None:
                alive[:] = False
                break
            indices = np.flatnonzero(alive)
            if not len(indices):
                break
            tracked = track_points_forward_backward(
                previous,
                current,
                positions[indices],
                lk,
                maximum_error=maximum_error,
            )
            height = target_box[3] - target_box[1]
            expansion = float(actor_spec["endpoint_box_expansion_fraction"]) * height
            survived = tracked.valid & _points_in_box(
                tracked.target, expand_box(target_box, expansion)
            )
            failed_indices = indices[~survived]
            alive[failed_indices] = False
            kept_indices = indices[survived]
            positions[kept_indices] = tracked.target[survived]
            links[kept_indices] += 1
            source_index = target_index
    required = math.ceil(0.75 * 15)
    return {
        "center_seed_points": count,
        "center_track_survival_fraction_75pct": float((links >= required).mean()),
        "center_track_mean_survived_links": float(links.mean()),
    }
