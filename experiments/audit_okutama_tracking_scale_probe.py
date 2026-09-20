"""Independent saved-array audit: no tracker, image decoding, or activity fitting.

All metric/mapping/least-squares algebra is local. The producer's feature,
tracking, metric, mapping and decision helpers are deliberately not imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full_frame_actor_only", "crop_low_detail", "crop_native_detail")
SIZES = ("typical", "tiny")
CASES = ("camera_only", "actor_only", "same_direction", "opposite_direction", "nonrigid")
SAMPLED = np.array([0, 8, 15, 22, 30, 38, 45, 52, 60])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path, *, root=None):
    path, root = Path(path).resolve(), Path(ROOT if root is None else root).resolve()
    require(path.is_relative_to(root), f"artifact outside audit root: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def compare(actual, expected, name, *, atol=1e-10):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected),
            f"{name}: dictionary schema differs",
        )
        for key in expected:
            compare(actual[key], expected[key], f"{name}/{key}", atol=atol)
    elif expected is None:
        require(actual is None, f"{name}: missing value differs")
    elif isinstance(expected, (bool, int, str)):
        require(actual == expected, f"{name}: expected {expected}, got {actual}")
    else:
        require(np.shape(actual) == np.shape(expected), f"{name}: shape differs")
        require(
            np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True),
            f"{name}: numerical mismatch",
        )


def reconstruct_metrics(saved, diagonal):
    """Rebuild all-nine and all-eight-return identity survival without repacking."""
    positions = np.asarray(saved["positions"], dtype=np.float64)
    valid = np.asarray(saved["valid"])
    times = np.asarray(saved["times_seconds"], dtype=np.float64)
    point_ids = np.asarray(saved["point_ids"]).astype(str)
    sampled = np.asarray(saved["sampled_indices"])
    center = int(np.asarray(saved["center_index"]).item())
    require(positions.ndim == 3 and positions.shape[2] == 2, "malformed trajectories")
    t, n, _ = positions.shape
    require(n > 0 and valid.shape == (t, n) and valid.dtype.kind == "b", "malformed visibility")
    require(
        times.shape == (t,) and np.isfinite(times).all() and (np.diff(times) > 0).all(),
        "invalid physical clock",
    )
    require(point_ids.shape == (n,) and len(set(point_ids)) == n, "point identity mismatch")
    require(
        np.isfinite(positions[valid]).all() and np.isnan(positions[~valid]).all(),
        "missing coordinates must remain NaN",
    )
    require(
        sampled.ndim == 1
        and sampled.dtype.kind in "iu"
        and len(sampled) == 9
        and np.array_equal(sampled, np.unique(sampled))
        and sampled.min() >= 0
        and sampled.max() < t,
        "invalid nine-frame sample",
    )
    require(np.count_nonzero(sampled == center) == 1, "center not uniquely sampled")
    require(np.isfinite(diagonal) and diagonal > 0, "invalid normalization diagonal")
    cycle = np.asarray(saved["cycle_error"], dtype=np.float64)
    cycle_valid = np.asarray(saved["cycle_valid"])
    require(
        cycle.shape == (9, n)
        and cycle_valid.shape == cycle.shape
        and cycle_valid.dtype.kind == "b",
        "cycle shape/type mismatch",
    )
    require(
        np.array_equal(np.isfinite(cycle), cycle_valid),
        "cycle missingness disagrees with valid mask",
    )
    require((cycle[cycle_valid] >= 0).all(), "negative cycle distance")
    center_row = int(np.flatnonzero(sampled == center)[0])
    require(not cycle_valid[center_row].any(), "center contributes to cycle error")
    require(
        not (cycle_valid & ~valid[sampled]).any(),
        "valid return cycle requires valid native forward endpoint",
    )
    visible = valid[sampled]
    returned = np.all(np.delete(cycle_valid, center_row, axis=0), axis=0)
    joint = np.all(visible, axis=0) & returned
    values = cycle[cycle_valid]
    return {
        "actor_survival_all_nine": float(np.mean(np.all(visible, axis=0))),
        "actor_reverse_survival_all_noncenter": float(np.mean(returned)),
        "actor_joint_forward_reverse_survival": float(np.mean(joint)),
        "cycle_median_normalized": float(np.median(values) / diagonal) if values.size else None,
        "cycle_pairs_observed": int(cycle_valid.sum()),
        "cycle_pairs_expected": 8 * n,
        "cycle_missing_pairs": 8 * n - int(cycle_valid.sum()),
        "center_excluded": True,
        "independent_tracker_roundtrip": True,
    }


def sample_pass(metrics):
    error = metrics["cycle_median_normalized"]
    return bool(
        metrics["actor_survival_all_nine"] >= 0.75
        and metrics["actor_joint_forward_reverse_survival"] >= 0.75
        and error is not None
        and error <= 0.05
    )


def check_primary_eligibility(count, declared, original_count):
    require(count == original_count and count in (35, 39, 64), "original point budget changed")
    expected = count == 64
    require(
        isinstance(declared, (bool, np.bool_)) and bool(declared) == expected,
        "seed-deficient center gained primary eligibility",
    )
    return expected


def reconstruct_native_mapping(saved):
    local = np.asarray(saved["local_positions"], dtype=np.float64)
    visible = np.asarray(saved["local_valid"])
    global_positions = np.asarray(saved["positions"], dtype=np.float64)
    camera = np.asarray(saved["camera_transforms"], dtype=np.float64)
    crop = np.asarray(saved["crop"])
    native_shape = np.asarray(saved["native_shape"])
    frame = str(np.asarray(saved["coordinate_frame"]).item())
    center = int(np.asarray(saved["center_index"]).item())
    require(
        local.shape == global_positions.shape
        and visible.shape == local.shape[:2]
        and visible.dtype.kind == "b",
        "local trajectory identity mismatch",
    )
    require(
        np.isfinite(local[visible]).all() and np.isnan(local[~visible]).all(),
        "local missingness mismatch",
    )
    require(
        native_shape.shape == (2,)
        and native_shape.dtype.kind in "iu"
        and (native_shape >= 2).all(),
        "invalid native dimensions",
    )
    require(
        camera.shape == (len(local), 3, 3) and np.isfinite(camera).all(),
        "invalid camera transforms",
    )
    compare(
        camera[:, 2],
        np.broadcast_to([0.0, 0.0, 1.0], (len(local), 3)),
        "camera affine row",
        atol=1e-12,
    )
    compare(camera[center], np.eye(3), "center camera must be exact identity", atol=0)
    compare(camera[:, 0, 0], camera[:, 1, 1], "camera similarity scale", atol=1e-10)
    compare(camera[:, 0, 1], -camera[:, 1, 0], "camera similarity rotation", atol=1e-10)
    require(np.all(camera[:, 0, 0] ** 2 + camera[:, 1, 0] ** 2 >= 1e-12), "singular camera")
    require(crop.shape == (4,) and crop.dtype.kind in "iu", "invalid crop representation")
    h, w = native_shape
    if frame == "native":
        require(np.array_equal(crop, [-1, -1, -1, -1]), "full-frame crop sentinel differs")
        result = local.copy()
        inside_local = np.ones(visible.shape, dtype=bool)
    else:
        require(frame == "camera_crop", "unknown coordinate frame")
        left, top, width, height = crop
        require(
            left >= 0
            and top >= 0
            and width >= 2
            and height >= 2
            and left + width <= w
            and top + height <= h,
            "reference crop leaves native image",
        )
        source = local + np.array([left, top])
        # Apply A[t] to each identity directly, independent of producer's mapper.
        result = np.einsum("tij,tnj->tni", camera[:, :2, :2], source) + camera[:, None, :2, 2]
        inside_local = (local[..., 0] >= 0) & (local[..., 0] <= width - 1)
        inside_local &= (local[..., 1] >= 0) & (local[..., 1] <= height - 1)
    inside_native = (result[..., 0] >= 0) & (result[..., 0] <= w - 1)
    inside_native &= (result[..., 1] >= 0) & (result[..., 1] <= h - 1)
    expected_valid = visible & inside_local & inside_native & np.isfinite(result).all(2)
    result[~expected_valid] = np.nan
    require(
        np.array_equal(saved["valid"], expected_valid), "native validity/bounds mapping differs"
    )
    compare(global_positions, result, "native inverse warp", atol=1e-8)
    local_cycle = np.asarray(saved["local_cycle_error"])
    local_cycle_valid = np.asarray(saved["local_cycle_valid"])
    require(
        local_cycle.shape == (9, local.shape[1]) and local_cycle_valid.shape == local_cycle.shape,
        "local cycle shape differs",
    )
    require(
        np.array_equal(np.isfinite(local_cycle), local_cycle_valid),
        "local cycle missingness differs",
    )
    cycle_valid = local_cycle_valid & expected_valid[np.asarray(saved["sampled_indices"])]
    require(np.array_equal(saved["cycle_valid"], cycle_valid), "native cycle remasking differs")
    compare(
        saved["cycle_error"],
        np.where(cycle_valid, local_cycle, np.nan),
        "native cycle distance differs",
        atol=0,
    )
    return frame


def compare_paired_geometry(arrays):
    full, low, high = (arrays[name] for name in ARMS)
    for name, item in arrays.items():
        for key in (
            "seeds_native",
            "point_ids",
            "times_seconds",
            "sampled_indices",
            "center_index",
            "camera_transforms",
            "native_shape",
        ):
            require(np.array_equal(item[key], full[key]), f"{name}: paired {key} differs")
    require(np.array_equal(low["crop"], high["crop"]), "high/low crop geometry differs")
    require(
        str(np.asarray(full["coordinate_frame"]).item()) == "native",
        "full arm not native coordinates",
    )
    for item in (low, high):
        require(
            str(np.asarray(item["coordinate_frame"]).item()) == "camera_crop",
            "crop arm not camera coordinates",
        )


def reconstruct_camera(saved, center=30, diagonal=None):
    points = np.asarray(saved["background_positions720"], dtype=float)
    valid = np.asarray(saved["background_valid"])
    fit = np.asarray(saved["fit_ids"])
    held = np.asarray(saved["held_ids"])
    require(
        points.shape == (61, 64, 2) and valid.shape == (61, 64),
        "camera background population differs",
    )
    require(
        np.array_equal(fit, np.arange(0, 64, 2)) and np.array_equal(held, np.arange(1, 64, 2)),
        "fixed camera fit/held identities changed",
    )
    require(np.isfinite(points[valid]).all(), "nonfinite visible background")
    transform = np.full((61, 3, 3), np.nan)
    down = np.array([[1 / 3, 0, -1 / 3], [0, 1 / 3, -1 / 3], [0, 0, 1.0]])
    up = np.linalg.inv(down)
    errors, fit_counts, held_counts, errors_by_time = [], [], [], []
    for t in range(61):
        indices = fit[valid[center, fit] & valid[t, fit]]
        fit_counts.append(len(indices))
        require(len(indices) >= 3, "insufficient fixed camera fitting support")
        x, y = points[center, indices].T
        design = np.zeros((2 * len(indices), 4))
        design[::2] = np.column_stack((x, -y, np.ones(len(x)), np.zeros(len(x))))
        design[1::2] = np.column_stack((y, x, np.zeros(len(x)), np.ones(len(x))))
        coef, _, rank, _ = np.linalg.lstsq(design, points[t, indices].ravel(), rcond=None)
        require(rank == 4 and np.isfinite(coef).all(), "rank-deficient camera estimate")
        a, b, tx, ty = coef
        require(a * a + b * b >= 1e-12, "singular camera estimate")
        low = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
        transform[t] = up @ low @ down
        frame_errors = []
        if t != center:
            ids = held[valid[center, held] & valid[t, held]]
            prediction = points[center, ids] @ low[:2, :2].T + low[:2, 2]
            frame_errors = np.linalg.norm(prediction - points[t, ids], axis=1).tolist()
            errors.extend(frame_errors)
        held_counts.append(len(frame_errors))
        errors_by_time.append(frame_errors)
    transform[center] = np.eye(3)
    compare(saved["camera_transforms"], transform, "fixed-fit camera reconstruction", atol=1e-8)
    result = {
        "held_noncenter_pairs": len(errors),
        "held_median_error720": float(np.median(errors)) if errors else None,
    }
    if diagonal is not None:
        result["receipt_metrics"] = {
            "status": "CAMERA_FOLLOWING_TRANSFORMS_COMPLETE",
            "camera_fit_ids": fit.tolist(),
            "camera_held_ids": held.tolist(),
            "fit_count_by_time": fit_counts,
            "held_count_by_time": held_counts,
            "held_expected_noncenter_pairs": 60 * 32,
            "held_observed_noncenter_pairs": len(errors),
            "held_noncenter_support_fraction": len(errors) / (60 * 32),
            "held_error_normalized_median": float(np.median(np.asarray(errors) * 3 / diagonal))
            if errors
            else None,
            "center_index": center,
            "center_exact_identity": True,
            "center_in_error_aggregate": False,
            "interpolated_frames": 0,
            "native_actor_diagonal": diagonal,
        }
        result["held_errors_normalized_by_time"] = [
            (np.asarray(e) * 3 / diagonal).tolist() for e in errors_by_time
        ]
    return result


def reconstruct_synthetic(saved, truth, nonrigid):
    """Known-camera compensation, fixed-ID root fit, held articulation scoring."""
    actor = np.asarray(truth["actor_truth"], dtype=float)
    art = np.asarray(truth["articulation_truth"], dtype=float)
    center = int(np.asarray(truth["center_index"]).item())
    diagonal = float(np.asarray(truth["diagonal"]).item())
    require(
        actor.shape == saved["positions"].shape == art.shape, "synthetic truth population differs"
    )
    compare(saved["times_seconds"], truth["times_seconds"], "synthetic truth clock", atol=0)
    require(np.array_equal(saved["point_ids"], truth["point_ids"]), "synthetic truth IDs differ")
    require(int(saved["center_index"]) == center, "synthetic center differs")
    compare(saved["seeds_native"], actor[center], "synthetic original queries", atol=1e-8)
    compare(saved["camera_transforms"], truth["camera_truth"], "synthetic camera truth", atol=1e-8)
    require(actor.shape == (9, 64, 2) and center == 4, "synthetic fixed nine/64 population differs")
    visible = np.asarray(saved["valid"]).copy()
    noncenter = np.arange(9) != center
    temporal = visible.copy()
    temporal[center] = False
    inv = np.linalg.inv(truth["camera_truth"])
    compensated = (
        np.einsum("tij,tnj->tni", inv[:, :2, :2], saved["positions"]) + inv[:, None, :2, 2]
    )
    compensated_truth = np.asarray(truth["camera_compensated_truth"], dtype=float)
    root_truth = np.asarray(truth["root_displacement_truth"], dtype=float)
    compare(
        compensated_truth,
        actor[center][None] + root_truth + art,
        "synthetic component truth identity",
        atol=1e-8,
    )
    compare(
        actor,
        np.einsum("tij,tnj->tni", truth["camera_truth"][:, :2, :2], compensated_truth)
        + truth["camera_truth"][:, None, :2, 2],
        "synthetic camera truth identity",
        atol=1e-8,
    )
    actor_error = np.linalg.norm(compensated - compensated_truth, axis=-1) / diagonal
    endpoint = float(np.median(actor_error[temporal])) if temporal.any() else None
    ids = saved["point_ids"].astype(str).tolist()
    ordered = sorted(ids)
    fit = np.array([ids.index(key) for key in ordered[::2]])
    held = np.array([ids.index(key) for key in ordered[1::2]])
    roots = np.full_like(compensated, np.nan)
    articulation = np.full_like(compensated, np.nan)
    geometry = np.zeros((9, 64), bool)
    for t in np.flatnonzero(noncenter):
        indices = fit[visible[t, fit] & visible[center, fit]]
        if len(indices) < 3:
            continue
        x, y = actor[center, indices].T
        design = np.zeros((2 * len(indices), 4))
        design[::2] = np.column_stack((x, -y, np.ones(len(x)), np.zeros(len(x))))
        design[1::2] = np.column_stack((y, x, np.zeros(len(x)), np.ones(len(x))))
        coef, _, rank, _ = np.linalg.lstsq(design, compensated[t, indices].ravel(), rcond=None)
        if rank != 4 or not np.isfinite(coef).all() or coef[0] ** 2 + coef[1] ** 2 < 1e-12:
            continue
        a, b, tx, ty = coef
        predicted = actor[center] @ np.array([[a, b], [-b, a]]) + np.array([tx, ty])
        mask = visible[t] & visible[center]
        roots[t, mask] = predicted[mask] - actor[center, mask]
        articulation[t, mask] = compensated[t, mask] - predicted[mask]
        geometry[t] = mask
    held_mask = geometry[:, held]
    root_error = np.linalg.norm(roots[:, held] - root_truth[:, held], axis=-1) / diagonal
    root_median = float(np.median(root_error[held_mask])) if held_mask.any() else None
    amplitude = np.linalg.norm(root_truth[:, held], axis=-1) / diagonal
    high = (amplitude > 0.05) & noncenter[:, None]
    high_valid = held_mask & high
    coverage = float(high_valid.sum() / high.sum()) if high.any() else None
    relative = (
        float(np.median(root_error[high_valid] / amplitude[high_valid]))
        if high_valid.any()
        else None
    )
    residual_error = articulation[:, held] - art[:, held]
    energy = float(np.sum(art[:, held][held_mask] ** 2))
    art_relative = (
        float(np.sqrt(np.sum(residual_error[held_mask] ** 2) / energy)) if energy > 1e-12 else None
    )
    metrics = reconstruct_metrics(saved, diagonal)
    center_error = np.linalg.norm(saved["positions"][center] - actor[center], axis=-1)
    checks = {
        "center_query_identity": bool(visible[center].all() and np.max(center_error) <= 0.01),
        "forward_survival": metrics["actor_survival_all_nine"] >= 0.75,
        "joint_independent_return_survival": metrics["actor_joint_forward_reverse_survival"]
        >= 0.75,
        "cycle_error": metrics["cycle_median_normalized"] is not None
        and metrics["cycle_median_normalized"] <= 0.05,
        "actor_known_truth_epe": endpoint is not None and endpoint <= 0.01,
        "held_geometry_support": held_mask.sum() / 256 >= 0.75,
        "root_known_truth_epe": root_median is not None and root_median <= 0.01,
        "relative_root_error": not high.any()
        or (coverage >= 0.75 and relative is not None and relative <= 0.2),
        "nonrigid_known_truth_recovery": not nonrigid
        or (art_relative is not None and art_relative <= 0.5),
    }
    return {
        "pass": all(bool(v) for v in checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "actor_endpoint_median_diagonal": endpoint,
        "root_endpoint_median_diagonal": root_median,
        "relative_root_error_above_0_05_diagonal": relative,
        "nonrigid_known_truth_relative_rms": art_relative,
        "forward_all_time_survival": metrics["actor_survival_all_nine"],
        "joint_forward_return_survival": metrics["actor_joint_forward_reverse_survival"],
        "cycle_median_diagonal": metrics["cycle_median_normalized"],
        "cycle_pairs_expected": 512,
        "cycle_pairs_observed": metrics["cycle_pairs_observed"],
        "actor_pairs_expected": 512,
        "actor_pairs_observed": int(temporal.sum()),
        "held_pairs_expected": 256,
        "held_pairs_observed": int(held_mask.sum()),
        "high_motion_pairs_expected": int(high.sum()),
        "high_motion_pairs_observed": int(high_valid.sum()),
        "center_excluded": True,
        "truth_scope": "synthetic physical correspondence only",
        "camera_estimator_evaluated": False,
        "real_point_gate_waived": False,
    }


def immutable_audit(path, result):
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(
            path.read_text(encoding="utf-8") == payload,
            "refusing to overwrite a different independent audit",
        )
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(payload)


def validate_stop_prefix(aggregate, npz_names, real_outputs):
    require(not real_outputs, "real tracking output exists before synthetic stop")
    require(
        aggregate.get("pass") is False and aggregate.get("stopped_arm") == ARMS[2],
        "not a native-control synthetic stop",
    )
    order = [f"{size}_{case}" for size in SIZES for case in CASES]
    stop = aggregate.get("stopped_case")
    require(stop in order, "unknown synthetic stopping case")
    names = order[: order.index(stop) + 1]
    require(
        set(aggregate["cases"]) == set(names), "synthetic prefix has omissions or post-stop cases"
    )
    allowed = {f"synthetic_truth_{name}.npz" for name in names}
    allowed.update(f"synthetic_{name}_{arm}.npz" for name in names for arm in ARMS)
    require(set(npz_names) == allowed, "synthetic files are not the exact ordered prefix")
    return names


def audit_synthetic_stop(run: Path) -> dict:
    """Validate only the ordered synthetic prefix through first native failure."""
    run = Path(run).resolve()
    require(run.is_relative_to(ROOT), "run outside repository")
    lock_path = run / "execution_lock.json"
    lock = read_json(lock_path)
    require(
        lock["status"] == "TRACKING_SCALE_LOCKED_BEFORE_INFERENCE" and tuple(lock["arms"]) == ARMS,
        "incorrect synthetic execution lock",
    )
    require(lock["activity_fits"] == 0 and lock["tracker_calls_at_lock"] == 0, "invalid phase")
    require(
        tuple(lock["protocol"]["synthetic_controls"]["sizes"]) == SIZES
        and tuple(lock["protocol"]["synthetic_controls"]["cases"]) == CASES,
        "synthetic order changed",
    )
    for item in lock["dependencies"]:
        actual = record(ROOT / item["path"])
        require(
            actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
            f"locked input changed: {item['path']}",
        )
    aggregate_path = run / "synthetic_controls.json"
    aggregate = read_json(aggregate_path)
    real = list(run.glob("clip_[0-9][0-9].json"))
    real += [path for arm in ARMS for path in run.glob(f"clip_*_{arm}.npz")]
    names = validate_stop_prefix(aggregate, [p.name for p in run.glob("synthetic*.npz")], real)
    artifacts, reports = [record(lock_path), record(aggregate_path)], []
    for i, name in enumerate(names):
        size, case = name.split("_", 1)
        truth_path = run / f"synthetic_truth_{name}.npz"
        truth = read_npz(truth_path)
        artifacts.append(record(truth_path))
        arrays = {}
        require(set(aggregate["cases"][name]) == set(ARMS), "synthetic case lacks all3arms")
        for arm in ARMS:
            prefix = run / f"synthetic_{name}_{arm}"
            saved = arrays[arm] = read_npz(prefix.with_suffix(".npz"))
            receipt = read_json(prefix.with_suffix(".json"))
            artifacts.extend(record(prefix.with_suffix(ext)) for ext in (".npz", ".json"))
            reconstruct_native_mapping(saved)
            expected = reconstruct_synthetic(saved, truth, case == "nonrigid")
            require(
                receipt["fixture_id"] == f"native4k_scale_v1::{size}::{case}",
                "synthetic fixture identity differs",
            )
            for key, value in expected.items():
                compare(receipt[key], value, f"{name}/{arm}/{key}")
            require(
                aggregate["cases"][name][arm] == receipt, "synthetic aggregate differs from receipt"
            )
            if arm == ARMS[2]:
                require(
                    expected["pass"] == (i < len(names) - 1),
                    "stop was not first failed native control",
                )
            reports.append({"size": size, "case": case, "arm": arm, "metrics": expected})
        compare_paired_geometry(arrays)
    result = {
        "status": "SCALE_PROBE_SYNTHETIC_STOP_REPLAY_PASS",
        "synthetic_arm_replays": len(reports),
        "completed_case_prefix": names,
        "stopped_case": names[-1],
        "stopped_arm": ARMS[2],
        "real_arm_replays": 0,
        "real_centers_run": 0,
        "tracker_inferences": 0,
        "image_decodes": 0,
        "activity_fits": 0,
        "execution_lock_sha256": sha256(lock_path),
        "checks": reports,
        "audited_artifacts": artifacts,
        "scope": "Independent physical-truth/geometry/cycle arithmetic for the saved synthetic prefix; no48-real-arm completion claim.",
    }
    immutable_audit(run / "independent_synthetic_audit.json", result)
    return result


def reconstruct_assessment(reports, population):
    require(
        len(reports) == 48 and len(population) == 16, "mechanism assessment needs complete48/16"
    )
    lookup = {(r["selection_index"], r["arm"]): r for r in reports}
    require(len(lookup) == 48, "duplicate arm reports")
    eligible = [r for r in population if r["primary_eligible"]]
    require(len(eligible) == 14, "mechanism population changed")
    counts = {
        arm: sum(
            r["primary_eligible"] and sample_pass(lookup[(r["selection_index"], arm)]["metrics"])
            for r in population
        )
        for arm in ARMS
    }
    values = np.array(
        [
            [
                lookup[(r["selection_index"], arm)]["metrics"][
                    "actor_joint_forward_reverse_survival"
                ]
                for arm in ARMS
            ]
            for r in eligible
        ]
    )
    delta = values[:, 2] - values[:, 1]
    full_delta = values[:, 2] - values[:, 0]
    scenes = sorted({r["scenario"] for r in eligible})
    scene_means = {
        scene: float(np.mean(delta[[r["scenario"] == scene for r in eligible]])) for scene in scenes
    }
    draws = np.random.default_rng(20260920).integers(0, 14, size=(10000, 14))
    interval = np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist()
    checks = {
        "at_least13_of_all16_native_pass": counts[ARMS[2]] >= 13,
        "at_least3_more_native_passes_than_low": counts[ARMS[2]] - counts[ARMS[1]] >= 3,
        "at_least3_more_native_passes_than_full": counts[ARMS[2]] - counts[ARMS[0]] >= 3,
        "median_native_minus_low_joint_at_least0_15": float(np.median(delta)) >= 0.15,
        "both_scenarios_positive_native_minus_low": len(scenes) == 2
        and all(v > 0 for v in scene_means.values()),
        "exploratory_center_bootstrap_lower_positive": interval[0] > 0,
    }
    return {
        "primary_pass_counts_of16": counts,
        "primary_eligible": 14,
        "native_minus_low_joint_median": float(np.median(delta)),
        "native_minus_full_joint_median": float(np.median(full_delta)),
        "native_minus_low_per_scenario_mean": scene_means,
        "center_bootstrap_95_mean_joint_gain": interval,
        "checks": checks,
        "mechanism_supported": all(checks.values()),
        "original_smoke_reclassified": False,
        "seed_deficient_centers_remain_primary_failures": True,
        "expanded128_authorized": False,
        "activity_training_authorized": False,
        "uncertainty_limit": "only2scenarios; center bootstrap is exploratory, not independent scenario generalization",
    }


def audit(run: Path) -> dict:
    run = Path(run).resolve()
    require(run.is_relative_to(ROOT), "run outside repository")
    lock_path = run / "execution_lock.json"
    lock = read_json(lock_path)
    require(
        lock["status"] == "TRACKING_SCALE_LOCKED_BEFORE_INFERENCE",
        "execution was not locked before inference",
    )
    require(
        tuple(lock["arms"]) == ARMS
        and lock["activity_fits"] == 0
        and lock["tracker_calls_at_lock"] == 0,
        "phase/arm lock differs",
    )
    for item in lock["dependencies"]:
        actual = record(ROOT / item["path"])
        require(
            actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
            f"locked dependency changed: {item['path']}",
        )
    population = lock["population"]
    require(
        len(population) == 16 and [r["selection_index"] for r in population] == list(range(16)),
        "fixed16 population changed",
    )
    require(len(set(r["sample_id"] for r in population)) == 16, "duplicate center identities")
    artifacts = [record(lock_path)]
    reports, camera_reports = [], []
    eligible_count = 0
    for clip in population:
        index = clip["selection_index"]
        prefix = run / f"clip_{index:02d}"
        row_path = prefix.with_suffix(".json")
        row = read_json(row_path)
        artifacts.append(record(row_path))
        old_path = ROOT / lock["protocol"]["parent_run"] / f"clip_{index:02d}.json"
        old = read_json(old_path)
        original_count = 64 if old["status"] == "COMPLETE" else old["actor_count"]
        eligible = check_primary_eligibility(
            clip["point_count"], clip["primary_eligible"], original_count
        )
        eligible_count += int(eligible)
        require(
            row["primary_eligible"] == eligible and row["point_count"] == original_count,
            "reported eligibility differs",
        )
        for key in ("selection_index", "sample_id", "scenario", "crop"):
            require(row[key] == clip[key], f"center receipt {key} differs from lock")
        seeds = read_npz(ROOT / clip["seeds"]["path"])
        require(seeds["points"].shape == (original_count, 2), "seed bank row count differs")
        box = np.asarray(clip["native_box"], dtype=float)
        diagonal = float(np.linalg.norm(box[2:] - box[:2]))
        compare(row["box_diagonal"], diagonal, "actor diagonal")
        require(
            len(row["frame_sha256"]) == 61 and row["frame_sha256"][30] == clip["center_rgb_sha256"],
            "decoded source clock/center binding differs",
        )
        require(
            row["frame_sha256"] == clip["frame_sha256"], "decoded frame bridge changed after lock"
        )
        if old["status"] == "COMPLETE":
            require(
                row["frame_sha256"] == old["frame_sha256"],
                "historical decoded frame bridge changed",
            )
        camera_path = run / f"clip_{index:02d}_camera.npz"
        camera = read_npz(camera_path)
        camera_check = reconstruct_camera(camera, diagonal=diagonal)
        camera_receipt_path = run / f"clip_{index:02d}_camera.json"
        camera_receipt = read_json(camera_receipt_path)
        for key, value in camera_check["receipt_metrics"].items():
            compare(camera_receipt[key], value, f"camera/{index}/{key}")
        for t, errors in enumerate(camera_check["held_errors_normalized_by_time"]):
            compare(
                camera_receipt["held_errors_normalized_by_time"][t],
                errors,
                f"camera/{index}/held-errors/{t}",
            )
        compare(
            camera["background_positions720"][30],
            (camera["background_seeds_native"] + 0.5) / 3 - 0.5,
            "background pixel-center query mapping",
            atol=1e-4,
        )
        camera_reports.append({"selection_index": index, **camera_check})
        artifacts.extend((record(camera_path), record(camera_receipt_path)))
        arrays = {}
        for arm in ARMS:
            path = run / f"clip_{index:02d}_{arm}.npz"
            saved = arrays[arm] = read_npz(path)
            artifacts.append(record(path))
            require(
                saved["positions"].shape == (61, original_count, 2),
                "real trajectory population differs",
            )
            require(
                np.array_equal(saved["seeds_native"], seeds["points"]),
                "original real queries changed",
            )
            require(
                np.array_equal(saved["point_ids"], seeds["point_ids"]),
                "original real point IDs changed",
            )
            require(
                np.array_equal(saved["sampled_indices"], SAMPLED)
                and int(saved["center_index"]) == 30,
                "real sampled clock differs",
            )
            require(
                np.array_equal(saved["native_shape"], [2160, 3840]),
                "real native dimensions changed",
            )
            compare(
                saved["camera_transforms"],
                camera["camera_transforms"],
                "arm camera provenance",
                atol=0,
            )
            compare(
                saved["times_seconds"],
                camera["times_seconds"],
                "camera/actor physical clock",
                atol=1e-8,
            )
            reconstruct_native_mapping(saved)
            compare(saved["positions"][30], seeds["points"], "center query identity", atol=0.01)
            require(saved["valid"][30].all(), "center query validity lost")
            metrics = reconstruct_metrics(saved, diagonal)
            for key, value in metrics.items():
                compare(row["arms"][arm][key], value, f"{index}/{arm}/{key}")
            passed = sample_pass(metrics)
            compare(row["arms"][arm]["sample_pass"], passed, "sample gate")
            compare(row["arms"][arm]["primary_pass"], eligible and passed, "primary gate")
            reports.append(
                {
                    "selection_index": index,
                    "arm": arm,
                    "point_count": original_count,
                    "primary_eligible": eligible,
                    "sample_pass": passed,
                    "primary_pass": eligible and passed,
                    "metrics": metrics,
                }
            )
        compare_paired_geometry(arrays)
        crop = clip["crop"]
        expected_crop = [crop[key] for key in ("left", "top", "width", "height")]
        require(
            np.array_equal(arrays[ARMS[1]]["crop"], expected_crop), "locked crop bounds changed"
        )
    require(
        eligible_count == 14 and len(reports) == 48, "incomplete primary population or arm matrix"
    )
    aggregate_path = run / "synthetic_controls.json"
    aggregate = read_json(aggregate_path)
    artifacts.append(record(aggregate_path))
    require(
        aggregate.get("pass") is True
        and set(aggregate["cases"]) == {f"{s}_{c}" for s in SIZES for c in CASES},
        "full audit requires all10native synthetic controls",
    )
    synthetic_reports = []
    for size in SIZES:
        for case in CASES:
            path = run / f"synthetic_truth_{size}_{case}.npz"
            truth = read_npz(path)
            artifacts.append(record(path))
            arrays = {}
            for arm in ARMS:
                prefix = run / f"synthetic_{size}_{case}_{arm}"
                saved = arrays[arm] = read_npz(prefix.with_suffix(".npz"))
                receipt = read_json(prefix.with_suffix(".json"))
                require(
                    receipt["fixture_id"] == f"native4k_scale_v1::{size}::{case}",
                    "synthetic fixture identity changed",
                )
                require(
                    aggregate["cases"][f"{size}_{case}"][arm] == receipt,
                    "synthetic aggregate differs from arm receipt",
                )
                artifacts.extend(record(prefix.with_suffix(ext)) for ext in (".npz", ".json"))
                reconstruct_native_mapping(saved)
                expected = reconstruct_synthetic(saved, truth, case == "nonrigid")
                for key, value in expected.items():
                    compare(receipt[key], value, f"synthetic/{size}/{case}/{arm}/{key}")
                if arm == ARMS[2]:
                    require(expected["pass"], "native-detail physical synthetic control failed")
                synthetic_reports.append(
                    {"size": size, "case": case, "arm": arm, "metrics": expected}
                )
            compare_paired_geometry(arrays)
    require(len(synthetic_reports) == 30, "incomplete synthetic arm matrix")
    result = {
        "status": "SCALE_PROBE_INDEPENDENT_ARRAY_AUDIT_PASS",
        "centers": 16,
        "real_arm_replays": 48,
        "synthetic_arm_replays": 30,
        "primary_eligible_centers": eligible_count,
        "tracker_inferences": 0,
        "image_decodes": 0,
        "activity_fits": 0,
        "execution_lock_sha256": sha256(lock_path),
        "checks": reports,
        "camera_checks": camera_reports,
        "synthetic_checks": synthetic_reports,
        "mechanism_assessment": reconstruct_assessment(reports, population),
        "audited_artifacts": artifacts,
        "limitations": [
            "Reconstructs saved geometry, metric/gate arithmetic and hashes; does not replay learned tracker inference.",
            "Independent least-squares camera reconstruction is measurement arithmetic on saved background tracks, not new scientific model training.",
            "Source hashes and paired transforms bind preprocessing implementation; rendered pixels are not re-decoded here.",
            "Real-cycle and visibility scores are not physical correspondence ground truth or classification performance.",
            "Only native-detail synthetic controls must pass; lower-detail arms are prespecified diagnostics.",
        ],
    }
    path = run / "independent_audit.json"
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(
            path.read_text(encoding="utf-8") == payload,
            "refusing to overwrite a different independent audit",
        )
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    result = audit(parser.parse_args().run)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "centers", "real_arm_replays", "synthetic_arm_replays")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
