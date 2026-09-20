"""Frozen-model actor-scale experiment with a matched low-detail crop control.

No activity fitting, human feedback, fabricated real-video points, or old evidence edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments import pilot_okutama_learned_persistent_articulation as parent  # noqa: E402
from experiments.tracking_scale_synthetic import (  # noqa: E402
    score_synthetic_scale_tracks,
    synthetic_scale_case,
)
from hac.camera_following_scale_probe import (  # noqa: E402
    camera_from_background,
    map_camera_tracks_to_native,
    prepare_camera_crop_pair,
)
from hac.learned_persistent_tracking import (  # noqa: E402
    CycledTracks,
    LocalCoTracker3,
    file_sha256,
    independently_cycled_tracks,
    local_model_identity,
    point_survival_and_cycle,
)
from hac.persistent_articulation import actor_seed_points  # noqa: E402
from hac.persistent_articulation_v2 import TrackSequence  # noqa: E402
from hac.persistent_tracking import track_persistent  # noqa: E402
from hac.tracking_scale_probe import (  # noqa: E402
    FixedCrop,
    fixed_actor_crop,
    points_to_crop,
)

PROTOCOL = ROOT / "experiments/okutama_tracking_scale_probe_protocol.json"
DEFAULT = ROOT / ".runs/research_20260920/tracking_scale_probe_v1"
ARMS = ("full_frame_actor_only", "crop_low_detail", "crop_native_detail")
SAMPLED = (0, 8, 15, 22, 30, 38, 45, 52, 60)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    content = json.dumps(parent.old.json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def save(path, **values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **values)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def record(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def source_paths():
    return [Path(__file__), PROTOCOL, Path(parent.__file__),
            ROOT / "src/hac/tracking_scale_probe.py", ROOT / "tests/test_tracking_scale_probe.py",
            ROOT / "src/hac/camera_following_scale_probe.py", ROOT / "tests/test_camera_following_scale_probe.py",
            ROOT / "experiments/tracking_scale_synthetic.py", ROOT / "tests/test_tracking_scale_synthetic.py",
            ROOT / "tests/test_tracking_scale_probe_runner.py",
            ROOT / "experiments/audit_okutama_tracking_scale_probe.py",
            ROOT / "tests/test_audit_tracking_scale_probe.py",
            *parent.SOURCE_FILES]


def center_frame(clip, metadata):
    path = Path(metadata["path"])
    require(path.stat().st_size == metadata["bytes"] and path.stat().st_mtime_ns == metadata["mtime_ns"], "video source changed")
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened() and cap.set(cv2.CAP_PROP_POS_FRAMES, clip["center_frame"]), "center seek failed")
        ok, frame = cap.read()
        require(ok and frame.shape == (2160, 3840, 3), "bad center frame")
        require(abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - clip["center_frame"] - 1) <= .5, "center frame index mismatch")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def prepare(run):
    require(not run.exists(), "prepare needs a fresh directory")
    protocol = read(PROTOCOL)
    previous = ROOT / protocol["parent_run"]
    plan_path = ROOT / protocol["parent_prepared_plan"]
    plan, summary = read(plan_path), read(previous / "summary.json")
    require(summary["status"] == "NO_GO_REAL_MEASUREMENT" and summary["centers"] == 16, "parent was not completed real-measurement NO_GO")
    require(plan["source_hashes"] == parent.source_hashes(), "historical prepared sources changed")
    require(local_model_identity(Path(plan["model"]["repository"]), Path(plan["model"]["checkpoint"])) == plan["model"], "tracker identity changed")
    run.mkdir(parents=True)
    population = []
    dependencies = [plan_path, previous / "summary.json", previous / "execution_lock.json", *source_paths(),
        ROOT / ".runs/research_20260920/tracking_scale_preflight_v1/CAMERA_CROP_RATIONALE.md",
        ROOT / ".runs/research_20260920/tracking_scale_preflight_v1/VALIDATION_RECEIPT.json",
        ROOT / ".runs/research_20260920/tracking_scale_preflight_v1/forensics.json",
        ROOT / ".runs/research_20260920/tracking_scale_preflight_v1/input_hashes.json"]
    for clip in plan["clips"]:
        old_report_path = previous / f"clip_{clip['selection_index']:02d}.json"
        old_report = read(old_report_path)
        frames, times, frame_hashes = parent.read_rgb_bridge(clip, plan["sources"][clip["recording"]])
        frame = frames[30]
        digest = hashlib.sha256(frame).hexdigest()
        points = actor_seed_points(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), tuple(clip["native_box"]), maximum=64).points
        require(len(points) > 0, "zero observable actor points; stop rather than invent points")
        eligible = len(points) == protocol["primary_query_count"]
        dependencies.append(old_report_path)
        if old_report["status"] == "COMPLETE":
            require(eligible and digest == old_report["frame_sha256"][30], "old center/seed eligibility did not replay")
            require(frame_hashes == old_report["frame_sha256"], "decoded source bridge differs from original screen")
            old_npz = previous / f"clip_{clip['selection_index']:02d}_native4k.npz"
            with np.load(old_npz, allow_pickle=False) as saved:
                require(np.array_equal(points, saved["seeds"][:64]), "original actor query bank changed")
            dependencies.append(old_npz)
        else:
            require(not eligible and len(points) == old_report["actor_count"], "old seed-shortage count changed")
        roi = fixed_actor_crop(tuple(clip["native_box"]), frame.shape[:2], tuple(protocol["model_shape"]), context=4.0)
        exclusions = [tuple(3.0 * x for x in b) for b in clip["known_boxes_720"]]
        background = parent.background_seed_points(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), tuple(clip["native_box"]), maximum=64, exclude_boxes=exclusions).points
        require(len(background) == 64, "camera query bank has insufficient observed background points")
        if old_report["status"] == "COMPLETE":
            with np.load(old_npz, allow_pickle=False) as saved:
                require(np.array_equal(background, saved["seeds"][64:]), "original background query bank changed")
        gray720 = [cv2.cvtColor(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
        lk = track_persistent(gray720, parent.scaled_pixel_centers(background, 1 / 3), center_index=30,
                              spec=parent.old.LK_SPEC, maximum_error=1.0)
        box = clip["native_box"]
        camera, camera_receipt = camera_from_background(lk.positions, lk.valid, center_index=30,
                        actor_diag_native=math.hypot(box[2] - box[0], box[3] - box[1]))
        camera_path = run / f"clip_{clip['selection_index']:02d}_camera.npz"
        save(camera_path, background_positions720=lk.positions, background_valid=lk.valid,
             fit_ids=np.arange(0,64,2), held_ids=np.arange(1,64,2), camera_transforms=camera,
             times_seconds=times, center_index=np.asarray(30), background_seeds_native=background)
        write(run / f"clip_{clip['selection_index']:02d}_camera.json", camera_receipt)
        dependencies.extend([camera_path, run / f"clip_{clip['selection_index']:02d}_camera.json"])
        ids = tuple(f"actor-{i:03d}" for i in range(len(points)))
        save(run / f"seeds_{clip['selection_index']:02d}.npz", points=points, point_ids=np.asarray(ids))
        population.append({**clip, "center_rgb_sha256": digest, "point_count": len(points), "primary_eligible": eligible,
                           "crop": asdict(roi), "seeds": record(run / f"seeds_{clip['selection_index']:02d}.npz"),
                           "camera": record(camera_path), "frame_sha256": frame_hashes})
        print(f"prepared center {clip['selection_index']:02d}: {len(points)} real points; primary={eligible}", flush=True)
        del frames, frame, gray720, lk
    require(len(population) == 16 and sum(x["primary_eligible"] for x in population) == 14, "fixed cohort changed")
    dependencies.extend(run / x["seeds"]["path"].split("/")[-1] for x in population)
    import torch
    lock = {"status": "TRACKING_SCALE_LOCKED_BEFORE_INFERENCE", "protocol": protocol,
            "population": population, "source_metadata": plan["sources"], "model": plan["model"],
            "dependencies": [record(p) for p in sorted(set(dependencies))],
            "activity_fits": 0, "tracker_calls_at_lock": 0, "arms": list(ARMS),
            "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "numpy": np.__version__, "opencv": cv2.__version__}}
    write(run / "execution_lock.json", lock)
    print("TRACKING_SCALE_LOCKED_BEFORE_INFERENCE", flush=True)


def validate(run):
    require(not (run / "failure.json").exists(), "failed run cannot be silently resumed")
    lock = read(run / "execution_lock.json")
    require(lock["status"] == "TRACKING_SCALE_LOCKED_BEFORE_INFERENCE", "wrong execution lock")
    for item in lock["dependencies"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and file_sha256(path) == item["sha256"], f"locked input changed: {path}")
    return lock


def native_tracks(item, crop, camera, native_shape):
    if crop is None:
        positions = item.sequence.positions.copy()
        height, width = native_shape
        valid = item.sequence.valid & np.isfinite(positions).all(-1)
        valid &= (positions[..., 0] >= 0) & (positions[..., 0] <= width - 1)
        valid &= (positions[..., 1] >= 0) & (positions[..., 1] <= height - 1)
        positions[~valid] = np.nan
    else:
        positions, valid = map_camera_tracks_to_native(item.sequence.positions, item.sequence.valid, camera, crop, native_shape)
    sequence = TrackSequence(positions, valid,
                             item.sequence.times_seconds, item.sequence.point_ids)
    cycle_valid = item.cycle_valid & valid[np.asarray(item.sampled_indices)]
    cycle_error = np.where(cycle_valid, item.cycle_error, np.nan)
    return CycledTracks(sequence, cycle_error, cycle_valid, item.sampled_indices,
                        item.predictor_calls, item.center_index)


def track(backend, video, seeds, times, ids, center, sampled, camera, native_shape, crop=None):
    query = seeds if crop is None else points_to_crop(seeds, crop)
    def physically_observed_predict(queries, reverse):
        positions, valid = backend.predict(video, queries, reverse=reverse)
        if crop is None:
            height, width = native_shape
            valid = valid & np.isfinite(positions).all(-1)
            valid &= (positions[..., 0] >= 0) & (positions[..., 0] <= width - 1)
            valid &= (positions[..., 1] >= 0) & (positions[..., 1] <= height - 1)
        else:
            _, valid = map_camera_tracks_to_native(positions, valid,
                        camera[::-1] if reverse else camera, crop, native_shape)
        positions = positions.copy()
        positions[~valid] = np.nan
        return positions, valid
    item = independently_cycled_tracks(physically_observed_predict, query,
            times_seconds=times, center_index=center, sampled_indices=sampled, point_ids=ids)
    return native_tracks(item, crop, camera, native_shape), item


def sample_pass(metrics):
    cycle = metrics["cycle_median_normalized"]
    return bool(metrics["actor_survival_all_nine"] >= .75 and metrics["actor_joint_forward_reverse_survival"] >= .75
                and cycle is not None and cycle <= .05)


def save_tracks(path, item, local, seeds, crop, camera, native_shape):
    save(path, positions=item.sequence.positions, valid=item.sequence.valid, times_seconds=item.sequence.times_seconds,
         point_ids=np.asarray(item.sequence.point_ids), cycle_error=item.cycle_error, cycle_valid=item.cycle_valid,
         sampled_indices=np.asarray(item.sampled_indices), center_index=np.asarray(item.center_index), seeds_native=seeds,
         crop=np.asarray([crop.left, crop.top, crop.width, crop.height] if crop else [-1, -1, -1, -1]),
         local_positions=local.sequence.positions, local_valid=local.sequence.valid, camera_transforms=camera,
         local_cycle_error=local.cycle_error, local_cycle_valid=local.cycle_valid,
         native_shape=np.asarray(native_shape), coordinate_frame=np.asarray("camera_crop" if crop else "native"))


def run_synthetic(backend, run, protocol):
    result = {}
    for size in protocol["synthetic_controls"]["sizes"]:
        for case in protocol["synthetic_controls"]["cases"]:
            name = size + "_" + case
            truth = synthetic_scale_case(size, case)
            seeds = truth["actor_truth"][4]
            ids = tuple(f"actor-{i:03d}" for i in range(len(seeds)))
            shape = truth["frames"][0].shape[:2]
            crop = fixed_actor_crop(truth["box_native"], shape, backend.model_shape, context=4.0)
            camera = truth["camera_truth"]
            save(run / f"synthetic_truth_{name}.npz", **{k: truth[k] for k in ("actor_truth", "articulation_truth", "camera_truth", "root_displacement_truth", "camera_compensated_truth", "times_seconds", "center_index", "box_native", "diagonal")}, point_ids=np.asarray(ids))
            full = backend.prepare_video(truth["frames"])
            high, low = prepare_camera_crop_pair(backend, full, truth["frames"], crop, camera)
            result[name] = {}
            for arm, video, roi in zip(ARMS, (full, low, high), (None, crop, crop), strict=True):
                item, local = track(backend, video, seeds, truth["times_seconds"], ids, 4, tuple(range(9)), camera, shape, roi)
                result[name][arm] = score_synthetic_scale_tracks(item.sequence, truth, item.cycle_error, item.cycle_valid)
                save_tracks(run / f"synthetic_{name}_{arm}.npz", item, local, seeds, roi, camera, shape)
                write(run / f"synthetic_{name}_{arm}.json", result[name][arm])
                print(json.dumps({"synthetic_case": name, "arm": arm,
                                  "pass": result[name][arm]["pass"]}), flush=True)
            if not result[name][ARMS[2]]["pass"]:
                return {"pass": False, "cases": result, "stopped_case": name, "stopped_arm": ARMS[2]}
            del full, low, high, truth
    return {"pass": True, "cases": result}


def center_bootstrap(delta):
    delta = np.asarray(delta, float)
    draws = np.random.default_rng(20260920).integers(0, len(delta), size=(10000, len(delta)))
    return np.quantile(delta[draws].mean(1), [.025, .975]).tolist()


def assess(rows):
    require(len(rows) == 16, "never score partial population")
    eligible = [r for r in rows if r["primary_eligible"]]
    require(len(eligible) == 14, "exactly14originally eligible centers required")
    counts = {a: sum(r["primary_eligible"] and sample_pass(r["arms"][a]) for r in rows) for a in ARMS}
    low_delta = np.asarray([r["arms"][ARMS[2]]["actor_joint_forward_reverse_survival"] - r["arms"][ARMS[1]]["actor_joint_forward_reverse_survival"] for r in eligible])
    full_delta = np.asarray([r["arms"][ARMS[2]]["actor_joint_forward_reverse_survival"] - r["arms"][ARMS[0]]["actor_joint_forward_reverse_survival"] for r in eligible])
    scenes = sorted({r["scenario"] for r in eligible})
    scene_delta = {s: float(np.mean([d for d, r in zip(low_delta, eligible, strict=True) if r["scenario"] == s])) for s in scenes}
    interval = center_bootstrap(low_delta)
    checks = {"at_least13_of_all16_native_pass": counts[ARMS[2]] >= 13,
              "at_least3_more_native_passes_than_low": counts[ARMS[2]] - counts[ARMS[1]] >= 3,
              "at_least3_more_native_passes_than_full": counts[ARMS[2]] - counts[ARMS[0]] >= 3,
              "median_native_minus_low_joint_at_least0_15": float(np.median(low_delta)) >= .15,
              "both_scenarios_positive_native_minus_low": len(scene_delta) == 2 and all(v > 0 for v in scene_delta.values()),
              "exploratory_center_bootstrap_lower_positive": interval[0] > 0}
    return {"primary_pass_counts_of16": counts, "primary_eligible": len(eligible),
            "native_minus_low_joint_median": float(np.median(low_delta)), "native_minus_full_joint_median": float(np.median(full_delta)),
            "native_minus_low_per_scenario_mean": scene_delta, "center_bootstrap_95_mean_joint_gain": interval,
            "checks": checks, "mechanism_supported": all(checks.values()),
            "original_smoke_reclassified": False, "seed_deficient_centers_remain_primary_failures": True,
            "expanded128_authorized": False, "activity_training_authorized": False,
            "uncertainty_limit": "only2scenarios; center bootstrap is exploratory, not independent scenario generalization"}


def execute(run):
    lock = validate(run)
    require(not (run / "inference_started.json").exists(), "inference already started; do not duplicate or silently retry")
    write(run / "inference_started.json", {"execution_lock_sha256": file_sha256(run / "execution_lock.json"), "unix_time": time.time()})
    backend = LocalCoTracker3(lock["model"])
    require(tuple(backend.model_shape) == tuple(lock["protocol"]["model_shape"]), "model-resolution assumption changed")
    start = time.perf_counter()
    backend.torch.cuda.reset_peak_memory_stats()
    synthetic = run_synthetic(backend, run, lock["protocol"])
    write(run / "synthetic_controls.json", synthetic)
    if not synthetic["pass"]:
        from experiments.audit_okutama_tracking_scale_probe import audit_synthetic_stop
        independent = audit_synthetic_stop(run)
        write(run / "summary.json", {"status": "NO_GO_SYNTHETIC_SCALE_CONTROL", "synthetic": synthetic,
            "activity_fits": 0, "real_centers": 0, "elapsed_seconds": time.perf_counter() - start,
            "predictor_calls": backend.calls, "peak_cuda_allocated_bytes": backend.torch.cuda.max_memory_allocated(),
            "audit_status": independent["status"],
            "audit_sha256": file_sha256(run / "independent_synthetic_audit.json"),
            "execution_lock_sha256": file_sha256(run / "execution_lock.json")})
        print("NO_GO_SYNTHETIC_SCALE_CONTROL", flush=True)
        return
    print("all10 native-detail synthetic cases PASS; all30 arm/case results saved; starting fixed16center probe", flush=True)
    rows = []
    for clip in lock["population"]:
        require(time.perf_counter() - start < 3600, "declared one-hour runtime budget exhausted")
        begin = time.perf_counter()
        frames, times, hashes = parent.read_rgb_bridge(clip, lock["source_metadata"][clip["recording"]])
        require(hashes == clip["frame_sha256"], "decoded physical frame bridge differs from preparation")
        with np.load(ROOT / clip["seeds"]["path"], allow_pickle=False) as saved:
            seeds, ids = saved["points"], tuple(saved["point_ids"].tolist())
        box = clip["native_box"]
        diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
        crop = FixedCrop(**clip["crop"])
        with np.load(ROOT / clip["camera"]["path"], allow_pickle=False) as saved:
            camera = saved["camera_transforms"]
            require(np.array_equal(times, saved["times_seconds"]), "camera clock differs from actor clock")
        full = backend.prepare_video(frames)
        high, low = prepare_camera_crop_pair(backend, full, frames, crop, camera)
        row = {"selection_index": clip["selection_index"], "sample_id": clip["sample_id"], "scenario": clip["scenario"],
               "point_count": len(seeds), "primary_eligible": clip["primary_eligible"], "crop": clip["crop"],
               "box_diagonal": diagonal, "frame_sha256": hashes, "arms": {}}
        for arm, video, roi in zip(ARMS, (full, low, high), (None, crop, crop), strict=True):
            arm_start = time.perf_counter()
            item, local = track(backend, video, seeds, times, ids, 30, SAMPLED, camera, frames[0].shape[:2], roi)
            stats = point_survival_and_cycle(item, diagonal, actor_count=len(ids))
            stats["sample_pass"] = sample_pass(stats)
            stats["primary_pass"] = bool(clip["primary_eligible"] and sample_pass(stats))
            stats["elapsed_seconds"] = time.perf_counter() - arm_start
            row["arms"][arm] = stats
            save_tracks(run / f"clip_{clip['selection_index']:02d}_{arm}.npz", item, local, seeds, roi, camera, frames[0].shape[:2])
            del item, local
        row["elapsed_seconds"] = time.perf_counter() - begin
        write(run / f"clip_{clip['selection_index']:02d}.json", row)
        rows.append(row)
        print(json.dumps({"center_complete": clip["selection_index"], "centers_done": len(rows), "total": 16,
                          "elapsed_seconds": time.perf_counter() - start}), flush=True)
        del frames, full, low, high
    from experiments.audit_okutama_tracking_scale_probe import audit, compare
    independent = audit(run)
    result = assess(rows)
    for key, expected in independent["mechanism_assessment"].items():
        compare(result[key], expected, f"final-mechanism/{key}")
    summary = {"status": "SCALE_MECHANISM_SUPPORTED_ONLY" if result["mechanism_supported"] else "NO_GO_SCALE_MECHANISM",
               **result, "synthetic": synthetic, "centers": 16, "learned_arm_evaluations": 48,
               "activity_fits": 0, "elapsed_seconds": time.perf_counter() - start,
               "peak_cuda_allocated_bytes": backend.torch.cuda.max_memory_allocated(), "predictor_calls": backend.calls,
               "audit_status": independent["status"], "audit_sha256": file_sha256(run / "independent_audit.json"),
               "execution_lock_sha256": file_sha256(run / "execution_lock.json"),
               "real_scope": "actor track visibility/cycle consistency, not physical correspondence ground truth or classification"}
    write(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == "prepare":
        require(not run.exists(), "prepare needs a fresh directory; existing evidence left untouched")
    else:
        require(not (run / "inference_started.json").exists(), "inference already started; existing evidence left untouched")
    try:
        (prepare if args.stage == "prepare" else execute)(run)
    except Exception as error:
        if run.exists() and not (run / "failure.json").exists():
            write(run / "failure.json", {"status": "INVALID_RUNTIME_OR_INTEGRITY_FAILURE", "error_type": type(error).__name__, "error": str(error),
                                        "activity_fits": 0, "no_automatic_retry": True})
        raise


if __name__ == "__main__":
    main()
