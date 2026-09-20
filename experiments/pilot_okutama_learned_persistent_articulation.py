"""Prepare or run the exclusive, conditional, label-blind 16-center tracker screen.

No download/install/training entry point. --prepare never loads model weights or
decodes images. --smoke requires its pinned prepared plan and independently
audited primary NO_GO artifact, and writes to a different fresh directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
from experiments import pilot_okutama_persistent_articulation as old  # noqa: E402
from hac.learned_persistent_tracking import (  # noqa: E402
    CycledTracks,
    LocalCoTracker3,
    file_sha256,
    independently_cycled_tracks,
    local_model_identity,
    point_survival_and_cycle,
)
from hac.persistent_articulation import actor_seed_points, background_seed_points  # noqa: E402
from hac.persistent_articulation_v2 import (  # noqa: E402
    TrackSequence,
    apply_similarity,
    common_held_endpoint_errors,
    decompose_tracks,
    fixed_point_split,
    identity_time_derivatives,
    similarity_design,
)
from hac.persistent_tracking import track_persistent, track_step  # noqa: E402

PROTOCOL = ROOT / "experiments/okutama_learned_persistent_articulation_protocol.json"
IDS = tuple(f"actor-{i:03d}" for i in range(64)) + tuple(f"background-{i:03d}" for i in range(64))
NAMES = ("learned_persistent", "lk_persistent", "pair_local_center_endpoint")
SAMPLED = tuple(value + 30 for value in old.OFFSETS)
SOURCE_FILES = [
    Path(__file__),
    PROTOCOL,
    ROOT / "src/hac/learned_persistent_tracking.py",
    ROOT / "src/hac/persistent_articulation_v2.py",
    ROOT / "src/hac/persistent_tracking.py",
    ROOT / "src/hac/persistent_articulation.py",
    Path(old.__file__),
    ROOT / "tests/test_learned_persistent_tracking.py",
    ROOT / "tests/test_persistent_articulation_v2.py",
    ROOT / "experiments/learned_tracker_cpu_dependencies.txt",
    ROOT / ".runs/research_20260919/learned_tracker_setup_v1/ENVIRONMENT.md",
]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(old.json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in SOURCE_FILES
    }


def validate_primary_no_go(path, *, artifact_root=ROOT):
    """Read only authorization/provenance fields, never activity arrays/features."""
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "CROSSING_EVENT_COMPLETE_NO_GO"
        or result.get("promotion_passed") is not False
        or result.get("gate_fits") != 45
    ):
        raise ValueError("Only a completed 45-fit primary NO_GO authorizes learned inference")
    run = path.parent
    audit_path, lock_path = run / "independent_audit.json", run / "execution_lock.json"
    if result.get("audit_sha256") != file_sha256(audit_path) or result.get(
        "execution_lock_sha256"
    ) != file_sha256(lock_path):
        raise ValueError("Primary summary audit/execution-lock hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "CROSSING_EVENT_INDEPENDENT_NO_FIT_AUDIT_PASS"
        or audit.get("gate_replays") != 45
    ):
        raise ValueError("Complete independent primary replay is required")
    for stem in ("execution_lock", "preflight", "ancestry_audit"):
        if audit.get(f"{stem}_sha256") != file_sha256(run / f"{stem}.json"):
            raise ValueError("Primary independent audit provenance changed")
    artifacts = audit.get("audited_artifacts", [])
    if len(artifacts) < 135:
        raise ValueError("Primary independent audit has incomplete output binding")
    for entry in artifacts:
        artifact = (artifact_root / entry["path"]).resolve()
        if not artifact.is_relative_to(artifact_root) or file_sha256(artifact) != entry["sha256"]:
            raise ValueError("Primary audited artifact changed")
    return {
        "summary": str(path),
        "summary_sha256": file_sha256(path),
        "audit_sha256": file_sha256(audit_path),
        "execution_lock_sha256": file_sha256(lock_path),
        "status": result["status"],
        "promotion_passed": False,
        "audited_gate_replays": 45,
        "use": "phase authorization only; not tracker/model inputs",
    }


def scaled_pixel_centers(points, scale):
    return (np.asarray(points, dtype=np.float64) + 0.5) * scale - 0.5


def _transform(dx=0.0, dy=0.0):
    return np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def synthetic_case(name, *, render=True):
    """Known identity correspondences: textured point sprites over moving texture.

    These are deliberately identifiable synthetic observations, not realistic
    action videos. Orthogonal nonrigid displacement cannot be absorbed by the
    root similarity on the fixed fitting identities.
    """
    if name not in (
        "camera_only",
        "actor_only",
        "same_direction",
        "opposite_direction",
        "nonrigid",
    ):
        raise ValueError("Unknown locked synthetic case")
    rng = np.random.default_rng(7319)
    actor0 = np.stack(np.meshgrid(np.linspace(122, 262, 8), np.linspace(58, 198, 8)), -1).reshape(
        -1, 2
    )
    grid = np.stack(np.meshgrid(np.linspace(15, 369, 24), np.linspace(15, 241, 16)), -1).reshape(
        -1, 2
    )
    outside = grid[(grid[:, 0] < 80) | (grid[:, 0] > 304)]
    background0 = outside[np.linspace(0, len(outside) - 1, 64).astype(int)]
    times = np.linspace(-1.0, 1.0, 9)
    split = fixed_point_split(IDS[:64])
    fit, _ = split.indices(IDS[:64])
    direction = rng.normal(size=actor0.shape)
    beta = np.linalg.lstsq(similarity_design(actor0[fit]), direction[fit].reshape(-1), rcond=None)[
        0
    ]
    direction -= (similarity_design(actor0) @ beta).reshape(-1, 2)
    direction /= np.max(np.abs(direction))
    camera_speed = np.asarray([0.0, 0.0] if name == "actor_only" else [8.0, -4.0])
    actor_speed = np.asarray(
        [0.0, 0.0]
        if name == "camera_only"
        else [-16.0, 0.0]
        if name == "opposite_direction"
        else [16.0, 0.0]
    )
    camera = np.stack([_transform(*(camera_speed * t)) for t in times])
    root = np.stack([_transform(*(actor_speed * t)) for t in times])
    art = np.stack(
        [
            5.0 * np.sin(t * np.pi / 2) * direction if name == "nonrigid" else np.zeros_like(actor0)
            for t in times
        ]
    )
    actor = np.stack(
        [
            apply_similarity(c, apply_similarity(r, actor0) + a)
            for c, r, a in zip(camera, root, art, strict=True)
        ]
    )
    background = np.stack([apply_similarity(c, background0) for c in camera])
    frames = []
    if render:
        texture = cv2.GaussianBlur(rng.integers(0, 180, (256, 384, 3), dtype=np.uint8), (3, 3), 0.6)
        sprites = rng.integers(30, 255, (64, 7, 7, 3), dtype=np.uint8)
        for c, points in zip(camera, actor, strict=True):
            frame = cv2.warpAffine(
                texture,
                c[:2],
                (384, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for sprite, (x, y) in zip(sprites, points, strict=True):
                left, top = int(np.floor(x)) - 6, int(np.floor(y)) - 6
                matrix = np.asarray(
                    [[1.0, 0.0, 3.0 + x - np.floor(x)], [0.0, 1.0, 3.0 + y - np.floor(y)]]
                )
                tile = cv2.warpAffine(sprite, matrix, (13, 13), flags=cv2.INTER_LINEAR)
                frame[top : top + 13, left : left + 13] = tile
            frames.append(frame)
    return {
        "frames": frames,
        "actor": actor,
        "background": background,
        "camera": camera,
        "root": root,
        "articulation": art,
        "times": times,
        "diagonal": math.hypot(160, 160),
    }


def geometry_from_sequence(sequence, center):
    actor = TrackSequence(
        sequence.positions[:, :64], sequence.valid[:, :64], sequence.times_seconds, IDS[:64]
    )
    bg = TrackSequence(
        sequence.positions[:, 64:], sequence.valid[:, 64:], sequence.times_seconds, IDS[64:]
    )
    return (
        actor,
        bg,
        decompose_tracks(
            actor,
            bg,
            center_index=center,
            actor_split=fixed_point_split(actor.point_ids),
            background_split=fixed_point_split(bg.point_ids),
        ),
    )


def coordinate_controls():
    cases = {}
    for name in ("camera_only", "actor_only", "same_direction", "opposite_direction", "nonrigid"):
        truth = synthetic_case(name, render=False)
        positions = np.concatenate((truth["actor"], truth["background"]), axis=1)
        seq = TrackSequence(positions, np.ones(positions.shape[:2], bool), truth["times"], IDS)
        actor, _, decomposition = geometry_from_sequence(seq, 4)
        recovered = decomposition.camera_compensated_actor
        expected = (
            np.stack([apply_similarity(r, truth["actor"][4]) for r in truth["root"]])
            + truth["articulation"]
        )
        errors = [
            np.max(np.abs(recovered - expected)) / truth["diagonal"],
            np.max(np.abs(decomposition.articulation - truth["articulation"])) / truth["diagonal"],
            np.max(np.abs(decomposition.camera_transforms - truth["camera"])) / truth["diagonal"],
        ]
        cases[name] = {
            "maximum_normalized_recovery_error": float(max(errors)),
            "pass": max(errors) <= 1e-6,
        }
    target = synthetic_case("same_direction", render=False)
    other = synthetic_case("opposite_direction", render=False)
    positions = np.concatenate((target["actor"], target["background"]), axis=1)
    seq = TrackSequence(positions, np.ones(positions.shape[:2], bool), target["times"], IDS)
    actor, _, correct = geometry_from_sequence(seq, 4)
    wrong_actor = np.stack(
        [
            apply_similarity(c, apply_similarity(r, actor.positions[4]))
            for c, r in zip(other["camera"], other["root"], strict=True)
        ]
    )
    fit, _ = fixed_point_split(IDS[:64]).indices(IDS[:64])
    permuted = positions.copy()
    for t in range(9):
        if t != 4:
            permuted[t, fit] = positions[t, np.roll(fit, 1)]
    _, _, wrong_id = geometry_from_sequence(
        TrackSequence(permuted, seq.valid, seq.times_seconds, IDS), 4
    )
    report = common_held_endpoint_errors(
        actor,
        {
            "correct": correct.root_actor_prediction,
            "wrong_actor": wrong_actor,
            "permuted": wrong_id.root_actor_prediction,
        },
        {
            "correct": correct.root_actor_prediction_valid,
            "wrong_actor": correct.root_actor_prediction_valid,
            "permuted": wrong_id.root_actor_prediction_valid,
        },
        split=fixed_point_split(actor.point_ids),
        center_index=4,
        actor_box_diagonal=target["diagonal"],
    )
    errors = report["median_error_normalized"]
    penalties = {
        key: (errors[key] - errors["correct"]) / max(errors["correct"], 0.005)
        for key in ("wrong_actor", "permuted")
    }
    return {
        "cases": cases,
        "specificity_penalties": penalties,
        "common_pairs": report["common_pairs"],
        "pass": all(item["pass"] for item in cases.values())
        and all(x >= 0.25 for x in penalties.values()),
        "scope": "direct-coordinate arithmetic only; not image-tracker validation",
    }


def prepare(args, run, protocol):
    if (
        file_sha256(old.SELECTION_PATH) != protocol["selection_sha256"]
        or file_sha256(old.CROP_MANIFEST_PATH) != protocol["crop_manifest_sha256"]
    ):
        raise RuntimeError("Locked selection/crop inputs changed")
    clips = old.load_selection(16)
    sources = {}
    for recording in sorted({c["recording"] for c in clips}):
        path = old._video_path(recording)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot inspect source metadata: {path}")
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        if shape != [3840, 2160] or not np.isclose(fps, old.FPS, rtol=1e-6):
            raise RuntimeError("Unexpected source dimensions/frame rate")
        needed = [
            c["center_frame"] + d for c in clips if c["recording"] == recording for d in (-30, 30)
        ]
        if min(needed) < 0 or max(needed) >= frames:
            raise RuntimeError("Dense source bridge would leave available frames")
        sources[recording] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "fps": fps,
            "frame_count": frames,
            "dimensions": shape,
            "source_integrity": "metadata pinned; decoded-frame hashes recorded on smoke, not a fresh full-video hash",
        }
    if bool(args.repository) != bool(args.checkpoint):
        raise ValueError("Repository and checkpoint must be supplied together")
    model = (
        local_model_identity(Path(args.repository), Path(args.checkpoint))
        if args.repository
        else None
    )
    if model is not None and (
        model["revision"] != protocol["learned"]["pinned_revision"]
        or model["checkpoint_sha256"] != protocol["learned"]["pinned_checkpoint_sha256"]
    ):
        raise RuntimeError("Local CoTracker weights/revision differ from the explicit official pin")
    controls = coordinate_controls()
    if not controls["pass"]:
        raise RuntimeError("Known-coordinate measurement controls failed")
    import torch

    ready = model is not None and torch.cuda.is_available()
    plan = {
        "status": "READY_FOR_CONDITIONAL_SMOKE" if ready else "PREPARED_NOT_RUNTIME_READY",
        "protocol": protocol,
        "source_hashes": source_hashes(),
        "clips": clips,
        "sources": sources,
        "selection_sha256": file_sha256(old.SELECTION_PATH),
        "crop_sha256": file_sha256(old.CROP_MANIFEST_PATH),
        "frame_manifest_sha256": file_sha256(old.P3_FRAME_MANIFEST),
        "model": model,
        "coordinate_controls": controls,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "inference_run": False,
        "activity_fits": 0,
        "prepared_unix": time.time(),
    }
    write_json(run / "prepared_plan.json", plan)
    print(json.dumps({"status": plan["status"], "plan": str(run / "prepared_plan.json")}))


def read_rgb_bridge(clip, metadata):
    path = Path(metadata["path"])
    if path.stat().st_size != metadata["bytes"] or path.stat().st_mtime_ns != metadata["mtime_ns"]:
        raise RuntimeError("Source file metadata changed since preparation")
    start = clip["center_frame"] - 30
    cap = cv2.VideoCapture(str(path))
    frames, stamps, hashes = [], [], []
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise RuntimeError("Source seek failed")
        for index in range(start, start + 61):
            ok, bgr = cap.read()
            if not ok or bgr.shape != (2160, 3840, 3):
                raise RuntimeError("Incomplete or malformed dense RGB bridge")
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
                raise RuntimeError("Decoder physical frame index disagrees")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            stamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            hashes.append(hashlib.sha256(rgb).hexdigest())
    finally:
        cap.release()
    stamps = np.asarray(stamps)
    if not np.isfinite(stamps).all() or not np.allclose(
        np.diff(stamps), 1 / metadata["fps"], rtol=0.005, atol=1e-5
    ):
        raise RuntimeError("Decoder timestamps are missing/non-CFR; no silent fps substitution")
    return frames, stamps - stamps[30], hashes


def lk_carriers(gray, seeds, times, max_error, spec):
    dense = track_persistent(gray, seeds, center_index=30, spec=spec, maximum_error=max_error)
    seq = TrackSequence(dense.positions, dense.valid, times, IDS)
    cycles = np.full((9, 128), np.nan)
    cycle_valid = np.zeros((9, 128), bool)
    for row, index in enumerate(SAMPLED):
        if index == 30:
            continue
        ids = np.flatnonzero(seq.valid[index])
        if not len(ids):
            continue
        # Independently propagate endpoint identities back along the full frame bridge.
        order = list(range(index, 29, -1)) if index > 30 else list(range(index, 31))
        reverse = track_persistent(
            [gray[t] for t in order],
            seq.positions[index, ids],
            center_index=0,
            spec=spec,
            maximum_error=max_error,
        )
        ok = reverse.valid[-1]
        cycles[row, ids[ok]] = np.linalg.norm(reverse.positions[-1, ok] - seeds[ids[ok]], axis=1)
        cycle_valid[row, ids[ok]] = True
    persistent = CycledTracks(seq, cycles, cycle_valid, SAMPLED, 0)
    positions = np.full((61, 128, 2), np.nan)
    valid = np.zeros((61, 128), bool)
    positions[30], valid[30] = seeds, True
    pair_cycles = np.full((9, 128), np.nan)
    pair_valid = np.zeros((9, 128), bool)
    for row, index in enumerate(SAMPLED):
        if index == 30:
            continue
        tracked = track_step(gray[30], gray[index], seeds, spec, maximum_error=max_error)
        positions[index], valid[index] = tracked.target, tracked.valid
        finite = np.isfinite(tracked.forward_backward_error)
        pair_cycles[row, finite], pair_valid[row, finite] = (
            tracked.forward_backward_error[finite],
            True,
        )
    pair = CycledTracks(
        TrackSequence(positions, valid, times, IDS), pair_cycles, pair_valid, SAMPLED, 0
    )
    return persistent, pair


def compare_carriers(carriers, diagonal):
    sampled, decomposed = {}, {}
    reference = next(iter(carriers.values())).sequence
    for name, item in carriers.items():
        seq = item.sequence
        if seq.point_ids != reference.point_ids or not np.array_equal(
            seq.times_seconds, reference.times_seconds
        ):
            raise ValueError("Carrier identity/time axes differ")
        if item.sampled_indices != SAMPLED:
            raise ValueError("Carrier sampled clock differs")
        take = np.asarray(SAMPLED)
        sampled[name] = TrackSequence(
            seq.positions[take], seq.valid[take], seq.times_seconds[take], seq.point_ids
        )
        _, _, decomposed[name] = geometry_from_sequence(sampled[name], 4)
    shared_reference_valid = np.logical_and.reduce(
        [item.valid[:, :64] for item in sampled.values()]
    )
    predictions = {key: value.root_actor_prediction for key, value in decomposed.items()}
    masks = {key: value.root_actor_prediction_valid for key, value in decomposed.items()}
    table = {}
    for name, seq in sampled.items():
        target = TrackSequence(
            seq.positions[:, :64], shared_reference_valid, seq.times_seconds, IDS[:64]
        )
        table[name] = common_held_endpoint_errors(
            target,
            predictions,
            masks,
            split=fixed_point_split(IDS[:64]),
            center_index=4,
            actor_box_diagonal=diagonal,
        )
    return table, sampled, decomposed


def score_synthetic_track(sequence, truth, *, nonrigid):
    actor, bg, dec = geometry_from_sequence(sequence, 4)
    _, held_actor = fixed_point_split(IDS[:64]).indices(IDS[:64])
    _, held_bg = fixed_point_split(IDS[64:]).indices(IDS[64:])
    amask = actor.valid[:, held_actor].copy()
    amask[4] = False
    bmask = bg.valid[:, held_bg] & dec.camera_background_valid[:, held_bg]
    bmask[4] = False
    aerror = (
        np.linalg.norm(actor.positions[:, held_actor] - truth["actor"][:, held_actor], axis=-1)
        / truth["diagonal"]
    )
    berror = (
        np.linalg.norm(
            dec.camera_background_prediction[:, held_bg] - truth["background"][:, held_bg], axis=-1
        )
        / truth["diagonal"]
    )
    expected_root = np.stack(
        [apply_similarity(r, truth["actor"][4]) - truth["actor"][4] for r in truth["root"]]
    )
    root_error = (
        np.linalg.norm(dec.root_displacement[:, held_actor] - expected_root[:, held_actor], axis=-1)
        / truth["diagonal"]
    )
    root_mask = amask & np.isfinite(root_error)
    amplitude = np.linalg.norm(expected_root[:, held_actor], axis=-1) / truth["diagonal"]
    true_high_motion = amplitude > 0.05
    true_high_motion[4] = False
    relative_mask = root_mask & true_high_motion
    relative = (
        float(np.median(root_error[relative_mask] / amplitude[relative_mask]))
        if relative_mask.any()
        else None
    )
    art_mask = amask & dec.articulation_valid[:, held_actor]
    true_art = truth["articulation"][:, held_actor]
    art_error = dec.articulation[:, held_actor] - true_art
    art_energy = float(np.sum(true_art[art_mask] ** 2))
    art_relative = (
        float(np.sqrt(np.sum(art_error[art_mask] ** 2) / art_energy))
        if art_energy > 1e-12
        else None
    )
    med_a = float(np.median(aerror[amask])) if amask.any() else None
    med_b = float(np.median(berror[bmask])) if bmask.any() else None
    med_root = float(np.median(root_error[root_mask])) if root_mask.any() else None
    actor_survival = float(np.mean(np.all(actor.valid, axis=0)))
    background_survival = float(np.mean(np.all(bg.valid, axis=0)))
    support = {
        "actor": float(amask.sum() / 256),
        "camera": float(bmask.sum() / 256),
        "root": float(root_mask.sum() / 256),
        "articulation": float(art_mask.sum() / 256),
    }
    high_motion_coverage = (
        float(relative_mask.sum() / true_high_motion.sum()) if true_high_motion.any() else None
    )
    passed = all(value is not None and value <= 0.01 for value in (med_a, med_b, med_root))
    passed &= (
        actor_survival >= 0.75 and background_survival >= 0.75 and min(support.values()) >= 0.75
    )
    passed &= not true_high_motion.any() or (
        high_motion_coverage >= 0.75 and relative is not None and relative <= 0.2
    )
    passed &= not nonrigid or (art_relative is not None and art_relative <= 0.5)
    return {
        "pass": bool(passed),
        "actor_endpoint_median": med_a,
        "camera_held_endpoint_median": med_b,
        "root_endpoint_median": med_root,
        "relative_root_error_above_0_05": relative,
        "nonrigid_known_truth_relative_rms": art_relative,
        "actor_survival_all_times": actor_survival,
        "background_survival_all_times": background_survival,
        "held_noncenter_support_fraction": support,
        "high_true_motion_support_fraction": high_motion_coverage,
        "actor_pairs": int(amask.sum()),
        "background_pairs": int(bmask.sum()),
        "physical_ground_truth": "synthetic rendered correspondences only",
    }


def run_synthetic_images(backend, protocol, run):
    result = {}
    for name in protocol["synthetic_image_controls"]["cases"]:
        truth = synthetic_case(name)
        seeds = np.concatenate((truth["actor"][4], truth["background"][4]))
        video = backend.prepare_video(truth["frames"])
        tracked = independently_cycled_tracks(
            lambda q, r, video=video: backend.predict(video, q, reverse=r),
            seeds,
            times_seconds=truth["times"],
            center_index=4,
            sampled_indices=tuple(range(9)),
            point_ids=IDS,
        )
        result[name] = score_synthetic_track(tracked.sequence, truth, nonrigid=name == "nonrigid")
        with (run / f"synthetic_{name}.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                positions=tracked.sequence.positions,
                valid=tracked.sequence.valid,
                cycle_error=tracked.cycle_error,
                actor_truth=truth["actor"],
                background_truth=truth["background"],
                times_seconds=truth["times"],
                point_ids=np.asarray(IDS),
            )
        write_json(run / f"synthetic_{name}.json", result[name])
        if not result[name]["pass"]:
            return {"pass": False, "cases": result, "stopped_on": name}
    return {"pass": True, "cases": result}


def smoke(args, run, protocol):
    if not args.prepared_plan or not args.primary_result:
        raise ValueError("--smoke requires --prepared-plan and --primary-result")
    authorization = validate_primary_no_go(args.primary_result)
    plan_path = Path(args.prepared_plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["protocol"] != protocol or plan["source_hashes"] != source_hashes():
        raise RuntimeError("Protocol/source changed after preparation")
    if plan["model"] is None or plan["status"] != "READY_FOR_CONDITIONAL_SMOKE":
        raise RuntimeError("Prepare again in the pinned CUDA environment with local model inputs")
    if (
        file_sha256(old.SELECTION_PATH) != plan["selection_sha256"]
        or file_sha256(old.CROP_MANIFEST_PATH) != plan["crop_sha256"]
        or file_sha256(old.P3_FRAME_MANIFEST) != plan["frame_manifest_sha256"]
    ):
        raise RuntimeError("Selection/box manifest changed")
    if old.load_selection(16) != plan["clips"]:
        raise RuntimeError("Prepared selected population differs")
    controls = coordinate_controls()
    if not controls["pass"]:
        raise RuntimeError("Known-coordinate preflight failed")
    write_json(
        run / "execution_lock.json",
        {
            "prepared_plan": str(plan_path),
            "prepared_plan_sha256": file_sha256(plan_path),
            "source_hashes": source_hashes(),
            "protocol": protocol,
            "primary_authorization": authorization,
            "activity_fits": 0,
        },
    )
    backend = LocalCoTracker3(plan["model"])
    backend.torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    synthetic = run_synthetic_images(backend, protocol, run)
    write_json(run / "synthetic_image_controls.json", synthetic)
    if not synthetic["pass"]:
        write_json(
            run / "summary.json",
            {
                "status": "NO_GO_SYNTHETIC_MEASUREMENT",
                "synthetic": synthetic,
                "real_centers_run": 0,
                "activity_fits": 0,
            },
        )
        return
    rows = []
    for clip in plan["clips"]:
        if time.perf_counter() - start > protocol["runtime_limits"]["maximum_wall_seconds"]:
            raise TimeoutError("Locked measurement wall-time limit reached; no policy substitution")
        clip_start = time.perf_counter()
        rgb, times, frame_hashes = read_rgb_bridge(clip, plan["sources"][clip["recording"]])
        native_gray = cv2.cvtColor(rgb[30], cv2.COLOR_RGB2GRAY)
        box = tuple(clip["native_box"])
        actor = actor_seed_points(native_gray, box, maximum=64).points
        exclusions = [
            tuple(3.0 * value for value in candidate) for candidate in clip["known_boxes_720"]
        ]
        background = background_seed_points(
            native_gray, box, maximum=64, exclude_boxes=exclusions
        ).points
        if len(actor) != 64 or len(background) != 64:
            row = {
                "sample_id": clip["sample_id"],
                "selection_index": clip["selection_index"],
                "status": "NO_GO_INSUFFICIENT_FIXED_SEEDS",
                "actor_count": len(actor),
                "background_count": len(background),
            }
            rows.append(row)
            write_json(run / f"clip_{clip['selection_index']:02d}.json", row)
            continue
        seeds_native = np.concatenate((actor, background))
        row = {
            "sample_id": clip["sample_id"],
            "selection_index": clip["selection_index"],
            "scenario": clip["scenario"],
            "sources": {},
            "frame_sha256": frame_hashes,
            "physical_seconds": times.tolist(),
            "status": "COMPLETE",
        }
        for source in protocol["sources"]:
            source_start = time.perf_counter()
            scale = 1.0 if source == "native4k" else 1 / 3
            frames = (
                rgb
                if scale == 1.0
                else [cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA) for f in rgb]
            )
            seeds = scaled_pixel_centers(seeds_native, scale)
            diagonal = math.hypot(box[2] - box[0], box[3] - box[1]) * scale
            video = backend.prepare_video(frames)
            learned = independently_cycled_tracks(
                lambda q, r, video=video: backend.predict(video, q, reverse=r),
                seeds,
                times_seconds=times,
                center_index=30,
                sampled_indices=SAMPLED,
                point_ids=IDS,
            )
            gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
            spec = {k: v for k, v in protocol["lk"].items() if k != "maximum_fb_native_pixels"}
            lk, pair = lk_carriers(
                gray, seeds, times, protocol["lk"]["maximum_fb_native_pixels"] * scale, spec
            )
            carriers = dict(zip(NAMES, (learned, lk, pair), strict=True))
            table, sampled, dec = compare_carriers(carriers, diagonal)
            row["sources"][source] = {
                "carriers": {
                    key: point_survival_and_cycle(value, diagonal)
                    for key, value in carriers.items()
                },
                "reference_sensitivity": table,
                "diagonal": diagonal,
                "elapsed_seconds": time.perf_counter() - source_start,
                "model_resolution": list(backend.model_shape),
            }
            arrays = {
                "point_ids": np.asarray(IDS),
                "times_seconds": times,
                "sampled_indices": np.asarray(SAMPLED),
                "seeds": seeds,
            }
            for name, item in carriers.items():
                velocity = identity_time_derivatives(
                    dec[name].articulation,
                    dec[name].articulation_valid,
                    sampled[name].times_seconds,
                )
                arrays.update(
                    {
                        f"{name}_positions": item.sequence.positions,
                        f"{name}_valid": item.sequence.valid,
                        f"{name}_cycle": item.cycle_error,
                        f"{name}_camera": dec[name].camera_transforms,
                        f"{name}_root": dec[name].root_transforms,
                        f"{name}_root_displacement": dec[name].root_displacement,
                        f"{name}_articulation": dec[name].articulation,
                        f"{name}_articulation_valid": dec[name].articulation_valid,
                        f"{name}_velocity": velocity.velocity,
                        f"{name}_velocity_valid": velocity.velocity_valid,
                    }
                )
            with (run / f"clip_{clip['selection_index']:02d}_{source}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            del video, frames, gray, carriers, learned, lk, pair, sampled, dec
        row["elapsed_seconds"] = time.perf_counter() - clip_start
        rows.append(row)
        write_json(run / f"clip_{clip['selection_index']:02d}.json", row)
        print(
            json.dumps(
                {
                    "clip_complete": clip["selection_index"],
                    "elapsed_seconds": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
        del rgb
    gates = {}
    for source in protocol["sources"]:
        metrics = [
            r["sources"][source]["carriers"]["learned_persistent"]
            for r in rows
            if r["status"] == "COMPLETE"
        ]
        survival_fraction = sum(m["actor_survival_all_nine"] >= 0.75 for m in metrics) / 16
        joint_fraction = (
            sum(m["actor_joint_forward_reverse_survival"] >= 0.75 for m in metrics) / 16
        )
        cycles = [
            m["cycle_median_normalized"]
            for m in metrics
            if m["cycle_median_normalized"] is not None
        ]
        median = float(np.median(cycles)) if cycles else None
        gates[source] = {
            "survival_fraction_of_all_16": survival_fraction,
            "joint_forward_reverse_fraction_of_all_16": joint_fraction,
            "cycle_median_of_center_medians": median,
            "pass": survival_fraction >= 0.8
            and joint_fraction >= 0.8
            and len(metrics) == 16
            and median is not None
            and median <= 0.05,
        }
    agreement = gates["native4k"]["pass"] == gates["downsample720"]["pass"]
    passed = all(g["pass"] for g in gates.values()) and agreement
    summary = {
        "status": "SMOKE_PASS_EXPANSION_REQUIRES_SEPARATE_PROTOCOL"
        if passed
        else "NO_GO_REAL_MEASUREMENT",
        "centers": len(rows),
        "gates": gates,
        "source_gate_direction_agreement": agreement,
        "synthetic": synthetic,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": backend.torch.cuda.max_memory_allocated(),
        "predictor_calls": backend.calls,
        "activity_fits": 0,
        "real_endpoint": "common-held inferred-track motion-prediction consistency, not physical ground truth",
        "expanded128_run": False,
        "classification_improvement_claimed": False,
    }
    write_json(run / "summary.json", summary)
    print(json.dumps(old.json_safe(summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repository")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prepared-plan")
    parser.add_argument("--primary-result")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    try:
        (prepare if args.prepare else smoke)(args, run, protocol)
    except Exception as error:
        write_json(
            run / "failure.json",
            {
                "status": "INVALID_OR_BLOCKED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "activity_fits": 0,
                "no_fallback_substitution": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
