"""Run the locked, label-blind Okutama CCAC measurement-feasibility pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import sys
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.ccac_motion import (  # noqa: E402
    CameraEstimate,
    PairAnalysis,
    _homography_has_safe_image_domain,
    analyze_actor_correspondence,
    analyze_pair,
    configure_opencv,
    estimate_camera,
)

STATUS = "OKUTAMA_CCAC_PILOT_COMPLETE"
WORKLOAD_STATUS = "OKUTAMA_CCAC_CLIP_WORKLOAD_COMPLETE"
PAIR_COUNT = 15


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Retained CCAC artifact cannot be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"CCAC artifact appeared during publication: {path}")
    temporary.replace(path)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Produce pickle-free NPZ bytes with sorted members and fixed ZIP metadata."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=6)
    return output.getvalue()


def _load_lock(root: Path, path: Path) -> tuple[dict[str, Any], Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    locker = importlib.import_module("tools.lock_okutama_ccac_pilot")
    return locker.validate_lock(root, path.resolve()), locker


def _pair_seed(domain: str, sample_id: str, pair_index: int) -> int:
    digest = hashlib.sha256((domain + sample_id + ":pair:" + str(pair_index)).encode()).hexdigest()
    return int(digest[:8], 16) >> 1


def _synthetic_checks(protocol: dict[str, Any]) -> dict[str, Any]:
    runtime = protocol["runtime"]
    observed_runtime = configure_opencv(
        threads=int(runtime["opencv_threads"]), opencl=bool(runtime["opencv_opencl"])
    )
    if observed_runtime != {
        "version": runtime["opencv_version"],
        "threads": runtime["opencv_threads"],
        "opencl": runtime["opencv_opencl"],
    }:
        raise RuntimeError("CCAC OpenCV runtime differs from the lock")
    camera_spec = protocol["camera_hypotheses"]
    gate = protocol["engineering_gates"]
    actor_spec = protocol["actor_correspondence"]
    decomposition = protocol["motion_decomposition"]

    xs = np.linspace(30, 1250, 12)
    ys = np.linspace(30, 690, 8)
    source = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    cells = np.asarray(
        [min(int(y * 3 / 720), 2) * 4 + min(int(x * 4 / 1280), 3) for x, y in source]
    )
    angle = np.deg2rad(1.25)
    scale = 1.005
    known = np.asarray(
        (
            (scale * np.cos(angle), -scale * np.sin(angle), 4.0),
            (scale * np.sin(angle), scale * np.cos(angle), -3.0),
            (0.0, 0.0, 1.0),
        )
    )
    homogeneous = np.column_stack((source, np.ones(len(source)))) @ known.T
    target = homogeneous[:, :2] / homogeneous[:, 2, None]
    camera = estimate_camera(
        source,
        target,
        cells,
        (720, 1280),
        seed=20260907,
        spec=camera_spec,
        gate=gate,
    )
    global_threshold = float(gate["synthetic_global_warp_max_median_residual_pixels"])
    global_pass = bool(
        camera.usable
        and camera.selected in {"H1", "H2"}
        and camera.metrics["camera_audit_median_pixels"] is not None
        and camera.metrics["camera_audit_median_pixels"] <= global_threshold
    )

    pole = np.asarray(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (-2 / 1280, 0.0, 1.0)))
    pole_rejected = not _homography_has_safe_image_domain(pole, (720, 1280))

    rng = np.random.default_rng(20260907)
    previous = np.full((360, 480), 24, dtype=np.uint8)
    current = previous.copy()
    actor_box = (160.0, 110.0, 260.0, 230.0)
    shift = 4
    patch = rng.integers(40, 256, (120, 100), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    previous[110:230, 160:260] = patch
    current[110:230, 164:264] = patch
    identity_camera = CameraEstimate(np.eye(3), "H0", True, {})
    actor_metrics, _ = analyze_actor_correspondence(
        previous,
        current,
        actor_box,
        (164.0, 110.0, 264.0, 230.0),
        identity_camera,
        elapsed_seconds=4 / 30,
        spec=actor_spec,
        decomposition=decomposition,
    )
    expected_translation = shift / (120 * (4 / 30))
    observed_translation = actor_metrics["compensated_translation_x_height_per_second"]
    articulation = actor_metrics["articulation_median_height_per_second"]
    rigid_pass = bool(
        actor_metrics["articulation_usable"]
        and observed_translation is not None
        and abs(observed_translation - expected_translation)
        <= float(gate["synthetic_rigid_actor_max_translation_error_height_per_second"])
        and articulation is not None
        and articulation <= float(gate["synthetic_rigid_actor_max_articulation_velocity"])
    )

    blank = np.zeros((120, 160), dtype=np.uint8)
    missing_metrics, _ = analyze_actor_correspondence(
        blank,
        blank,
        (30.0, 20.0, 80.0, 100.0),
        (30.0, 20.0, 80.0, 100.0),
        identity_camera,
        elapsed_seconds=4 / 30,
        spec=actor_spec,
        decomposition=decomposition,
    )
    missing_is_null = bool(
        not missing_metrics["translation_usable"]
        and missing_metrics["raw_translation_x_height_per_second"] is None
        and missing_metrics["compensated_translation_x_height_per_second"] is None
        and missing_metrics["articulation_median_height_per_second"] is None
    )
    checks = {
        "global_similarity_recovery": {
            "passed": global_pass,
            "selected_model": camera.selected,
            "audit_median_pixels": camera.metrics["camera_audit_median_pixels"],
            "threshold_pixels": global_threshold,
        },
        "projective_pole_rejection": {"passed": pole_rejected},
        "independent_rigid_actor_translation": {
            "passed": rigid_pass,
            "expected_x_height_per_second": expected_translation,
            "observed_x_height_per_second": observed_translation,
            "translation_absolute_error": (
                abs(observed_translation - expected_translation)
                if observed_translation is not None
                else None
            ),
            "articulation_median_height_per_second": articulation,
            "retained_points": actor_metrics["actor_primary_retained_points"],
        },
        "failed_measurement_is_null": {
            "passed": missing_is_null,
            "metrics": missing_metrics,
        },
    }
    return {
        "status": "OKUTAMA_CCAC_SYNTHETIC_CHECKS_COMPLETE",
        "passed": all(item["passed"] for item in checks.values()),
        "opencv": observed_runtime,
        "checks": checks,
    }


def _decode_frame(
    archive: zipfile.ZipFile,
    member_metadata: dict[str, dict[str, Any]],
    frame: dict[str, Any],
) -> np.ndarray | None:
    if not frame["valid_frame"]:
        return None
    member = frame["image_member"]
    if member is None or member not in member_metadata:
        raise RuntimeError("CCAC attempted to decode a member outside its lock")
    declared = member_metadata[member]
    info = archive.getinfo(member)
    if info.CRC != declared["crc32"] or info.file_size != declared["size_bytes"]:
        raise RuntimeError("CCAC ZIP metadata changed after locking")
    raw = archive.read(member)
    if len(raw) != declared["size_bytes"] or (zlib.crc32(raw) & 0xFFFFFFFF) != declared["crc32"]:
        raise RuntimeError("CCAC JPEG payload failed its locked member receipt")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.dtype != np.uint8 or image.shape != (720, 1280):
        raise RuntimeError("CCAC JPEG did not decode at locked native dimensions")
    return image


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    values = dict(metrics)
    models = values.pop("camera_models", {})
    reasons = values.get("camera_failure_reasons", [])
    values["camera_failure_reasons"] = ";".join(reasons)
    for model, model_metrics in sorted(models.items()):
        for name, value in model_metrics.items():
            values[f"camera_{model}_{name}"] = value
    return values


def _missing_pair_row(clip: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    previous, current = clip["frames"][index : index + 2]
    missing = []
    if not previous["valid_frame"]:
        missing.append("source_frame_unavailable")
    if not current["valid_frame"]:
        missing.append("target_frame_unavailable")
    return {
        "sample_id": clip["sample_id"],
        "selection_rank": clip["selection_rank"],
        "scenario": clip["scenario"],
        "size_band": clip["size_band"],
        "long_valid": clip["long_valid"],
        "pair_index": index,
        "source_time_index": index,
        "target_time_index": index + 1,
        "source_frame": previous["source_frame"],
        "target_frame": current["source_frame"],
        "elapsed_seconds": current["nominal_time_seconds"] - previous["nominal_time_seconds"],
        "pair_seed": seed,
        "pair_available": False,
        "pair_failure_reason": ";".join(missing),
        "camera_usable": False,
        "camera_failure_reasons": "pair_unavailable",
        "translation_usable": False,
        "translation_failure_reason": "pair_unavailable",
        "articulation_usable": False,
        "articulation_failure_reason": "pair_unavailable",
    }


def _overlay(
    clip: dict[str, Any],
    images: list[np.ndarray | None],
    analyses: dict[int, PairAnalysis],
    rows: list[dict[str, Any]],
) -> bytes:
    preferred = 7
    index = preferred if preferred in analyses else (min(analyses) if analyses else None)
    if index is None:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(canvas, "No available adjacent pair", (40, 80), 0, 1.2, (255, 255, 255), 2)
    else:
        current = images[index + 1]
        assert current is not None
        canvas = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        result = analyses[index]
        for source, target in zip(result.background_source, result.background_target, strict=True):
            cv2.arrowedLine(
                canvas,
                tuple(np.rint(source).astype(int)),
                tuple(np.rint(target).astype(int)),
                (255, 160, 0),
                1,
                tipLength=0.25,
            )
        for source, target in zip(result.actor_source, result.actor_target, strict=True):
            cv2.arrowedLine(
                canvas,
                tuple(np.rint(source).astype(int)),
                tuple(np.rint(target).astype(int)),
                (0, 255, 0),
                2,
                tipLength=0.3,
            )
        box = clip["frames"][index + 1]["bbox"]
        if box is not None:
            cv2.rectangle(
                canvas,
                (int(round(box[0])), int(round(box[1]))),
                (int(round(box[2])), int(round(box[3]))),
                (0, 0, 255),
                2,
            )
        row = rows[index]
        text = (
            f"pair={index} camera={row.get('camera_selected')} "
            f"camera_ok={row.get('camera_usable')} actor={row.get('actor_primary_retained_points', 0)}"
        )
        cv2.rectangle(canvas, (0, 0), (900, 42), (0, 0, 0), -1)
        cv2.putText(canvas, text, (12, 29), 0, 0.72, (255, 255, 255), 2)
    ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("CCAC overlay encoding failed")
    return encoded.tobytes()


def _compute_clip(
    clip: dict[str, Any],
    lock: dict[str, Any],
    archive: zipfile.ZipFile,
    *,
    include_overlay: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes | None]:
    protocol = lock["protocol"]
    domain = protocol["selection"]["hash_domain"]
    member_metadata = {row["image_member"]: row for row in lock["selected_image_members"]}
    images = [_decode_frame(archive, member_metadata, frame) for frame in clip["frames"]]
    boxes = [
        tuple(frame["bbox"]) if frame["bbox"] is not None else None for frame in clip["frames"]
    ]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    analyses: dict[int, PairAnalysis] = {}
    for index in range(PAIR_COUNT):
        seed = _pair_seed(domain, clip["sample_id"], index)
        previous_frame, current_frame = clip["frames"][index : index + 2]
        if (
            images[index] is None
            or images[index + 1] is None
            or boxes[index] is None
            or boxes[index + 1] is None
        ):
            row = _missing_pair_row(clip, index, seed)
            result = None
        else:
            elapsed = current_frame["nominal_time_seconds"] - previous_frame["nominal_time_seconds"]
            previous_member = previous_frame["image_member"]
            current_member = current_frame["image_member"]
            assert previous_member is not None and current_member is not None
            result = analyze_pair(
                images[index],
                images[index + 1],
                boxes[index],
                boxes[index + 1],
                [tuple(box) for box in lock["known_boxes_by_member"][previous_member]],
                [tuple(box) for box in lock["known_boxes_by_member"][current_member]],
                elapsed_seconds=elapsed,
                seed=seed,
                background_spec=protocol["background_correspondence"],
                actor_spec=protocol["actor_correspondence"],
                camera_spec=protocol["camera_hypotheses"],
                decomposition=protocol["motion_decomposition"],
                gate=protocol["engineering_gates"],
            )
            metrics = _flatten_metrics(result.metrics)
            row = {
                "sample_id": clip["sample_id"],
                "selection_rank": clip["selection_rank"],
                "scenario": clip["scenario"],
                "size_band": clip["size_band"],
                "long_valid": clip["long_valid"],
                "pair_index": index,
                "source_time_index": index,
                "target_time_index": index + 1,
                "source_frame": previous_frame["source_frame"],
                "target_frame": current_frame["source_frame"],
                "elapsed_seconds": elapsed,
                "pair_seed": seed,
                "pair_available": True,
                "pair_failure_reason": None,
                **metrics,
            }
            analyses[index] = result
        rows.append(row)
        prefix = f"pair_{index:02d}"
        arrays[f"{prefix}_background_source"] = (
            result.background_source if result is not None else np.empty((0, 2), np.float32)
        )
        arrays[f"{prefix}_background_target"] = (
            result.background_target if result is not None else np.empty((0, 2), np.float32)
        )
        arrays[f"{prefix}_background_cells"] = (
            result.background_cells if result is not None else np.empty(0, np.int16)
        )
        arrays[f"{prefix}_actor_source"] = (
            result.actor_source if result is not None else np.empty((0, 2), np.float32)
        )
        arrays[f"{prefix}_actor_target"] = (
            result.actor_target if result is not None else np.empty((0, 2), np.float32)
        )
        arrays[f"{prefix}_actor_fb_error"] = (
            result.actor_forward_backward_error if result is not None else np.empty(0, np.float32)
        )
        arrays[f"{prefix}_camera_transform"] = (
            result.camera_transform if result is not None else np.full((3, 3), np.nan)
        )
    from hac.ccac_motion import center_seeded_survival

    survival = center_seeded_survival(
        images,
        boxes,
        protocol["actor_correspondence"],
        center_slot=int(protocol["metadata_contract"]["center_slot"]),
    )
    camera_count = sum(bool(row["camera_usable"]) for row in rows)
    translation_count = sum(bool(row["translation_usable"]) for row in rows)
    articulation_count = sum(bool(row["articulation_usable"]) for row in rows)
    clip_metrics = {
        "sample_id": clip["sample_id"],
        "selection_rank": clip["selection_rank"],
        "scenario": clip["scenario"],
        "provider_recording_id": clip["provider_recording_id"],
        "provider_track_id": clip["provider_track_id"],
        "track_key": clip["track_key"],
        "center_frame": clip["center_frame"],
        "native_center_height": clip["native_center_height"],
        "size_band": clip["size_band"],
        "long_valid": clip["long_valid"],
        "any_known_occlusion": clip["any_known_occlusion"],
        "requested_pairs": PAIR_COUNT,
        "available_pairs": sum(bool(row["pair_available"]) for row in rows),
        "camera_usable_pairs": camera_count,
        "translation_usable_pairs": translation_count,
        "articulation_usable_pairs": articulation_count,
        "camera_clip_usable": camera_count >= 10,
        "translation_clip_usable": translation_count >= 10,
        "articulation_clip_usable": articulation_count >= 10,
        "jpeg_payloads_decoded": sum(image is not None for image in images),
        **survival,
    }
    overlay = _overlay(clip, images, analyses, rows) if include_overlay else None
    return rows, clip_metrics, deterministic_npz_bytes(arrays), overlay


def _clip_request(lock: dict[str, Any], clip: dict[str, Any], request_sha: str) -> dict[str, Any]:
    domain = lock["protocol"]["selection"]["hash_domain"]
    members = [
        frame["image_member"]
        for frame in clip["frames"]
        if frame["valid_frame"] and frame["image_member"]
    ]
    known = {member: lock["known_boxes_by_member"][member] for member in sorted(set(members))}
    return {
        "status": "OKUTAMA_CCAC_CLIP_REQUEST_BEFORE_PIXEL_ACCESS",
        "run_request_sha256": request_sha,
        "sample_id": clip["sample_id"],
        "selection_rank": clip["selection_rank"],
        "frames": clip["frames"],
        "pair_seeds": [_pair_seed(domain, clip["sample_id"], index) for index in range(PAIR_COUNT)],
        "known_boxes_sha256": canonical_digest(known),
        "overlay": clip["sample_id"] in lock["overlay_sample_ids"],
    }


def _validate_workload(
    directory: Path, request: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_path = directory / "request.json"
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != WORKLOAD_STATUS or receipt.get("request") != request:
        raise RuntimeError("Retained CCAC workload request/status changed")
    if not request_path.exists() or json.loads(request_path.read_text()) != request:
        raise RuntimeError("Retained CCAC pre-pixel clip request changed")
    expected = set(receipt.get("artifacts", {}))
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected | {"request.json", "receipt.json"}:
        raise RuntimeError("Retained CCAC workload inventory changed")
    for name, digest in receipt["artifacts"].items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError("Retained CCAC workload artifact changed")
    pair_payload = json.loads((directory / "pair_metrics.json").read_text())
    clip_metrics = json.loads((directory / "clip_metrics.json").read_text())
    if (
        pair_payload.get("sample_id") != request["sample_id"]
        or len(pair_payload.get("pairs", [])) != PAIR_COUNT
    ):
        raise RuntimeError("Retained CCAC pair metrics changed")
    return pair_payload["pairs"], clip_metrics


def _run_workload(
    output: Path,
    lock: dict[str, Any],
    clip: dict[str, Any],
    request_sha: str,
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    directory = output / "workloads" / f"{int(clip['selection_rank']):03d}"
    request = _clip_request(lock, clip, request_sha)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        pairs, metrics = _validate_workload(directory, request)
        return pairs, metrics, False
    request_path = directory / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Incomplete CCAC workload belongs to another request")
        extras = {path.name for path in directory.iterdir() if path.name != "request.json"}
        if extras:
            raise RuntimeError("Incomplete CCAC workload has partial artifacts")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("Nonempty CCAC workload lacks its pre-pixel request")
        atomic_json(request_path, request)
    started = time.perf_counter()
    pairs, metrics, correspondence_bytes, overlay = _compute_clip(
        clip,
        lock,
        archive,
        include_overlay=bool(request["overlay"]),
    )
    metrics = {**metrics, "workload_wall_time_seconds": time.perf_counter() - started}
    artifacts = {
        "pair_metrics.json": json_bytes({"sample_id": clip["sample_id"], "pairs": pairs}),
        "clip_metrics.json": json_bytes(metrics),
        "correspondences.npz": correspondence_bytes,
    }
    if overlay is not None:
        artifacts["overlay.png"] = overlay
    for name, raw in artifacts.items():
        atomic_bytes(directory / name, raw)
    receipt = {
        "status": WORKLOAD_STATUS,
        "request": request,
        "request_sha256": canonical_digest(request),
        "artifacts": {name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()},
        "jpeg_payloads_decoded": metrics["jpeg_payloads_decoded"],
        "requested_pairs": PAIR_COUNT,
    }
    atomic_json(receipt_path, receipt)
    return pairs, metrics, True


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("CCAC cannot publish an empty table")
    fields = sorted({key for row in rows for key in row})
    preferred = [
        name
        for name in (
            "sample_id",
            "selection_rank",
            "scenario",
            "size_band",
            "long_valid",
            "pair_index",
        )
        if name in fields
    ]
    fields = preferred + [name for name in fields if name not in preferred]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode()


def _fraction_rows(clips: list[dict[str, Any]], key: str, group_key: str) -> list[dict[str, Any]]:
    groups = sorted({str(row[group_key]) for row in clips})
    output = []
    for group in groups:
        subset = [row for row in clips if str(row[group_key]) == group]
        passed = sum(bool(row[key]) for row in subset)
        output.append(
            {
                group_key: group,
                "clips": len(subset),
                "articulation_usable_clips": passed,
                "articulation_usable_fraction": passed / len(subset),
                "translation_usable_clips": sum(
                    bool(row["translation_clip_usable"]) for row in subset
                ),
                "camera_usable_clips": sum(bool(row["camera_clip_usable"]) for row in subset),
            }
        )
    return output


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64
    )
    if not len(values):
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _failure_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in pairs:
        if row.get("pair_failure_reason"):
            for reason in str(row["pair_failure_reason"]).split(";"):
                counts[("pair", reason)] += 1
        if row.get("camera_failure_reasons"):
            for reason in str(row["camera_failure_reasons"]).split(";"):
                counts[("camera", reason)] += 1
        if row.get("translation_failure_reason"):
            counts[("translation", str(row["translation_failure_reason"]))] += 1
        if row.get("articulation_failure_reason"):
            counts[("articulation", str(row["articulation_failure_reason"]))] += 1
    return [
        {"stage": stage, "reason": reason, "pair_count": count}
        for (stage, reason), count in sorted(counts.items())
    ]


def _quality_plot(
    scenario_rows: list[dict[str, Any]],
    size_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(
        [row["scenario"] for row in scenario_rows],
        [row["articulation_usable_fraction"] for row in scenario_rows],
        color="#3B82F6",
    )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Clip feasibility by scenario", ylabel="Fraction", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=55)
    axes[1].bar(
        [row["size_band"] for row in size_rows],
        [row["articulation_usable_fraction"] for row in size_rows],
        color="#10B981",
    )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Clip feasibility by native size", ylabel="Fraction", ylim=(0, 1))
    residuals = [
        row["camera_audit_median_pixels"]
        for row in pairs
        if row.get("camera_audit_median_pixels") is not None
    ]
    axes[2].hist(residuals, bins=30, color="#F59E0B")
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set(title="Disjoint camera-audit residuals", xlabel="Median pixels")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, metadata={"Software": "HAC CCAC pilot"})
    plt.close(fig)
    return buffer.getvalue()


def _validate_output(output: Path, summary: dict[str, Any], request_sha: str) -> None:
    if summary.get("status") != STATUS or summary.get("request_sha256") != request_sha:
        raise RuntimeError("Retained CCAC summary status/request changed")
    for name, digest in summary.get("artifacts", {}).items():
        if not (output / name).is_file() or sha256_file(output / name) != digest:
            raise RuntimeError(f"Retained CCAC artifact changed: {name}")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = set(summary["artifacts"]) | {"request.json", "summary.json", "completion.json"}
    if observed != expected:
        raise RuntimeError("Retained CCAC aggregate inventory changed")
    completion = json.loads((output / "completion.json").read_text())
    if completion != {
        "status": "OKUTAMA_CCAC_PILOT_PUBLICATION_COMPLETE",
        "request_sha256": request_sha,
        "summary_sha256": sha256_file(output / "summary.json"),
    }:
        raise RuntimeError("Retained CCAC completion receipt changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    runs = (root / ".runs").resolve()
    if output == runs or not output.is_relative_to(runs):
        raise RuntimeError("CCAC output must be a dedicated directory below .runs")
    if args.max_new_clips is not None and not 1 <= args.max_new_clips <= 128:
        raise ValueError("max_new_clips must be between 1 and 128")
    lock, locker = _load_lock(root, args.protocol_lock)
    protocol = lock["protocol"]
    configured = configure_opencv(
        threads=int(protocol["runtime"]["opencv_threads"]),
        opencl=bool(protocol["runtime"]["opencv_opencl"]),
    )
    if (
        configured["version"] != lock["opencv"]["version"]
        or configured["threads"] != 1
        or configured["opencl"]
    ):
        raise RuntimeError("CCAC deterministic OpenCV configuration failed")
    request = {
        "status": "OKUTAMA_CCAC_PILOT_REQUEST_BEFORE_PIXEL_ACCESS",
        "lock_sha256": sha256_file(args.protocol_lock.resolve()),
        "protocol_sha256": lock["protocol_sha256"],
        "selected_sample_ids_sha256": lock["selected_sample_ids_sha256"],
        "selected_clip_records_sha256": lock["selected_clip_records_sha256"],
        "selected_image_members_sha256": lock["selected_image_members_sha256"],
        "known_boxes_sha256": lock["known_boxes_sha256"],
        "selected_clips": len(lock["selected_clips"]),
        "requested_pairs": len(lock["selected_clips"]) * PAIR_COUNT,
        "opencv": configured,
        "runner_sha256": sha256_file(Path(__file__)),
        "output": str(output),
        "action_labels_read": False,
        "prediction_rows_read": False,
        "classifier_fits": 0,
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("CCAC output belongs to another locked request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty CCAC output lacks its pre-pixel request")
    else:
        atomic_json(request_path, request)
    request_sha = canonical_digest(request)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_output(output, summary, request_sha)
        return summary
    aggregate_names = {
        "selection.csv",
        "pair_metrics.csv",
        "clip_metrics.csv",
        "scenario_metrics.csv",
        "size_metrics.csv",
        "failure_reasons.csv",
        "quality_overview.png",
    }
    if any((output / name).exists() for name in aggregate_names):
        raise RuntimeError("Incomplete CCAC aggregate evidence retained; refusing overwrite")
    synthetic = _synthetic_checks(protocol)
    synthetic_path = output / "synthetic_checks.json"
    if synthetic_path.exists():
        if json.loads(synthetic_path.read_text()) != synthetic:
            raise RuntimeError("Retained CCAC synthetic checks changed")
    else:
        atomic_json(synthetic_path, synthetic)
    if not synthetic["passed"]:
        raise RuntimeError("CCAC synthetic correctness gate failed before real-pixel access")

    all_pairs: list[dict[str, Any]] = []
    all_clips: list[dict[str, Any]] = []
    new_clips = 0
    archive_path = Path(lock["archive"]["path"])
    with zipfile.ZipFile(archive_path) as archive:
        for clip in lock["selected_clips"]:
            receipt = output / "workloads" / f"{int(clip['selection_rank']):03d}" / "receipt.json"
            if (
                args.max_new_clips is not None
                and new_clips >= args.max_new_clips
                and not receipt.exists()
            ):
                return {
                    "status": "OKUTAMA_CCAC_PILOT_BOUNDED_RUN_PAUSED",
                    "new_clips": new_clips,
                    "completed_clips": len(all_clips),
                }
            pairs, metrics, computed = _run_workload(output, lock, clip, request_sha, archive)
            all_pairs.extend(pairs)
            all_clips.append(metrics)
            new_clips += int(computed)
        replay_checks = []
        by_id = {clip["sample_id"]: clip for clip in lock["selected_clips"]}
        for sample_id in lock["replay_sample_ids"]:
            clip = by_id[sample_id]
            pairs, metrics, correspondence, overlay = _compute_clip(
                clip,
                lock,
                archive,
                include_overlay=sample_id in lock["overlay_sample_ids"],
            )
            directory = output / "workloads" / f"{int(clip['selection_rank']):03d}"
            retained_pairs = json.loads((directory / "pair_metrics.json").read_text())["pairs"]
            retained_metrics = json.loads((directory / "clip_metrics.json").read_text())
            retained_metrics.pop("workload_wall_time_seconds", None)
            correspondence_exact = hashlib.sha256(correspondence).hexdigest() == sha256_file(
                directory / "correspondences.npz"
            )
            overlay_exact = overlay is None or hashlib.sha256(overlay).hexdigest() == sha256_file(
                directory / "overlay.png"
            )
            replay_checks.append(
                {
                    "sample_id": sample_id,
                    "pair_metrics_exact": pairs == retained_pairs,
                    "clip_metrics_exact": metrics == retained_metrics,
                    "correspondence_bytes_exact": correspondence_exact,
                    "overlay_bytes_exact_if_applicable": overlay_exact,
                }
            )
    if len(all_pairs) != 128 * PAIR_COUNT or len(all_clips) != 128:
        raise RuntimeError("CCAC aggregate denominator changed")
    if not all(all(item.values()) for item in replay_checks):
        raise RuntimeError("CCAC exact replay check failed")

    scenario_rows = _fraction_rows(all_clips, "articulation_clip_usable", "scenario")
    size_rows = _fraction_rows(all_clips, "articulation_clip_usable", "size_band")
    failure_rows = _failure_rows(all_pairs)
    selection_rows = [
        {key: value for key, value in clip.items() if key != "frames"}
        for clip in lock["selected_clips"]
    ]
    aggregate_artifacts: dict[str, bytes] = {
        "selection.csv": _csv_bytes(selection_rows),
        "pair_metrics.csv": _csv_bytes(all_pairs),
        "clip_metrics.csv": _csv_bytes(all_clips),
        "scenario_metrics.csv": _csv_bytes(scenario_rows),
        "size_metrics.csv": _csv_bytes(size_rows),
        "failure_reasons.csv": _csv_bytes(failure_rows),
        "quality_overview.png": _quality_plot(scenario_rows, size_rows, all_pairs),
    }
    for name, raw in aggregate_artifacts.items():
        atomic_bytes(output / name, raw)

    gates = protocol["engineering_gates"]
    camera_pairs = sum(bool(row["camera_usable"]) for row in all_pairs)
    translation_pairs = sum(bool(row["translation_usable"]) for row in all_pairs)
    articulation_pairs = sum(bool(row["articulation_usable"]) for row in all_pairs)
    camera_fraction = camera_pairs / len(all_pairs)
    translation_clip_count = sum(bool(row["translation_clip_usable"]) for row in all_clips)
    articulation_clip_count = sum(bool(row["articulation_clip_usable"]) for row in all_clips)
    translation_clip_fraction = translation_clip_count / len(all_clips)
    articulation_clip_fraction = articulation_clip_count / len(all_clips)
    scenario_pass_count = sum(row["articulation_usable_fraction"] >= 0.5 for row in scenario_rows)
    size_pass = all(
        row["articulation_usable_fraction"]
        >= float(gates["minimum_clip_fraction_per_native_size_band"])
        for row in size_rows
    )
    gate_results = {
        "synthetic_correctness": synthetic["passed"],
        "exact_replay": all(all(item.values()) for item in replay_checks),
        "camera_pair_fraction": camera_fraction
        >= float(gates["camera_usable_pair_fraction_at_least"]),
        "overall_articulation_clip_fraction": articulation_clip_fraction
        >= float(gates["full_articulation_clip_fraction_at_least"]),
        "scenario_coverage": scenario_pass_count
        >= int(gates["minimum_scenarios_with_half_clips_usable"]),
        "native_size_band_coverage": size_pass,
    }
    full = all(gate_results.values())
    narrow = bool(
        not full
        and gate_results["camera_pair_fraction"]
        and (translation_clip_fraction >= 0.5 or articulation_clip_fraction >= 0.5)
    )
    decision = (
        "FULL_CCAC_EXTRACTION_AUTHORIZED"
        if full
        else (
            "BOUNDED_TRANSLATION_OR_TRACKER_REVIEW_AUTHORIZED"
            if narrow
            else "INITIAL_LK_EXTRACTOR_ABANDONED"
        )
    )
    summary: dict[str, Any] = {
        "status": STATUS,
        "study_id": protocol["study_id"],
        "decision": decision,
        "interpretation": (
            "This size-balanced, label-blind pilot measures engineering feasibility only; "
            "it is not a representative Okutama estimate and does not forecast action F1."
        ),
        "request_sha256": request_sha,
        "execution_lock_sha256": request["lock_sha256"],
        "cohort": lock["selection_summary"],
        "accounting": {
            "clips": len(all_clips),
            "requested_pairs": len(all_pairs),
            "available_pairs": sum(bool(row["pair_available"]) for row in all_pairs),
            "jpeg_payloads_decoded_primary": sum(
                int(row["jpeg_payloads_decoded"]) for row in all_clips
            ),
            "replay_clips": len(replay_checks),
            "replay_jpeg_payloads_decoded": sum(
                sum(
                    frame["valid_frame"]
                    for frame in next(
                        clip
                        for clip in lock["selected_clips"]
                        if clip["sample_id"] == item["sample_id"]
                    )["frames"]
                )
                for item in replay_checks
            ),
            "action_labels_read": 0,
            "prediction_rows_read": 0,
            "classifier_fits": 0,
            "protected_rows_read": 0,
        },
        "coverage": {
            "camera_usable_pairs": camera_pairs,
            "camera_usable_pair_fraction_all_requested": camera_fraction,
            "translation_usable_pairs": translation_pairs,
            "translation_usable_pair_fraction_all_requested": translation_pairs / len(all_pairs),
            "articulation_usable_pairs": articulation_pairs,
            "articulation_usable_pair_fraction_all_requested": articulation_pairs / len(all_pairs),
            "translation_usable_clips": translation_clip_count,
            "translation_usable_clip_fraction": translation_clip_fraction,
            "articulation_usable_clips": articulation_clip_count,
            "articulation_usable_clip_fraction": articulation_clip_fraction,
            "scenarios_at_or_above_half": scenario_pass_count,
            "scenario_rows": scenario_rows,
            "size_rows": size_rows,
        },
        "camera_model_counts_all_available_pairs": dict(
            sorted(
                Counter(
                    row.get("camera_selected") for row in all_pairs if row["pair_available"]
                ).items()
            )
        ),
        "distributions": {
            key: _distribution(all_pairs, key)
            for key in (
                "background_retained_points",
                "background_occupied_cells",
                "camera_audit_median_pixels",
                "camera_audit_p90_pixels",
                "camera_reduction_fraction",
                "actor_seed_points",
                "actor_primary_retained_points",
                "actor_vertical_regions",
                "actor_primary_fb_median_pixels",
                "actor_window_disagreement_median_pixels",
                "compensated_translation_norm_height_per_second",
                "articulation_median_height_per_second",
            )
        },
        "center_track_survival": {
            "clips_with_no_center_seeds": sum(row["center_seed_points"] == 0 for row in all_clips),
            "survival_fraction_distribution": _distribution(
                all_clips, "center_track_survival_fraction_75pct"
            ),
            "mean_survived_links_distribution": _distribution(
                all_clips, "center_track_mean_survived_links"
            ),
        },
        "failure_reasons": failure_rows,
        "synthetic_checks": synthetic,
        "exact_replay_checks": replay_checks,
        "gate_results": gate_results,
    }
    inventory = [output / "synthetic_checks.json", *(output / name for name in aggregate_artifacts)]
    inventory.extend(sorted(path for path in (output / "workloads").rglob("*") if path.is_file()))
    summary["artifacts"] = {
        path.relative_to(output).as_posix(): sha256_file(path) for path in inventory
    }
    atomic_json(summary_path, summary)
    atomic_json(
        output / "completion.json",
        {
            "status": "OKUTAMA_CCAC_PILOT_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": sha256_file(summary_path),
        },
    )
    _validate_output(output, summary, request_sha)
    locker.validate_lock(root, args.protocol_lock.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-clips", type=int)
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result.get("decision"),
                "completed_clips": result.get(
                    "completed_clips", result.get("accounting", {}).get("clips")
                ),
                "coverage": result.get("coverage"),
                "gate_results": result.get("gate_results"),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
