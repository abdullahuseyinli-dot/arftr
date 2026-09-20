"""Label-independent source-fidelity and two-view audit for authorized HAC centers.

Only the 21 retained development recordings may be decoded. The training video
archive may be downloaded opaquely; archive-wide decompression is never used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac.okutama_native_video import EXPECTED_RECORDINGS, EXPECTED_SCENARIOS
from hac.video_encoders import sha256_file

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / ".runs/research_20260907/okutama_native_video_p3"
SOURCE_DIR = Path(r"C:\Users\DELL\hac_external_data\OkutamaAction\source4k")
SOURCE_URL = (
    "https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/"
    "ADtwW9gmCdlhrvyaY-grV3A/TrainSetVideos.zip?dl=1&e=1&rlkey=7u7131amaul29amyr4jbnnu03"
)
SOURCE_BYTES = 14_985_760_603
OFFICIAL_PAGE = "https://raw.githubusercontent.com/miquelmarti/Okutama-Action/master/index.md"


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_allowed_manifest():
    lock = json.loads((P3 / "extraction_lock.json").read_text(encoding="utf-8"))
    for name in ("clip_index", "frame_manifest", "image_allowlist"):
        receipt = lock["manifest"]["artifacts"][name]
        path = ROOT / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"Original manifest hash changed: {name}")
    clips = read_csv(P3 / "manifest/clip_index.csv")
    if len(clips) != 4977 or {r["provider_recording_id"] for r in clips} != set(
        EXPECTED_RECORDINGS
    ):
        raise RuntimeError("Retained primary cohort changed")
    # Only label-independent fields leave this reader. No action values determine
    # selection, matching, image processing, or feature extraction.
    fields = (
        "sample_id",
        "recording_id",
        "provider_recording_id",
        "provider_track_id",
        "center_frame",
        "all_frames_valid",
    )
    clips = [{key: row[key] for key in fields} for row in clips]
    frames = defaultdict(list)
    for row in read_csv(P3 / "manifest/frame_manifest.csv"):
        if row["provider_recording_id"] not in EXPECTED_RECORDINGS:
            raise RuntimeError("Frame outside authorized recordings")
        frames[row["sample_id"]].append(row)
    for rows in frames.values():
        rows.sort(key=lambda row: int(row["time_index"]))
    allowlist = {r["image_member"] for r in read_csv(P3 / "manifest/image_allowlist.csv")}
    return lock, clips, dict(frames), allowlist


def select_pilot(clips: list[dict[str, str]], count: int = 128) -> list[dict[str, str]]:
    """Fixed hash order, round robin scenarios, complete long inputs only."""
    groups = defaultdict(list)
    for row in clips:
        if row["all_frames_valid"].lower() in {"1", "true"}:
            groups[row["recording_id"]].append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda r: hashlib.sha256(
                ("hac-source-dense-20260908:" + r["sample_id"]).encode()
            ).digest()
        )
    result = []
    depth = 0
    while len(result) < count:
        progressed = False
        for scenario in sorted(groups):
            if depth < len(groups[scenario]):
                result.append(groups[scenario][depth])
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            raise RuntimeError("Insufficient complete clips for fixed pilot")
        depth += 1
    return result


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lock, clips, frames, _ = load_allowed_manifest()
    selected = select_pilot(clips)
    write_json(output / "pilot_selection.json", selected)
    by_scenario = defaultdict(set)
    by_time = defaultdict(set)
    for row in clips:
        by_scenario[row["recording_id"]].add(row["provider_recording_id"])
        by_time[(row["recording_id"], int(row["center_frame"]))].add(row["provider_recording_id"])
    paired = {k: sorted(v) for k, v in by_scenario.items() if len(v) == 2}
    paired_time = {k for k, v in by_time.items() if len(v) == 2}
    probes = []
    for url, method in ((OFFICIAL_PAGE, "GET"), (SOURCE_URL, "HEAD")):
        try:
            request = urllib.request.Request(
                url, method=method, headers={"User-Agent": "HAC-source-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                item = {
                    "url": url,
                    "method": method,
                    "status": response.status,
                    "content_length": response.headers.get("Content-Length"),
                    "accept_ranges": response.headers.get("Accept-Ranges"),
                }
                if method == "GET":
                    body = response.read()
                    (output / "provider_index.md").write_bytes(body)
                    item["sha256"] = hashlib.sha256(body).hexdigest()
                probes.append(item)
        except Exception as exc:
            probes.append({"url": url, "method": method, "error": str(exc)})
    result = {
        "status": "SOURCE_AND_VIEW_METADATA_AUDIT_COMPLETE",
        "provider_repository": "https://github.com/miquelmarti/Okutama-Action",
        "license": "CC-BY-NC-SA-3.0; provider index source retained",
        "source_url": SOURCE_URL,
        "source_expected_bytes_observed_head": SOURCE_BYTES,
        "probes": probes,
        "allowed_recordings": sorted(EXPECTED_RECORDINGS),
        "allowed_scenarios": sorted(EXPECTED_SCENARIOS),
        "retained_rows": len(clips),
        "paired_scenarios": paired,
        "paired_scenario_rows": sum(r["recording_id"] in paired for r in clips),
        "cross_view_timestamp_keys": len(paired_time),
        "rows_with_other_view_timestamp": sum(
            (r["recording_id"], int(r["center_frame"])) in paired_time for r in clips
        ),
        "verified_cross_view_actor_matches": 0,
        "frame_sync_verified": False,
        "pilot_selection": {
            "count": len(selected),
            "rule": "round_robin_scenario_then_sha256_id;complete_long_only",
            "sha256": sha256_file(output / "pilot_selection.json"),
            "scenario_counts": dict(Counter(r["recording_id"] for r in selected)),
        },
        "archive_720p": lock["inputs"]["archive"],
        "image_payloads_decoded": 0,
        "action_values_used": 0,
        "models_fitted": 0,
        "protected_data_payloads_read": 0,
    }
    write_json(output / "summary.json", result)
    return result


def fidelity(output: Path, selection_path: Path, archive_path: Path) -> dict:
    """Decode fixed center frames and neighboring frame offsets only, CPU-only."""
    import cv2
    from PIL import Image

    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lock, _, frames_by_id, allowlist = load_allowed_manifest()
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    visual_ids = {row["sample_id"] for row in selected[:12]}
    if len(selected) != 128 or selected != select_pilot(load_allowed_manifest()[1]):
        raise RuntimeError("Fidelity pilot differs from predeclared label-independent selection")
    if archive_path.stat().st_size != SOURCE_BYTES:
        raise RuntimeError("4K training archive byte count is incomplete or changed")
    archive_hash = sha256_file(archive_path)
    started = time.perf_counter()
    rows = []
    video_receipts = []
    recording_rows = defaultdict(list)
    for row in selected:
        recording_rows[row["provider_recording_id"]].append(row)
    extraction_dir = SOURCE_DIR / "allowed_videos"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(archive_path) as source,
        zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as old,
    ):
        # Only directory metadata is enumerated. Do not read annotation or other
        # recording payloads and do not call extractall/testzip.
        members = defaultdict(list)
        for info in source.infolist():
            basename = Path(info.filename).name
            match = re.fullmatch(r"(\d+\.\d+\.\d+)\.(?:mov|mp4|avi|mkv)", basename, re.I)
            if match and match[1] in EXPECTED_RECORDINGS:
                members[match[1]].append(info)
        for recording, choices in sorted(recording_rows.items()):
            if len(members[recording]) != 1:
                raise RuntimeError(f"Expected exactly one allowed video member: {recording}")
            info = members[recording][0]
            path = extraction_dir / Path(info.filename).name
            if not path.exists():
                with source.open(info) as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if path.stat().st_size != info.file_size:
                raise RuntimeError("Existing extracted video size mismatch")
            digest, crc = hashlib.sha256(), 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
            if crc != info.CRC:
                raise RuntimeError("Extracted video does not match archive member CRC")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open allowed video {recording}")
            width, height = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (width, height) != (3840, 2160):
                raise RuntimeError(f"Source is not genuine 3840x2160: {recording}")
            video_receipts.append(
                {
                    "recording": recording,
                    "zip_member": info.filename,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                    "sha256": digest.hexdigest(),
                    "size": [width, height],
                    "fps": fps,
                    "frame_count": frame_count,
                }
            )
            for clip in choices:
                frame_row = frames_by_id[clip["sample_id"]][8]
                member = frame_row["image_member"]
                if member not in allowlist:
                    raise RuntimeError("Center JPEG outside exact allowed frame manifest")
                jpeg = old.read(member)
                with Image.open(io.BytesIO(jpeg)) as im:
                    rgb720 = np.asarray(im.convert("RGB"))
                center = int(frame_row["source_frame"])
                scores = []
                targets = {}
                # Offsets audit provider zero/one-based indexing and nearby timing.
                cap.set(cv2.CAP_PROP_POS_FRAMES, center - 2)
                for offset in (-2, -1, 0, 1, 2):
                    success, bgr = cap.read()
                    if not success:
                        continue
                    rgb4k = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    down = cv2.resize(rgb4k, (1280, 720), interpolation=cv2.INTER_AREA)
                    mae = float(np.abs(down.astype(np.float32) - rgb720).mean())
                    scores.append(
                        {
                            "offset": offset,
                            "mae720": mae,
                            "pts_ms_opencv": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                        }
                    )
                    targets[offset] = rgb4k
                if not scores:
                    raise RuntimeError("No decoded source candidates")
                best = min(scores, key=lambda value: value["mae720"])
                rgb4k = targets[best["offset"]]
                box = [
                    float(frame_row[k])
                    for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                ]
                x0, y0, x1, y1 = [int(round(3 * v)) for v in box]
                x0, x1 = max(0, x0), min(3840, x1)
                y0, y1 = max(0, y0), min(2160, y1)
                crop4k = rgb4k[y0:y1, x0:x1]
                if not crop4k.size:
                    raise RuntimeError("Empty allowed actor crop")
                degraded = cv2.resize(
                    cv2.resize(
                        crop4k,
                        (max(1, crop4k.shape[1] // 3), max(1, crop4k.shape[0] // 3)),
                        interpolation=cv2.INTER_AREA,
                    ),
                    (crop4k.shape[1], crop4k.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                gray4 = cv2.cvtColor(crop4k, cv2.COLOR_RGB2GRAY)
                graylow = cv2.cvtColor(degraded, cv2.COLOR_RGB2GRAY)
                # Record measurable high-frequency content, not a semantic claim.
                detail = float(
                    np.mean((gray4.astype(np.float32) - graylow.astype(np.float32)) ** 2)
                )
                if clip["sample_id"] in visual_ids:
                    from PIL import ImageDraw, ImageOps

                    panel = Image.new("RGB", (768, 430), (245, 245, 245))
                    for column, pixels in enumerate((crop4k, degraded)):
                        tile = ImageOps.pad(
                            Image.fromarray(pixels),
                            (384, 384),
                            method=Image.Resampling.NEAREST,
                            color=(124, 116, 104),
                        )
                        panel.paste(tile, (384 * column, 46))
                    draw = ImageDraw.Draw(panel)
                    draw.text((8, 5), clip["sample_id"], fill=(0, 0, 0))
                    draw.text((8, 25), "Native 4K actor pixels", fill=(0, 0, 0))
                    draw.text((392, 25), "Exact 3x downsample + upsample control", fill=(0, 0, 0))
                    panel.save(output / f"{clip['sample_id']}_detail.png")
                rows.append(
                    {
                        "sample_id": clip["sample_id"],
                        "recording": recording,
                        "center_frame": center,
                        "source_frame_offset_best": best["offset"],
                        "candidate_alignment": scores,
                        "actor_size_4k": [crop4k.shape[1], crop4k.shape[0]],
                        "detail_vs_exact_downsample_mse": detail,
                        "source_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                    }
                )
            cap.release()
            print(
                json.dumps({"fidelity_recording": recording, "completed_centers": len(rows)}),
                flush=True,
            )
    result = {
        "status": "FOUR_K_FIXED_CENTER_FIDELITY_AUDIT_COMPLETE",
        "centers": len(rows),
        "source_archive": {
            "path": str(archive_path),
            "bytes": SOURCE_BYTES,
            "sha256": archive_hash,
            "url": SOURCE_URL,
            "digest_authority": "observed_acquisition",
        },
        "selection_sha256": sha256_file(selection_path),
        "videos": video_receipts,
        "best_offset_counts": dict(Counter(str(r["source_frame_offset_best"]) for r in rows)),
        "median_detail_vs_exact_downsample_mse": float(
            np.median([r["detail_vs_exact_downsample_mse"] for r in rows])
        ),
        "median_best_frame_mae720": float(
            np.median([min(c["mae720"] for c in r["candidate_alignment"]) for r in rows])
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "decoded_recordings": sorted(recording_rows),
        "protected_data_payloads_read": 0,
        "annotation_payloads_read": 0,
        "action_values_used": 0,
        "models_fitted": 0,
        "claim_scope": "128 center images; no classification, semantic detail, cross-view sync or full-clip timing claim",
    }
    write_json(output / "summary.json", result)
    return result


def paired_views(output: Path) -> dict:
    """Thirty fixed scene pairs; geometry proposals, never actor truth."""
    import cv2
    from PIL import Image
    from scipy.optimize import linear_sum_assignment

    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lock, clips, frames, allowlist = load_allowed_manifest()
    by_key = defaultdict(lambda: defaultdict(list))
    for row in clips:
        by_key[(row["recording_id"], int(row["center_frame"]))][
            row["provider_recording_id"]
        ].append(row)
    groups = defaultdict(list)
    for key, views in by_key.items():
        if len(views) == 2:
            groups[key[0]].append(key)
    selection = []
    for scenario in sorted(groups):
        keys = sorted(
            groups[scenario],
            key=lambda key: hashlib.sha256(f"hac-paired-source-20260908:{key}".encode()).digest(),
        )
        for key in keys[:3]:
            selection.append(
                {"scenario": scenario, "frame": key[1], "recordings": sorted(by_key[key])}
            )
    write_json(output / "selection_before_image_access.json", selection)
    results = []
    repeated = Counter()
    with zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as archive:
        for choice in selection:
            scenario, frame = choice["scenario"], choice["frame"]
            views = by_key[(scenario, frame)]
            images, boxes, ids, masks = [], [], [], []
            receipt = {
                **choice,
                "image_members": [],
                "image_sha256": [],
                "verified_actor_matches": 0,
            }
            for recording in choice["recordings"]:
                rows = sorted(views[recording], key=lambda row: int(row["provider_track_id"]))
                center = frames[rows[0]["sample_id"]][8]
                member = center["image_member"]
                if member not in allowlist or recording not in EXPECTED_RECORDINGS:
                    raise RuntimeError("Paired-view pixel access outside primary allowlist")
                raw = archive.read(member)
                with Image.open(io.BytesIO(raw)) as im:
                    rgb = np.asarray(im.convert("RGB"))
                view_boxes = np.asarray(
                    [
                        [
                            float(frames[r["sample_id"]][8][k])
                            for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
                        ]
                        for r in rows
                    ]
                )
                mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
                for box in view_boxes:
                    x0, y0, x1, y1 = np.rint(box).astype(int)
                    cv2.rectangle(
                        mask,
                        (max(0, x0 - 5), max(0, y0 - 5)),
                        (min(1279, x1 + 5), min(719, y1 + 5)),
                        0,
                        -1,
                    )
                images.append(rgb)
                masks.append(mask)
                boxes.append(view_boxes)
                ids.append([r["provider_track_id"] for r in rows])
                receipt["image_members"].append(member)
                receipt["image_sha256"].append(hashlib.sha256(raw).hexdigest())
            orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
            detected = [
                orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), mask)
                for img, mask in zip(images, masks, strict=True)
            ]
            receipt["feature_counts"] = [len(item[0]) for item in detected]
            if any(item[1] is None for item in detected):
                receipt["status"] = "INSUFFICIENT_SCENE_FEATURES"
                results.append(receipt)
                continue
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(detected[0][1], detected[1][1], k=2)
            good = [
                m
                for pair in pairs
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            receipt["ratio_matches"] = len(good)
            if len(good) < 24:
                receipt["status"] = "INSUFFICIENT_CROSS_VIEW_SCENE_MATCHES"
                results.append(receipt)
                continue
            xy0 = np.asarray([detected[0][0][m.queryIdx].pt for m in good], dtype=np.float32)
            xy1 = np.asarray([detected[1][0][m.trainIdx].pt for m in good], dtype=np.float32)
            fit = np.arange(len(good)) % 3 != 0
            H, inliers = cv2.findHomography(
                xy0[fit], xy1[fit], cv2.USAC_MAGSAC, 4.0, maxIters=10000, confidence=0.999
            )
            if H is None or inliers is None:
                receipt["status"] = "CAMERA_HOMOGRAPHY_FAILED"
                results.append(receipt)
                continue
            projected_audit = cv2.perspectiveTransform(xy0[~fit, None, :], H)[:, 0, :]
            audit_error = np.linalg.norm(projected_audit - xy1[~fit], axis=1)
            receipt.update(
                {
                    "fit_inliers": int(inliers.sum()),
                    "fit_count": int(fit.sum()),
                    "audit_count": int((~fit).sum()),
                    "audit_inlier_fraction_4px": float((audit_error < 4).mean()),
                    "audit_median_error_px": float(np.median(audit_error)),
                    "homography": H.tolist(),
                }
            )
            # A homography models a plane, not the whole articulated actor. Feet
            # are the least-bad image proxy for ground-plane position.
            foot0 = np.column_stack(((boxes[0][:, 0] + boxes[0][:, 2]) / 2, boxes[0][:, 3])).astype(
                np.float32
            )
            foot1 = np.column_stack(((boxes[1][:, 0] + boxes[1][:, 2]) / 2, boxes[1][:, 3])).astype(
                np.float32
            )
            projected = cv2.perspectiveTransform(foot0[:, None, :], H)[:, 0, :]
            distances = np.linalg.norm(projected[:, None, :] - foot1[None, :, :], axis=2)
            normalized = distances / np.maximum(8, boxes[1][:, 3] - boxes[1][:, 1])[None, :]
            candidates = []
            ii, jj = linear_sum_assignment(normalized)
            scene_ok = int(inliers.sum()) >= 12 and receipt["audit_inlier_fraction_4px"] >= 0.5
            for i, j in zip(ii, jj, strict=True):
                if scene_ok and normalized[i, j] <= 1.5:
                    item = {
                        "track_view0": ids[0][i],
                        "track_view1": ids[1][j],
                        "footpoint_distance_px": float(distances[i, j]),
                        "distance_in_target_actor_heights": float(normalized[i, j]),
                        "status": "UNVERIFIED_GEOMETRY_PROPOSAL",
                    }
                    candidates.append(item)
                    repeated[(scenario, ids[0][i], ids[1][j])] += 1
                    from PIL import ImageDraw, ImageOps

                    panel = Image.new("RGB", (512, 296), (245, 245, 245))
                    for view, actor in enumerate((i, j)):
                        x0, y0, x1, y1 = boxes[view][actor]
                        margin = (y1 - y0) * 0.15
                        bounds = (
                            max(0, int(x0 - margin)),
                            max(0, int(y0 - margin)),
                            min(1280, int(x1 + margin)),
                            min(720, int(y1 + margin)),
                        )
                        crop = Image.fromarray(images[view]).crop(bounds)
                        panel.paste(
                            ImageOps.pad(
                                crop,
                                (256, 256),
                                method=Image.Resampling.NEAREST,
                                color=(124, 116, 104),
                            ),
                            (view * 256, 40),
                        )
                    draw = ImageDraw.Draw(panel)
                    draw.text(
                        (8, 5), f"UNVERIFIED: scenario {scenario}, frame {frame}", fill=(0, 0, 0)
                    )
                    draw.text((8, 22), f"View0 track {ids[0][i]}", fill=(0, 0, 0))
                    draw.text((264, 22), f"View1 track {ids[1][j]}", fill=(0, 0, 0))
                    panel.save(output / f"proposal_{scenario}_{frame}_{ids[0][i]}_{ids[1][j]}.png")
            receipt["proposals"] = candidates
            receipt["status"] = (
                "GEOMETRY_PROPOSALS_ONLY" if scene_ok else "HELD_SCENE_MATCH_QUALITY_FAILED"
            )
            results.append(receipt)
    result = {
        "status": "FIXED_PAIRED_VIEW_GEOMETRY_PILOT_COMPLETE",
        "scene_pairs": len(selection),
        "selection_sha256": sha256_file(output / "selection_before_image_access.json"),
        "scenarios": len(groups),
        "rows": results,
        "status_counts": dict(Counter(row["status"] for row in results)),
        "unverified_geometry_proposals": sum(len(row.get("proposals", [])) for row in results),
        "repeated_proposals": [
            {"scenario": key[0], "track_view0": key[1], "track_view1": key[2], "times": count}
            for key, count in sorted(repeated.items())
            if count >= 2
        ],
        "verified_actor_matches": 0,
        "frame_sync_verified": False,
        "action_values_used": 0,
        "annotation_payloads_read": 0,
        "image_decodes": 2 * len(selection),
        "protected_data_payloads_read": 0,
        "interpretation": "Same frame indices remain a timing hypothesis. Homography proposals require actor identity and timing verification before training.",
    }
    write_json(output / "summary.json", result)
    return result


def alignment_search(output: Path, selection_path: Path, fidelity_path: Path) -> dict:
    """Fit only label-blind pixel timebases; validate on disjoint physical frames."""
    import cv2
    from PIL import Image

    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lock, clips, frames_by_id, allowlist = load_allowed_manifest()
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if selected != select_pilot(clips):
        raise RuntimeError("Alignment selection changed from fixed 128-center pilot")
    fidelity_receipt = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if fidelity_receipt["selection_sha256"] != sha256_file(selection_path):
        raise RuntimeError("Fidelity selection mismatch")
    suspect = {
        row["recording"]
        for row in fidelity_receipt["rows"]
        if row["source_frame_offset_best"] != 0
        or min(c["mae720"] for c in row["candidate_alignment"]) > 6
    }
    queries = {}
    for index, clip in enumerate(selected):
        key = (clip["provider_recording_id"], int(clip["center_frame"]))
        if key not in queries:
            queries[key] = dict(
                clip, split="fit" if index < 64 else "validation", supplemental=False
            )
    # Supply otherwise absent views without moving any original query between splits.
    # Selection uses only IDs/physical times and is committed before image access.
    for recording in sorted(EXPECTED_RECORDINGS):
        candidates = sorted(
            (
                c
                for c in clips
                if c["provider_recording_id"] == recording
                and c["all_frames_valid"].lower() in {"1", "true"}
            ),
            key=lambda c: hashlib.sha256(
                ("hac-alignment-support-20260908:" + c["sample_id"]).encode()
            ).digest(),
        )
        for split in ("fit", "validation"):
            while sum(k[0] == recording and q["split"] == split for k, q in queries.items()) < 2:
                clip = next(
                    (c for c in candidates if (recording, int(c["center_frame"])) not in queries),
                    None,
                )
                if clip is None:
                    raise RuntimeError(f"Insufficient disjoint support: {recording}")
                queries[(recording, int(clip["center_frame"]))] = dict(
                    clip, split=split, supplemental=True
                )
    selection = list(queries.values())
    write_json(output / "selection_before_image_access.json", selection)
    video_receipts = {v["recording"]: v for v in fidelity_receipt["videos"]}
    rows, mappings = [], []
    started = time.perf_counter()
    with zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as old:
        for recording in sorted(EXPECTED_RECORDINGS):
            video_receipt = video_receipts[recording]
            path = SOURCE_DIR / "allowed_videos" / Path(video_receipt["zip_member"]).name
            if sha256_file(path) != video_receipt["sha256"]:
                raise RuntimeError(f"Allowed video digest mismatch: {recording}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {recording}")
            radius = 120 if recording in suspect else 2
            recording_results = []
            for (query_recording, center), clip in queries.items():
                if query_recording != recording:
                    continue
                frame = frames_by_id[clip["sample_id"]][8]
                if int(frame["source_frame"]) != center or frame["image_member"] not in allowlist:
                    raise RuntimeError("Physical center or image allowlist mismatch")
                jpeg = old.read(frame["image_member"])
                with Image.open(io.BytesIO(jpeg)) as im:
                    reference = cv2.resize(
                        np.asarray(im.convert("RGB")), (320, 180), interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
                first = max(0, center - radius)
                last = min(video_receipt["frame_count"] - 1, center + radius)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                scores = []
                first_undecodable_index = None
                for source_index in range(first, last + 1):
                    success, bgr = cap.read()
                    if not success:
                        first_undecodable_index = source_index
                        break
                    small = cv2.cvtColor(
                        cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
                    )
                    scores.append(
                        {
                            "source_index": source_index,
                            "mae320": float(np.abs(small.astype(np.float32) - reference).mean()),
                            "pts_ms_opencv": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                            "reported_next_frame_opencv": float(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        }
                    )
                if not scores:
                    raise RuntimeError(f"No candidate decoded: {recording}/{center}")
                best = min(scores, key=lambda score: score["mae320"])
                result = {
                    "sample_id": clip["sample_id"],
                    "recording": recording,
                    "center_frame": center,
                    "split": clip["split"],
                    "supplemental": clip["supplemental"],
                    "radius": radius,
                    "source_jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "best": best,
                    "best_offset": best["source_index"] - center,
                    "best_at_search_boundary": best["source_index"] in {first, last},
                    "candidates": scores,
                    "first_undecodable_index": first_undecodable_index,
                }
                rows.append(result)
                recording_results.append(result)
                write_json(output / "queries" / f"{recording}_{center}.json", result)
            cap.release()
            fit = [r for r in recording_results if r["split"] == "fit"]
            validation = [r for r in recording_results if r["split"] == "validation"]
            lookup = {
                r["sample_id"]: {s["source_index"]: s for s in r["candidates"]}
                for r in recording_results
            }
            model_scores = []
            for scale in (1.0, 1000 / 1001):
                for offset in range(-radius, radius + 1):
                    predicted = [
                        lookup[r["sample_id"]].get(round(scale * r["center_frame"] + offset))
                        for r in fit
                    ]
                    if all(p is not None for p in predicted):
                        model_scores.append(
                            {
                                "scale": scale,
                                "offset": offset,
                                "fit_mae320": float(np.mean([p["mae320"] for p in predicted])),
                            }
                        )
            model_scores.sort(key=lambda m: (m["fit_mae320"], m["scale"] != 1.0, abs(m["offset"])))
            model = model_scores[0]
            checks = []
            for r in validation:
                index = round(model["scale"] * r["center_frame"] + model["offset"])
                score = lookup[r["sample_id"]].get(index)
                checks.append(
                    {
                        "sample_id": r["sample_id"],
                        "source_index": index,
                        "mae320": None if score is None else score["mae320"],
                        "excess_mae320_over_search_best": None
                        if score is None
                        else score["mae320"] - r["best"]["mae320"],
                        "pass": score is not None
                        and score["mae320"] < 6.0
                        and score["mae320"] - r["best"]["mae320"] < 0.25,
                    }
                )
            mapping = {
                "recording": recording,
                "selected_model": model,
                "fit_frames": len(fit),
                "validation_frames": len(validation),
                "validation_checks": checks,
                "best_model_per_timebase": [
                    next(m for m in model_scores if m["scale"] == scale)
                    for scale in (1.0, 1000 / 1001)
                ],
                "validated": all(c["pass"] for c in checks),
                "scope": "observed disjoint center frames only; every requested feature input still requires pixel verification",
            }
            mappings.append(mapping)
            write_json(
                output / f"recording_{recording}.json",
                {"mapping": mapping, "rows": recording_results},
            )
            print(
                json.dumps(
                    {
                        "recording": recording,
                        "fit": len(fit),
                        "validation": len(validation),
                        "model": model,
                        "validated": mapping["validated"],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    result = {
        "status": "LABEL_BLIND_SOURCE_TIMEBASE_AUDIT_COMPLETE",
        "selection_sha256": sha256_file(output / "selection_before_image_access.json"),
        "initial_selection_sha256": sha256_file(selection_path),
        "fidelity_summary_sha256": sha256_file(fidelity_path),
        "unique_physical_frames": len(rows),
        "supplemental_frames": sum(r["supplemental"] for r in rows),
        "fit_frames": sum(r["split"] == "fit" for r in rows),
        "validation_frames": sum(r["split"] == "validation" for r in rows),
        "suspect_recordings": sorted(suspect),
        "mappings": mappings,
        "all_recordings_validated": all(m["validated"] for m in mappings),
        "elapsed_seconds": time.perf_counter() - started,
        "action_values_used": 0,
        "protected_data_payloads_read": 0,
        "cross_view_sync_verified": False,
        "rows": rows,
    }
    write_json(output / "summary.json", result)
    return result


def validate_alignment(output: Path, search_path: Path) -> dict:
    """Freeze a fit-only simple-timebase rule, then check fresh temporal strata."""
    import av
    import cv2
    from PIL import Image

    search = json.loads(search_path.read_text(encoding="utf-8"))
    lock, clips, frames, allowlist = load_allowed_manifest()
    prior_keys = {(r["recording"], r["center_frame"]) for r in search["rows"]}
    selected, mappings = [], []
    for previous in search["mappings"]:
        candidates = previous["best_model_per_timebase"]
        constant = next(m for m in candidates if m["scale"] == 1.0)
        variable = next(m for m in candidates if m["scale"] != 1.0)
        chosen = variable if variable["fit_mae320"] + 0.05 < constant["fit_mae320"] else constant
        recording = previous["recording"]
        mappings.append(
            {
                "recording": recording,
                "selected_model": chosen,
                "fit_rule": "nonunit_timebase_requires_0.05_fit_MAE320_advantage",
                "fit_candidate_models": candidates,
            }
        )
        unique = {}
        for clip in clips:
            center = int(clip["center_frame"])
            if (
                clip["provider_recording_id"] == recording
                and clip["all_frames_valid"].lower() in {"1", "true"}
                and (recording, center) not in prior_keys
            ):
                unique.setdefault(center, clip)
        for fraction in (0.2, 0.5, 0.8):
            ordered = sorted(unique)
            center = ordered[round(fraction * (len(ordered) - 1))]
            selected.append(dict(unique.pop(center), validation_quantile=fraction))
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output / "frozen_mapping_before_new_images.json", mappings)
    write_json(output / "selection_before_image_access.json", selected)
    cv2.setNumThreads(1)
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as archive:
        for mapping in mappings:
            recording = mapping["recording"]
            model = mapping["selected_model"]
            choices = [c for c in selected if c["provider_recording_id"] == recording]
            paths = [p for p in (SOURCE_DIR / "allowed_videos").iterdir() if p.stem == recording]
            if len(paths) != 1 or recording not in EXPECTED_RECORDINGS:
                raise RuntimeError("Native video allowlist mismatch")
            cap = cv2.VideoCapture(str(paths[0]))
            container = av.open(str(paths[0]))
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            checks = []
            for clip in choices:
                row = frames[clip["sample_id"]][8]
                if row["image_member"] not in allowlist:
                    raise RuntimeError("New validation center outside manifest")
                center = int(row["source_frame"])
                predicted = round(model["scale"] * center + model["offset"])
                with Image.open(io.BytesIO(archive.read(row["image_member"]))) as im:
                    reference = cv2.resize(
                        np.asarray(im.convert("RGB")), (320, 180), interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, predicted - 3))
                scores, predicted_small = [], None
                candidate_small = {}
                for index in range(max(0, predicted - 3), predicted + 4):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    small = cv2.cvtColor(
                        cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
                    )
                    scores.append(
                        {
                            "index": index,
                            "mae320": float(np.abs(small.astype(np.float32) - reference).mean()),
                        }
                    )
                    candidate_small[index] = small
                    if index == predicted:
                        predicted_small = small
                target_pts = round(predicted / float(stream.average_rate) / float(stream.time_base))
                container.seek(target_pts, stream=stream, backward=True)
                av_frame = next((f for f in container.decode(stream) if f.pts >= target_pts), None)
                if av_frame is None or predicted_small is None:
                    raise RuntimeError("Mapped frame absent in one decoder")
                av_small = cv2.resize(
                    av_frame.to_ndarray(format="rgb24"), (320, 180), interpolation=cv2.INTER_AREA
                )
                decoder_mae = float(
                    np.abs(av_small.astype(np.float32) - predicted_small.astype(np.float32)).mean()
                )
                decoder_candidates = {
                    index: float(
                        np.abs(av_small.astype(np.float32) - pixels.astype(np.float32)).mean()
                    )
                    for index, pixels in candidate_small.items()
                }
                decoder_excess = decoder_mae - min(decoder_candidates.values())
                prediction = next(s for s in scores if s["index"] == predicted)
                excess = prediction["mae320"] - min(s["mae320"] for s in scores)
                check = {
                    "sample_id": clip["sample_id"],
                    "recording": recording,
                    "center_frame": center,
                    "native_index": predicted,
                    "mae320": prediction["mae320"],
                    "excess_mae320": excess,
                    "pyav_actual_pts": av_frame.pts,
                    "pyav_target_pts": target_pts,
                    "pyav_time_base": str(av_frame.time_base),
                    "pyav_actual_seconds": av_frame.time,
                    "cross_decoder_mae320": decoder_mae,
                    "cross_decoder_excess_mae320": decoder_excess,
                    "cross_decoder_candidates": decoder_candidates,
                    "candidates": scores,
                    "passed": prediction["mae320"] < 6
                    and excess < 0.25
                    and decoder_mae < 2.0
                    and decoder_excess < 0.05
                    and av_frame.pts == target_pts,
                }
                rows.append(check)
                checks.append(check)
            cap.release()
            container.close()
            mapping["validation_checks"] = checks
            mapping["validated"] = all(c["passed"] for c in checks)
            write_json(output / f"recording_{recording}.json", mapping)
            print(
                json.dumps(
                    {
                        "fresh_validation_recording": recording,
                        "model": model,
                        "validated": mapping["validated"],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    result = {
        "status": "FRESH_DISJOINT_TIMEBASE_AND_TRUE_PTS_VALIDATION_COMPLETE",
        "search_summary_sha256": sha256_file(search_path),
        "fidelity_summary_sha256": search["fidelity_summary_sha256"],
        "selection_sha256": sha256_file(output / "selection_before_image_access.json"),
        "frozen_mapping_sha256": sha256_file(output / "frozen_mapping_before_new_images.json"),
        "all_recordings_validated": all(m["validated"] for m in mappings),
        "mappings": mappings,
        "new_disjoint_frames": len(rows),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "action_values_used": 0,
        "protected_data_payloads_read": 0,
        "pyav_version": av.__version__,
        "cross_view_sync_verified": False,
        "interpretation": "Fresh within-recording native/JPEG timebase validation, not cross-drone synchronization",
    }
    write_json(output / "summary.json", result)
    return result


def support_census(output: Path, alignment_path: Path, fidelity_path: Path) -> dict:
    """Demux packet timestamps only; no image decoding or row dropping."""
    import av

    lock, clips, frames, _ = load_allowed_manifest()
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if not alignment["all_recordings_validated"] or alignment[
        "fidelity_summary_sha256"
    ] != sha256_file(fidelity_path):
        raise RuntimeError("Support census requires verified lineage")
    p0_path = ROOT / lock["p0_extraction_lock"]["path"]
    if sha256_file(p0_path) != lock["p0_extraction_lock"]["sha256"]:
        raise RuntimeError("P0 fallback lock changed")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    artifact = p0["manifest"]["artifacts"]["frame_manifest"]
    if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
        raise RuntimeError("P0 fallback frame manifest changed")
    short = defaultdict(list)
    for row in read_csv(ROOT / artifact["path"]):
        short[row["sample_id"]].append(row)
    for clip in clips:
        if clip["all_frames_valid"].lower() not in {"1", "true"}:
            frames[clip["sample_id"]] = sorted(
                short[clip["sample_id"]], key=lambda r: int(r["time_index"])
            )
    models = {m["recording"]: m["selected_model"] for m in alignment["mappings"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    video_rows, support = [], {}
    for video in fidelity["videos"]:
        recording = video["recording"]
        if recording not in EXPECTED_RECORDINGS:
            raise RuntimeError("Video census outside retained recordings")
        path = SOURCE_DIR / "allowed_videos" / Path(video["zip_member"]).name
        if sha256_file(path) != video["sha256"]:
            raise RuntimeError("Video census source bytes changed")
        receipt = {"recording": recording, "sha256": video["sha256"]}
        for mode, options in (("default", {}), ("ignore_editlist", {"ignore_editlist": "1"})):
            container = av.open(str(path), options=options)
            stream = container.streams.video[0]
            indices = []
            for packet in container.demux(stream):
                if packet.pts is not None:
                    value = float(packet.pts * stream.time_base * stream.average_rate)
                    if abs(value - round(value)) > 1e-6:
                        raise RuntimeError("Nonintegral packet timebase requires separate audit")
                    indices.append(round(value))
            container.close()
            nonnegative = {i for i in indices if i >= 0}
            support[(recording, mode)] = nonnegative
            receipt[mode] = {
                "packets_with_pts": len(indices),
                "negative_pts_packets": sum(i < 0 for i in indices),
                "nonnegative_indices": len(nonnegative),
                "min_index": min(indices),
                "max_index": max(indices),
                "interior_missing_indices": sorted(
                    set(range(min(nonnegative), max(nonnegative) + 1)) - nonnegative
                ),
                "sorted_index_sha256": hashlib.sha256(
                    np.asarray(sorted(indices), dtype=np.int64).tobytes()
                ).hexdigest(),
            }
        video_rows.append(receipt)
    rows = []
    for clip in clips:
        recording = clip["provider_recording_id"]
        source = [int(r["source_frame"]) for r in frames[clip["sample_id"]]]
        model = models[recording]
        mapped = [round(model["scale"] * frame + model["offset"]) for frame in source]
        missing_default = [f for f in mapped if f not in support[(recording, "default")]]
        missing_unedited = [f for f in source if f not in support[(recording, "ignore_editlist")]]
        rows.append(
            {
                "sample_id": clip["sample_id"],
                "recording": recording,
                "historical_short_fallback": clip["all_frames_valid"].lower() not in {"1", "true"},
                "source_frames": source,
                "default_missing_mapped_frames": missing_default,
                "ignore_editlist_missing_source_frames": missing_unedited,
                "whole_clip_existing720_fallback_required_default": bool(missing_default),
                "whole_clip_existing720_fallback_required_ignore_editlist": bool(missing_unedited),
            }
        )
    result = {
        "status": "ALL_4977_EXACT_SUPPORT_PACKET_METADATA_CENSUS_COMPLETE",
        "centers": len(rows),
        "historical_short_fallbacks": sum(r["historical_short_fallback"] for r in rows),
        "default_native_complete_centers": sum(
            not r["default_missing_mapped_frames"] for r in rows
        ),
        "ignore_editlist_native_complete_centers": sum(
            not r["ignore_editlist_missing_source_frames"] for r in rows
        ),
        "rows": rows,
        "videos": video_rows,
        "image_decodes": 0,
        "models_loaded": 0,
        "action_values_used": 0,
        "protected_data_payloads_read": 0,
        "policy": "Preserve all4977 centers. Any missing exact native frame requires whole-clip existing720 fallback, never drop or interpolate frames.",
        "ignore_editlist_pixel_mapping_verified_by_this_census": False,
        "alignment_sha256": sha256_file(alignment_path),
        "fidelity_sha256": sha256_file(fidelity_path),
        "pyav_version": av.__version__,
    }
    write_json(output / "summary.json", result)
    return result


def streaming_benchmark(
    output: Path,
    alignment_path: Path,
    fidelity_path: Path,
    census_path: Path,
    *,
    full: bool = False,
) -> dict:
    """Chronological two-view uint8 crop cache benchmark; no encoder or label use."""
    import cv2
    import torch
    from PIL import Image

    from experiments.cache_okutama_video_features import _crop_with_mean_padding
    from hac.okutama_native_video import crop_geometry
    from hac.video_encoders import PADDING_RGB, preprocess_rgb_clip

    lock, clips, frames, allowed = load_allowed_manifest()
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    census = json.loads(census_path.read_text(encoding="utf-8"))
    if (
        not alignment["all_recordings_validated"]
        or census["alignment_sha256"] != sha256_file(alignment_path)
        or census["fidelity_sha256"] != sha256_file(fidelity_path)
    ):
        raise RuntimeError("Chronological benchmark lineage mismatch")
    fallback_ids = {
        r["sample_id"]
        for r in census["rows"]
        if r["whole_clip_existing720_fallback_required_default"]
    }
    if len(fallback_ids) != 29:
        raise RuntimeError("Whole-clip native fallback policy changed")
    p0_path = ROOT / lock["p0_extraction_lock"]["path"]
    if sha256_file(p0_path) != lock["p0_extraction_lock"]["sha256"]:
        raise RuntimeError("Short fallback lock changed")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    short = defaultdict(list)
    for name in ("frame_manifest", "image_allowlist"):
        artifact = p0["manifest"]["artifacts"][name]
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Short fallback manifest changed")
        for row in read_csv(ROOT / artifact["path"]):
            if name == "frame_manifest":
                short[row["sample_id"]].append(row)
            else:
                allowed.add(row["image_member"])
    for clip in clips:
        if clip["all_frames_valid"].lower() not in {"1", "true"}:
            frames[clip["sample_id"]] = sorted(
                short[clip["sample_id"]], key=lambda r: int(r["time_index"])
            )
    models = {m["recording"]: m["selected_model"] for m in alignment["mappings"]}
    videos = {v["recording"]: v for v in fidelity["videos"]}
    crop_keys, crop_rows, clip_crops = {}, [], {}
    frame_crops = defaultdict(lambda: defaultdict(list))
    frame_members = {}
    for clip in clips:
        if clip["sample_id"] in fallback_ids:
            continue
        recording = clip["provider_recording_id"]
        indices = []
        for row in frames[clip["sample_id"]]:
            if row["image_member"] not in allowed:
                raise RuntimeError("Benchmark JPEG outside immutable allowlist")
            source = int(row["source_frame"])
            mapped = round(models[recording]["scale"] * source + models[recording]["offset"])
            box = tuple(float(row[k]) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
            key = (recording, row["provider_track_id"], source, *box)
            if key not in crop_keys:
                index = len(crop_rows)
                crop_keys[key] = index
                bounds = crop_geometry(box, output_size=384, context_fraction=0.25).crop_box
                height, width = bounds[3] - bounds[1], bounds[2] - bounds[0]
                crop_rows.append(
                    {
                        "crop_index": index,
                        "recording": recording,
                        "source_frame": source,
                        "native_frame": mapped,
                        "box720": box,
                        "bounds720": bounds,
                        "native_shape": [3 * height, 3 * width, 3],
                        "low_shape": [height, width, 3],
                        "native_bytes": 27 * height * width,
                        "low_bytes": 3 * height * width,
                    }
                )
                frame_crops[recording][mapped].append(index)
            indices.append(crop_keys[key])
            member_key = (recording, mapped)
            if member_key in frame_members and frame_members[member_key] != row["image_member"]:
                raise RuntimeError("Same frame maps to inconsistent JPEG identity")
            frame_members[member_key] = row["image_member"]
        clip_crops[clip["sample_id"]] = indices
    recording_metadata = []
    for recording, groups in sorted(frame_crops.items()):
        ids = [i for values in groups.values() for i in values]
        recording_metadata.append(
            {
                "recording": recording,
                "offset": models[recording]["offset"],
                "unique_frames": len(groups),
                "decode_span": max(groups) - min(groups) + 1,
                "unique_crops": len(ids),
                "native_clips": sum(
                    c["provider_recording_id"] == recording and c["sample_id"] not in fallback_ids
                    for c in clips
                ),
                "native_bytes": sum(crop_rows[i]["native_bytes"] for i in ids),
                "low_bytes": sum(crop_rows[i]["low_bytes"] for i in ids),
            }
        )
    selected_recordings = []
    for edited in (False, True):
        choices = sorted(
            (r for r in recording_metadata if (r["offset"] != 0) == edited),
            key=lambda r: (r["unique_crops"], r["recording"]),
        )
        selected_recordings.append(choices[len(choices) // 2]["recording"])
    if full:
        selected_recordings = sorted(EXPECTED_RECORDINGS)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    metadata = {
        "all_centers": 4977,
        "native_centers": len(clip_crops),
        "whole_clip_existing720_fallbacks": sorted(fallback_ids),
        "requested_actor_frame_uses": 16 * len(clip_crops),
        "unique_native_frames": len(frame_members),
        "unique_actor_frame_crops": len(crop_rows),
        "decode_span": sum(r["decode_span"] for r in recording_metadata),
        "native_uint8_bytes": sum(r["native_bytes"] for r in crop_rows),
        "low_uint8_bytes": sum(r["low_bytes"] for r in crop_rows),
        "recordings": recording_metadata,
        "selected_recordings": selected_recordings,
        "execution_scope": "all21_authorized_full_cache" if full else "two_view_benchmark",
        "selection_rule": "median unique-crop workload view separately for zero and nonzero offsets",
        "alignment_sha256": sha256_file(alignment_path),
        "census_sha256": sha256_file(census_path),
        "action_values_used": 0,
    }
    write_json(output / "metadata_and_selection_before_images.json", metadata)
    write_json(output / "cohort_crop_index.json", {"crops": crop_rows, "clip_crops": clip_crops})
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    measured, stored = [], {}
    started = time.perf_counter()
    with zipfile.ZipFile(lock["inputs"]["archive"]["path"]) as archive:
        for recording in selected_recordings:
            path = SOURCE_DIR / "allowed_videos" / Path(videos[recording]["zip_member"]).name
            if sha256_file(path) != videos[recording]["sha256"]:
                raise RuntimeError("Chronological source digest changed")
            cap = cv2.VideoCapture(str(path))
            targets = frame_crops[recording]
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(targets))
            rec_start = time.perf_counter()
            decode_time = gate_time = crop_time = 0.0
            frame_checks, decoded, retrieved, crops_written = [], 0, 0, 0
            native_path, low_path = (
                output / f"{recording}_native.uint8",
                output / f"{recording}_down720.uint8",
            )
            with (
                native_path.open("xb", buffering=8 * 1024 * 1024) as native_stream,
                low_path.open("xb", buffering=8 * 1024 * 1024) as low_stream,
            ):
                for source_index in range(min(targets), max(targets) + 1):
                    tick = time.perf_counter()
                    if not cap.grab():
                        raise RuntimeError(
                            f"Chronological decode unavailable: {recording}/{source_index}"
                        )
                    decoded += 1
                    if source_index not in targets:
                        decode_time += time.perf_counter() - tick
                        continue
                    ok, bgr = cap.retrieve()
                    if not ok:
                        raise RuntimeError("Required native frame could not be retrieved")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    retrieved += 1
                    decode_time += time.perf_counter() - tick
                    tick = time.perf_counter()
                    low = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)
                    with Image.open(
                        io.BytesIO(archive.read(frame_members[(recording, source_index)]))
                    ) as im:
                        reference = cv2.resize(
                            np.asarray(im.convert("RGB")), (320, 180), interpolation=cv2.INTER_AREA
                        )
                    mae = float(
                        np.abs(
                            reference.astype(np.float32)
                            - cv2.resize(low, (320, 180), interpolation=cv2.INTER_AREA).astype(
                                np.float32
                            )
                        ).mean()
                    )
                    frame_checks.append({"native_frame": source_index, "mae320": mae})
                    if mae >= 6:
                        raise RuntimeError(
                            f"Chronological timing gate failed: {recording}/{source_index}: {mae}"
                        )
                    gate_time += time.perf_counter() - tick
                    tick = time.perf_counter()
                    native_image, low_image = Image.fromarray(rgb), Image.fromarray(low)
                    for index in targets[source_index]:
                        row = crop_rows[index]
                        left, top, right, bottom = (3 * v for v in row["bounds720"])
                        tile = Image.new("RGB", (right - left, bottom - top), PADDING_RGB)
                        valid = (max(0, left), max(0, top), min(3840, right), min(2160, bottom))
                        tile.paste(native_image.crop(valid), (valid[0] - left, valid[1] - top))
                        native_array = np.asarray(tile, dtype=np.uint8)
                        low_array = np.asarray(
                            _crop_with_mean_padding(low_image, tuple(row["box720"])), dtype=np.uint8
                        )
                        if (
                            list(native_array.shape) != row["native_shape"]
                            or list(low_array.shape) != row["low_shape"]
                        ):
                            raise RuntimeError("Physical crop support changed")
                        stored[index] = {
                            "crop_index": index,
                            "recording": recording,
                            "native_offset": native_stream.tell(),
                            "low_offset": low_stream.tell(),
                            "native_shape": row["native_shape"],
                            "low_shape": row["low_shape"],
                        }
                        native_stream.write(native_array.tobytes())
                        low_stream.write(low_array.tobytes())
                        crops_written += 1
                    crop_time += time.perf_counter() - tick
            cap.release()
            result = {
                "recording": recording,
                "seconds": time.perf_counter() - rec_start,
                "decode_seconds": decode_time,
                "pixel_gate_seconds": gate_time,
                "crop_and_write_seconds": crop_time,
                "decoded_frames": decoded,
                "retrieved_frames": retrieved,
                "unique_crops": crops_written,
                "native_bytes": native_path.stat().st_size,
                "low_bytes": low_path.stat().st_size,
                "max_mae320": max(r["mae320"] for r in frame_checks),
                "checks": frame_checks,
            }
            measured.append(result)
            write_json(output / f"recording_{recording}.json", result)
            print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
    write_json(output / "stored_crop_offsets.json", list(stored.values()))
    samples = []
    for recording in selected_recordings:
        sample_ids = [
            c["sample_id"]
            for c in clips
            if c["provider_recording_id"] == recording and c["sample_id"] in clip_crops
        ]
        sample_ids.sort(
            key=lambda s: hashlib.sha256(("hac-uint8-readback-20260908:" + s).encode()).digest()
        )
        for sample_id in sample_ids[:16]:
            for arm, filename, offset_key, shape_key in (
                ("native", f"{recording}_native.uint8", "native_offset", "native_shape"),
                ("down720", f"{recording}_down720.uint8", "low_offset", "low_shape"),
            ):
                tick = time.perf_counter()
                images = []
                with (output / filename).open("rb") as stream:
                    for index in clip_crops[sample_id]:
                        receipt = stored[index]
                        stream.seek(receipt[offset_key])
                        shape = tuple(receipt[shape_key])
                        images.append(
                            np.frombuffer(stream.read(int(np.prod(shape))), dtype=np.uint8).reshape(
                                shape
                            )
                        )
                pixels = preprocess_rgb_clip(images)
                if pixels.shape != (3, 16, 384, 384) or not torch.isfinite(pixels).all():
                    raise RuntimeError("Uint8 shard readback failed")
                samples.append(
                    {"sample_id": sample_id, "arm": arm, "seconds": time.perf_counter() - tick}
                )
    totals = {
        key: sum(r[key] for r in measured)
        for key in (
            "decode_seconds",
            "pixel_gate_seconds",
            "crop_and_write_seconds",
            "decoded_frames",
            "retrieved_frames",
            "unique_crops",
        )
    }
    estimate = (
        totals["decode_seconds"] / totals["decoded_frames"] * metadata["decode_span"]
        + totals["pixel_gate_seconds"]
        / totals["retrieved_frames"]
        * metadata["unique_native_frames"]
        + totals["crop_and_write_seconds"]
        / totals["unique_crops"]
        * metadata["unique_actor_frame_crops"]
    )
    medians = {
        arm: float(np.median([s["seconds"] for s in samples if s["arm"] == arm]))
        for arm in ("native", "down720")
    }
    result = {
        "status": "ALL21_CHRONOLOGICAL_UINT8_CROP_CACHE_COMPLETE"
        if full
        else "TWO_RECORDING_CHRONOLOGICAL_UINT8_BENCHMARK_COMPLETE",
        "metadata": metadata,
        "recording_results": measured,
        "readback_clips": len(samples) // 2,
        "readback_median_seconds_per_arm": medians,
        "projected_full_crop_build_seconds": estimate,
        "projected_full_two_arm_read_preprocess_seconds": len(clip_crops) * sum(medians.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(output.glob("*.uint8"))
        },
        "readback_rows": samples,
        "gpu_models_loaded": 0,
        "classifier_fits": 0,
        "action_values_used": 0,
        "protected_data_payloads_read": 0,
        "timing_concurrency": "Main-agent small-head training may overlap; no exclusive hardware claim",
        "extrapolation_scope": "Actual full21 crop extraction"
        if full
        else "Two metadata-selected median-workload views, not completed full21 extraction",
    }
    write_json(output / "summary.json", result)
    return result


def verify_streaming_cache(output: Path) -> dict:
    """Read-only shard replay against already pixel-gated native64 input tensors."""
    import torch

    from hac.video_encoders import preprocess_rgb_clip

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for name, receipt in summary["artifacts"].items():
        if (output / name).stat().st_size != receipt["bytes"] or sha256_file(
            output / name
        ) != receipt["sha256"]:
            raise RuntimeError("Chronological uint8 shard bytes changed")
    index = json.loads((output / "cohort_crop_index.json").read_text(encoding="utf-8"))[
        "clip_crops"
    ]
    stored = {
        r["crop_index"]: r
        for r in json.loads((output / "stored_crop_offsets.json").read_text(encoding="utf-8"))
    }
    native64 = ROOT / ".runs/research_20260908/native4k_matched_pilot"
    selection = json.loads(
        (native64 / "selection_before_image_access.json").read_text(encoding="utf-8")
    )
    gate = json.loads((native64 / "pixel_gate.json").read_text(encoding="utf-8"))
    if sha256_file(native64 / "selection_before_image_access.json") != gate["selection_sha256"]:
        raise RuntimeError("Native64 selection digest changed")
    torch.set_num_threads(1)
    checks = []
    for clip_index, clip in enumerate(selection):
        sample_id, recording = clip["sample_id"], clip["provider_recording_id"]
        if sample_id not in index or not all(i in stored for i in index[sample_id]):
            continue
        for arm, filename, offset_key, shape_key in (
            ("native4k", f"{recording}_native.uint8", "native_offset", "native_shape"),
            ("exact4k_downsample720", f"{recording}_down720.uint8", "low_offset", "low_shape"),
        ):
            images = []
            with (output / filename).open("rb") as stream:
                for crop in index[sample_id]:
                    receipt = stored[crop]
                    stream.seek(receipt[offset_key])
                    shape = tuple(receipt[shape_key])
                    images.append(
                        np.frombuffer(stream.read(int(np.prod(shape))), dtype=np.uint8).reshape(
                            shape
                        )
                    )
            pixels = preprocess_rgb_clip(images).numpy()
            reference_path = native64 / "pixels" / f"{clip_index}_{arm}.npy"
            if sha256_file(reference_path) != gate["pixel_artifacts"][reference_path.name]:
                raise RuntimeError("Native64 reference tensor changed")
            reference = np.load(reference_path, allow_pickle=False)
            checks.append(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    "bit_exact": bool(np.array_equal(pixels, reference)),
                    "max_abs_difference": float(np.max(np.abs(pixels - reference))),
                }
            )
    if not checks or not all(c["bit_exact"] for c in checks):
        raise RuntimeError("Chronological cache differs from exact pilot preprocessing")
    result = {
        "status": "CHRONOLOGICAL_UINT8_CACHE_EXACT_INPUT_REPLAY_VERIFIED",
        "clips": len(checks) // 2,
        "arm_tensors": len(checks),
        "all_bit_exact": True,
        "max_abs_difference": max(c["max_abs_difference"] for c in checks),
        "source_summary_sha256": sha256_file(output / "summary.json"),
        "native64_pixel_gate_sha256": sha256_file(native64 / "pixel_gate.json"),
        "metadata_artifacts": {
            name: sha256_file(output / name)
            for name in (
                "metadata_and_selection_before_images.json",
                "cohort_crop_index.json",
                "stored_crop_offsets.json",
            )
        },
        "rows": checks,
        "image_decodes": 0,
        "gpu_models_loaded": 0,
    }
    write_json(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "prepare",
            "fidelity",
            "paired",
            "align",
            "validate_alignment",
            "support_census",
            "streaming_benchmark",
            "streaming_verify",
            "streaming_full",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--fidelity-summary", type=Path)
    parser.add_argument("--alignment-search", type=Path)
    parser.add_argument("--support-census", type=Path)
    parser.add_argument("--archive4k", type=Path, default=SOURCE_DIR / "TrainSetVideos.zip")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.output_dir)
    elif args.mode == "paired":
        result = paired_views(args.output_dir)
    elif args.mode == "align":
        if args.selection is None or args.fidelity_summary is None:
            parser.error("align requires --selection and --fidelity-summary")
        result = alignment_search(args.output_dir, args.selection, args.fidelity_summary)
    elif args.mode == "validate_alignment":
        if args.alignment_search is None:
            parser.error("validate_alignment requires --alignment-search")
        result = validate_alignment(args.output_dir, args.alignment_search)
    elif args.mode == "support_census":
        if args.alignment_search is None or args.fidelity_summary is None:
            parser.error("support_census requires --alignment-search and --fidelity-summary")
        result = support_census(args.output_dir, args.alignment_search, args.fidelity_summary)
    elif args.mode in {"streaming_benchmark", "streaming_full"}:
        if (
            args.alignment_search is None
            or args.fidelity_summary is None
            or args.support_census is None
        ):
            parser.error(
                "streaming_benchmark requires --alignment-search, --fidelity-summary and --support-census"
            )
        result = streaming_benchmark(
            args.output_dir,
            args.alignment_search,
            args.fidelity_summary,
            args.support_census,
            full=args.mode == "streaming_full",
        )
    elif args.mode == "streaming_verify":
        result = verify_streaming_cache(args.output_dir)
    else:
        if args.selection is None:
            parser.error("fidelity requires --selection")
        result = fidelity(args.output_dir, args.selection, args.archive4k)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "videos"}}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
