"""Execute, shard, and finalize the full label-blind Okutama CCAC extraction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import run_okutama_ccac_pilot as pilot_run

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.ccac_features import (  # noqa: E402
    CCACFeatures,
    aggregate_clip_features,
    validate_feature_blocks,
)

STATUS = "OKUTAMA_CCAC_FULL_EXTRACTION_COMPLETE"
REUSE_STATUS = "OKUTAMA_CCAC_PILOT_WORKLOAD_REUSED_BY_REFERENCE"
SHARD_STATUS = "OKUTAMA_CCAC_FULL_EXTRACTION_SHARD_COMPLETE"


class CachedZip:
    """Bounded per-process cache of decompressed JPEG bytes."""

    def __init__(self, path: Path, maximum_members: int):
        self.archive = zipfile.ZipFile(path)
        self.maximum_members = int(maximum_members)
        self.cache: OrderedDict[str, bytes] = OrderedDict()

    def getinfo(self, member: str) -> zipfile.ZipInfo:
        return self.archive.getinfo(member)

    def read(self, member: str) -> bytes:
        if member in self.cache:
            raw = self.cache.pop(member)
            self.cache[member] = raw
            return raw
        raw = self.archive.read(member)
        self.cache[member] = raw
        if len(self.cache) > self.maximum_members:
            self.cache.popitem(last=False)
        return raw

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> CachedZip:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _locker():
    return importlib.import_module("tools.lock_okutama_ccac_full")


def _compatibility_lock(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": lock["measurement_protocol"],
        "selected_image_members": lock["selected_image_members"],
        "known_boxes_by_member": lock["known_boxes_by_member"],
        "overlay_sample_ids": lock["pilot_overlay_sample_ids"],
    }


def _output_path(root: Path, value: Path) -> Path:
    output = value.resolve()
    runs = (root / ".runs").resolve()
    if output == runs or not output.is_relative_to(runs):
        raise RuntimeError("Full CCAC output must be a dedicated directory below .runs")
    return output


def _base_request(lock_path: Path, lock: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "status": "OKUTAMA_CCAC_FULL_REQUEST_BEFORE_NONPILOT_PIXEL_ACCESS",
        "lock_sha256": pilot_run.sha256_file(lock_path),
        "protocol_sha256": lock["protocol_sha256"],
        "selected_sample_ids_sha256": lock["selected_sample_ids_sha256"],
        "selected_clip_records_sha256": lock["selected_clip_records_sha256"],
        "selected_image_members_sha256": lock["selected_image_members_sha256"],
        "known_boxes_sha256": lock["known_boxes_sha256"],
        "clips": 4977,
        "requested_pairs": 74655,
        "pilot_reuse_workloads": 128,
        "new_pixel_workloads": 4849,
        "fixed_shard_count": 8,
        "opencv": {"version": lock["opencv"]["version"], "threads": 1, "opencl": False},
        "runner_sha256": pilot_run.sha256_file(Path(__file__)),
        "output": str(output),
        "action_labels_read": False,
        "prediction_rows_read": False,
        "classifier_fits": 0,
    }


def _retained_lock(lock_path: Path) -> dict[str, Any]:
    locker = _locker()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != locker.STATUS or lock.get("authorization") != locker.AUTHORIZATION:
        raise RuntimeError("Expected retained full CCAC execution lock")
    return lock


def _require_request(
    lock_path: Path, lock: dict[str, Any], output: Path
) -> tuple[dict[str, Any], str]:
    request = _base_request(lock_path, lock, output)
    path = output / "request.json"
    if not path.exists() or json.loads(path.read_text()) != request:
        raise RuntimeError("Full CCAC initialization request is absent or changed")
    return request, pilot_run.canonical_digest(request)


def initialize(lock_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    lock = _locker().validate_lock(root, lock_path)
    configured = pilot_run.configure_opencv(threads=1, opencl=False)
    if configured != {"version": lock["opencv"]["version"], "threads": 1, "opencl": False}:
        raise RuntimeError("Full CCAC OpenCV runtime changed")
    request = _base_request(lock_path, lock, output)
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise RuntimeError("Full CCAC output belongs to another request")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty full CCAC output lacks its pre-pixel request")
    else:
        pilot_run.atomic_json(request_path, request)
    synthetic = pilot_run._synthetic_checks(lock["measurement_protocol"])
    synthetic_path = output / "synthetic_checks.json"
    if synthetic_path.exists():
        if json.loads(synthetic_path.read_text()) != synthetic:
            raise RuntimeError("Full CCAC retained synthetic checks changed")
    else:
        pilot_run.atomic_json(synthetic_path, synthetic)
    if not synthetic["passed"]:
        raise RuntimeError("Full CCAC synthetic checks failed before nonpilot pixels")
    return {
        "status": "OKUTAMA_CCAC_FULL_INITIALIZED_BEFORE_NONPILOT_PIXEL_ACCESS",
        "request_sha256": pilot_run.canonical_digest(request),
        "clips": 4977,
        "shards": 8,
    }


def _reuse_request(
    output: Path,
    compatibility: dict[str, Any],
    clip: dict[str, Any],
    request_sha: str,
    pilot_output: Path,
) -> None:
    rank = int(clip["selection_rank"])
    source_directory = pilot_output / "workloads" / f"{rank:03d}"
    source_request = json.loads((source_directory / "request.json").read_text())
    source_pairs, source_clip = pilot_run._validate_workload(source_directory, source_request)
    del source_pairs, source_clip
    target_request = pilot_run._clip_request(compatibility, clip, request_sha)
    left = {key: value for key, value in source_request.items() if key != "run_request_sha256"}
    right = {key: value for key, value in target_request.items() if key != "run_request_sha256"}
    if left != right:
        raise RuntimeError("Pilot workload is not measurement-identical to full extraction")
    directory = output / "workloads" / f"{rank:03d}"
    request_path, receipt_path = directory / "request.json", directory / "receipt.json"
    source_receipt = json.loads((source_directory / "receipt.json").read_text())
    relative_source = source_directory.relative_to(_ROOT).as_posix()
    receipt = {
        "status": REUSE_STATUS,
        "request": target_request,
        "request_sha256": pilot_run.canonical_digest(target_request),
        "source_directory": relative_source,
        "source_request_sha256": pilot_run.canonical_digest(source_request),
        "source_artifacts": source_receipt["artifacts"],
        "jpeg_payloads_decoded": 0,
        "requested_pairs": 15,
    }
    if receipt_path.exists():
        if (
            not request_path.exists()
            or json.loads(request_path.read_text()) != target_request
            or json.loads(receipt_path.read_text()) != receipt
        ):
            raise RuntimeError("Retained full CCAC reuse receipt changed")
        return
    if request_path.exists():
        if json.loads(request_path.read_text()) != target_request or {
            path.name for path in directory.iterdir()
        } != {"request.json"}:
            raise RuntimeError("Incomplete full CCAC reuse receipt changed")
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Incomplete full CCAC reuse receipt retained")
    else:
        pilot_run.atomic_json(request_path, target_request)
    pilot_run.atomic_json(receipt_path, receipt)


def _workload_request(
    compatibility: dict[str, Any], clip: dict[str, Any], request_sha: str
) -> dict[str, Any]:
    return pilot_run._clip_request(compatibility, clip, request_sha)


def _load_workload(
    output: Path,
    compatibility: dict[str, Any],
    clip: dict[str, Any],
    request_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    rank = int(clip["selection_rank"])
    directory = output / "workloads" / f"{rank:03d}"
    request = _workload_request(compatibility, clip, request_sha)
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        raise RuntimeError(f"Full CCAC workload {rank} is incomplete")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") == REUSE_STATUS:
        if (
            receipt.get("request") != request
            or json.loads((directory / "request.json").read_text()) != request
        ):
            raise RuntimeError("Full CCAC reuse workload request changed")
        source = (_ROOT / receipt["source_directory"]).resolve()
        expected_source = (
            _ROOT / ".runs/research_20260907/okutama_ccac_pilot/results/workloads" / f"{rank:03d}"
        ).resolve()
        if source != expected_source:
            raise RuntimeError("Full CCAC reuse source directory changed")
        source_request = json.loads((source / "request.json").read_text())
        pairs, metrics = pilot_run._validate_workload(source, source_request)
        source_receipt = json.loads((source / "receipt.json").read_text())
        if source_receipt["artifacts"] != receipt["source_artifacts"]:
            raise RuntimeError("Full CCAC reuse source artifact map changed")
        if receipt.get("request_sha256") != pilot_run.canonical_digest(request) or receipt.get(
            "source_request_sha256"
        ) != pilot_run.canonical_digest(source_request):
            raise RuntimeError("Full CCAC reuse request digest changed")
        return pairs, metrics, source
    pairs, metrics = pilot_run._validate_workload(directory, request)
    return pairs, metrics, directory


def extract_shard(lock_path: Path, output: Path, shard_index: int) -> dict[str, Any]:
    lock = _retained_lock(lock_path)
    protocol = lock["protocol"]
    shard_count = int(protocol["execution"]["fixed_shard_count"])
    if not 0 <= shard_index < shard_count:
        raise ValueError("Full CCAC shard index is outside the fixed shard count")
    configured = pilot_run.configure_opencv(threads=1, opencl=False)
    if configured != {"version": lock["opencv"]["version"], "threads": 1, "opencl": False}:
        raise RuntimeError("Full CCAC shard OpenCV runtime changed")
    _, request_sha = _require_request(lock_path, lock, output)
    compatibility = _compatibility_lock(lock)
    pilot_ids = set(lock["pilot_sample_ids"])
    assigned = [
        clip
        for clip in lock["selected_clips"]
        if int(clip["selection_rank"]) % shard_count == shard_index
    ]
    assigned.sort(
        key=lambda clip: (
            clip["provider_recording_id"],
            int(clip["center_frame"]),
            int(clip["provider_track_id"]),
            int(clip["selection_rank"]),
        )
    )
    pilot_output = (_ROOT / ".runs/research_20260907/okutama_ccac_pilot/results").resolve()
    archive_path = Path(lock["archive"]["path"])
    reused = computed = retained = 0
    started = time.perf_counter()
    with CachedZip(
        archive_path,
        int(protocol["execution"]["per_process_decompressed_jpeg_byte_lru_members"]),
    ) as archive:
        for clip in assigned:
            rank = int(clip["selection_rank"])
            directory = output / "workloads" / f"{rank:03d}"
            if clip["sample_id"] in pilot_ids:
                existed = (directory / "receipt.json").exists()
                _reuse_request(output, compatibility, clip, request_sha, pilot_output)
                reused += int(not existed)
                retained += int(existed)
            else:
                _, _, created = pilot_run._run_workload(
                    output,
                    compatibility,
                    {**clip, "selection_rank": rank},
                    request_sha,
                    archive,
                )
                computed += int(created)
                retained += int(not created)
    ranks = sorted(int(clip["selection_rank"]) for clip in assigned)
    marker = {
        "status": SHARD_STATUS,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_workloads": len(assigned),
        "assigned_ranks_sha256": pilot_run.canonical_digest(ranks),
        "new_reuse_receipts": reused,
        "new_pixel_workloads": computed,
        "retained_workloads": retained,
        "request_sha256": request_sha,
    }
    marker_path = output / "shards" / f"shard-{shard_index}.json"
    if marker_path.exists():
        prior = json.loads(marker_path.read_text())
        stable_keys = {
            key: value
            for key, value in marker.items()
            if key not in {"new_reuse_receipts", "new_pixel_workloads", "retained_workloads"}
        }
        prior_stable = {
            key: value
            for key, value in prior.items()
            if key not in {"new_reuse_receipts", "new_pixel_workloads", "retained_workloads"}
        }
        if prior_stable != stable_keys:
            raise RuntimeError("Retained full CCAC shard marker changed")
    else:
        pilot_run.atomic_json(marker_path, marker)
    return {**marker, "wall_time_seconds": time.perf_counter() - started}


def _feature_arrays(
    clips: list[dict[str, Any]], pairs_by_clip: list[list[dict[str, Any]]]
) -> tuple[CCACFeatures, dict[str, np.ndarray]]:
    quality, raw, compensated, within = [], [], [], []
    translation_valid, articulation_valid = [], []
    counts = {
        name: [] for name in ("requested", "available", "camera", "translation", "articulation")
    }
    for clip, pairs in zip(clips, pairs_by_clip, strict=True):
        q, r, c, a, tv, av = aggregate_clip_features(pairs, clip)
        quality.append(q)
        raw.append(r)
        compensated.append(c)
        within.append(a)
        translation_valid.append(tv)
        articulation_valid.append(av)
        counts["requested"].append(15)
        counts["available"].append(sum(bool(row["pair_available"]) for row in pairs))
        counts["camera"].append(sum(bool(row["camera_usable"]) for row in pairs))
        counts["translation"].append(sum(bool(row["translation_usable"]) for row in pairs))
        counts["articulation"].append(sum(bool(row["articulation_usable"]) for row in pairs))
    features = CCACFeatures(
        np.stack(quality),
        np.stack(raw),
        np.stack(compensated),
        np.stack(within),
        np.asarray(translation_valid, dtype=bool),
        np.asarray(articulation_valid, dtype=bool),
    )
    validate_feature_blocks(features, len(clips))
    arrays = {
        "sample_ids": np.asarray([clip["sample_id"] for clip in clips]),
        "long_valid": np.asarray([clip["long_valid"] for clip in clips], dtype=bool),
        "quality": features.quality,
        "raw_translation": features.raw_translation,
        "compensated_translation": features.compensated_translation,
        "within_actor": features.within_actor,
        "translation_feature_valid": features.translation_valid,
        "articulation_feature_valid": features.articulation_valid,
        **{
            f"{name}_pair_counts": np.asarray(values, dtype=np.int16)
            for name, values in counts.items()
        },
    }
    return features, arrays


def _replay_nonpilot(lock: dict[str, Any], output: Path, request_sha: str) -> list[dict[str, Any]]:
    compatibility = _compatibility_lock(lock)
    by_id = {clip["sample_id"]: clip for clip in lock["selected_clips"]}
    checks = []
    with CachedZip(Path(lock["archive"]["path"]), 128) as archive:
        for sample_id in lock["nonpilot_replay_sample_ids"]:
            clip = by_id[sample_id]
            pairs, metrics, correspondence, overlay = pilot_run._compute_clip(
                clip, compatibility, archive, include_overlay=False
            )
            assert overlay is None
            retained_pairs, retained_metrics, directory = _load_workload(
                output, compatibility, clip, request_sha
            )
            retained_metrics = dict(retained_metrics)
            retained_metrics.pop("workload_wall_time_seconds", None)
            checks.append(
                {
                    "sample_id": sample_id,
                    "pair_metrics_exact": pairs == retained_pairs,
                    "clip_metrics_exact": metrics == retained_metrics,
                    "correspondence_bytes_exact": hashlib.sha256(correspondence).hexdigest()
                    == pilot_run.sha256_file(directory / "correspondences.npz"),
                }
            )
    if not all(all(value for key, value in item.items() if key != "sample_id") for item in checks):
        raise RuntimeError("Full CCAC nonpilot exact replay failed")
    return checks


def _aggregate_csv(rows: list[dict[str, Any]]) -> bytes:
    return pilot_run._csv_bytes(rows)


def _validate_publication(output: Path, summary: dict[str, Any], request_sha: str) -> None:
    if summary.get("status") != STATUS or summary.get("request_sha256") != request_sha:
        raise RuntimeError("Retained full CCAC summary changed")
    for name, digest in summary.get("artifacts", {}).items():
        if not (output / name).is_file() or pilot_run.sha256_file(output / name) != digest:
            raise RuntimeError(f"Retained full CCAC artifact changed: {name}")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = set(summary["artifacts"]) | {"request.json", "summary.json", "completion.json"}
    if observed != expected:
        raise RuntimeError("Retained full CCAC publication inventory changed")
    completion = json.loads((output / "completion.json").read_text())
    if completion != {
        "status": "OKUTAMA_CCAC_FULL_PUBLICATION_COMPLETE",
        "request_sha256": request_sha,
        "summary_sha256": pilot_run.sha256_file(output / "summary.json"),
        "features_sha256": pilot_run.sha256_file(output / "ccac_features.npz"),
    }:
        raise RuntimeError("Retained full CCAC completion receipt changed")


def finalize(lock_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    lock = _locker().validate_lock(root, lock_path)
    _, request_sha = _require_request(lock_path, lock, output)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        _validate_publication(output, summary, request_sha)
        return summary
    aggregate_names = (
        "ccac_features.npz",
        "feature_manifest.json",
        "pair_metrics.csv",
        "clip_metrics.csv",
        "scenario_metrics.csv",
        "size_metrics.csv",
        "failure_reasons.csv",
        "quality_overview.png",
    )
    if any((output / name).exists() for name in aggregate_names):
        raise RuntimeError("Partial full CCAC aggregate artifacts retained")
    compatibility = _compatibility_lock(lock)
    pairs_by_clip: list[list[dict[str, Any]]] = []
    clip_metrics: list[dict[str, Any]] = []
    workload_sources: list[Path] = []
    for clip in lock["selected_clips"]:
        pairs, metrics, source = _load_workload(output, compatibility, clip, request_sha)
        pairs_by_clip.append(pairs)
        clip_metrics.append(metrics)
        workload_sources.append(source)
    all_pairs = [row for pairs in pairs_by_clip for row in pairs]
    if len(all_pairs) != 74655 or len(clip_metrics) != 4977:
        raise RuntimeError("Full CCAC aggregate denominator changed")
    features, arrays = _feature_arrays(clip_metrics, pairs_by_clip)
    del features
    feature_bytes = pilot_run.deterministic_npz_bytes(arrays)
    replay = _replay_nonpilot(lock, output, request_sha)
    scenario_rows = pilot_run._fraction_rows(clip_metrics, "articulation_clip_usable", "scenario")
    size_rows = pilot_run._fraction_rows(clip_metrics, "articulation_clip_usable", "size_band")
    failure_rows = pilot_run._failure_rows(all_pairs)
    feature_manifest = {
        "status": "OKUTAMA_CCAC_FULL_FEATURES_COMPLETE",
        "rows": 4977,
        "sample_ids_sha256": lock["selected_sample_ids_sha256"],
        "columns": lock["feature_columns"],
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "quantiles": [0.1, 0.5, 0.9],
        "quantile_method": "linear",
        "zero_and_mask_policy": lock["protocol"]["feature_recipe"]["numeric_zero_policy"],
        "labels_present": False,
        "predictions_present": False,
        "measurement_source_sha256": lock["measurement_source_sha256"],
        "per_clip_sources_sha256": pilot_run.canonical_digest(
            [path.relative_to(root).as_posix() for path in workload_sources]
        ),
    }
    artifacts = {
        "ccac_features.npz": feature_bytes,
        "feature_manifest.json": pilot_run.json_bytes(feature_manifest),
        "pair_metrics.csv": _aggregate_csv(all_pairs),
        "clip_metrics.csv": _aggregate_csv(clip_metrics),
        "scenario_metrics.csv": _aggregate_csv(scenario_rows),
        "size_metrics.csv": _aggregate_csv(size_rows),
        "failure_reasons.csv": _aggregate_csv(failure_rows),
        "quality_overview.png": pilot_run._quality_plot(scenario_rows, size_rows, all_pairs),
    }
    for name, raw in artifacts.items():
        pilot_run.atomic_bytes(output / name, raw)
    requested = len(all_pairs)
    camera_pairs = sum(bool(row["camera_usable"]) for row in all_pairs)
    translation_pairs = sum(bool(row["translation_usable"]) for row in all_pairs)
    articulation_pairs = sum(bool(row["articulation_usable"]) for row in all_pairs)
    summary: dict[str, Any] = {
        "status": STATUS,
        "study_id": lock["protocol"]["study_id"],
        "request_sha256": request_sha,
        "execution_lock_sha256": pilot_run.sha256_file(lock_path),
        "rows": 4977,
        "requested_pairs": requested,
        "available_pairs": sum(bool(row["pair_available"]) for row in all_pairs),
        "camera_usable_pairs": camera_pairs,
        "camera_usable_pair_fraction_all_requested": camera_pairs / requested,
        "translation_usable_pairs": translation_pairs,
        "translation_usable_pair_fraction_all_requested": translation_pairs / requested,
        "articulation_usable_pairs": articulation_pairs,
        "articulation_usable_pair_fraction_all_requested": articulation_pairs / requested,
        "translation_feature_valid_rows": int(arrays["translation_feature_valid"].sum()),
        "articulation_feature_valid_rows": int(arrays["articulation_feature_valid"].sum()),
        "scenario_rows": scenario_rows,
        "size_rows": size_rows,
        "camera_model_counts_available_pairs": dict(
            sorted(
                Counter(
                    row.get("camera_selected") for row in all_pairs if bool(row["pair_available"])
                ).items()
            )
        ),
        "distributions": {
            key: pilot_run._distribution(all_pairs, key)
            for key in (
                "background_retained_points",
                "camera_audit_median_pixels",
                "camera_audit_p90_pixels",
                "camera_reduction_fraction",
                "actor_seed_points",
                "actor_primary_retained_points",
                "actor_primary_fb_median_pixels",
                "actor_window_disagreement_median_pixels",
                "compensated_translation_norm_height_per_second",
                "articulation_median_height_per_second",
            )
        },
        "failure_reasons": failure_rows,
        "pilot_reused_workloads": 128,
        "new_pixel_workloads": 4849,
        "nonpilot_exact_replay": replay,
        "action_labels_read": 0,
        "prediction_rows_read": 0,
        "classifier_fits": 0,
        "protected_rows_read": 0,
        "interpretation": (
            "Complete adaptive-development measurement extraction; no classifier was fit "
            "and no action-performance or independent-confirmation claim is authorized."
        ),
    }
    inventory = [output / "synthetic_checks.json", *(output / name for name in artifacts)]
    inventory.extend(sorted(path for path in (output / "shards").rglob("*") if path.is_file()))
    inventory.extend(sorted(path for path in (output / "workloads").rglob("*") if path.is_file()))
    summary["artifacts"] = {
        path.relative_to(output).as_posix(): pilot_run.sha256_file(path) for path in inventory
    }
    pilot_run.atomic_json(summary_path, summary)
    pilot_run.atomic_json(
        output / "completion.json",
        {
            "status": "OKUTAMA_CCAC_FULL_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": pilot_run.sha256_file(summary_path),
            "features_sha256": pilot_run.sha256_file(output / "ccac_features.npz"),
        },
    )
    _validate_publication(output, summary, request_sha)
    return summary


def check(lock_path: Path, output: Path) -> dict[str, Any]:
    lock = _retained_lock(lock_path)
    _, request_sha = _require_request(lock_path, lock, output)
    summary = json.loads((output / "summary.json").read_text())
    _validate_publication(output, summary, request_sha)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("initialize", "extract", "finalize", "check"), required=True
    )
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock_path = args.protocol_lock.resolve()
    output = _output_path(root, args.output_dir)
    if args.mode == "initialize":
        if args.shard_index is not None:
            parser.error("--shard-index is valid only for --mode extract")
        result = initialize(lock_path, output)
    elif args.mode == "extract":
        if args.shard_index is None:
            parser.error("--mode extract requires --shard-index")
        result = extract_shard(lock_path, output, args.shard_index)
    elif args.mode == "finalize":
        if args.shard_index is not None:
            parser.error("--shard-index is valid only for --mode extract")
        result = finalize(lock_path, output)
    else:
        if args.shard_index is not None:
            parser.error("--shard-index is valid only for --mode extract")
        result = check(lock_path, output)
    concise = {
        key: result.get(key)
        for key in (
            "status",
            "shard_index",
            "assigned_workloads",
            "new_reuse_receipts",
            "new_pixel_workloads",
            "retained_workloads",
            "wall_time_seconds",
            "rows",
            "camera_usable_pair_fraction_all_requested",
            "translation_feature_valid_rows",
            "articulation_feature_valid_rows",
        )
        if result.get(key) is not None
    }
    print(json.dumps(concise, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
