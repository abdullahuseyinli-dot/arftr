"""Freeze complete role-safe video/image caches and nested splits before probe fitting."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import lock_okutama_video_protocol as p0  # noqa: E402

STATUS = "OKUTAMA_VIDEO_P1_LOCKED_BEFORE_PROBE_FITTING"
PROTOCOL_PATH = "experiments/okutama_video_p1_protocol.json"


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads(p0.common._read_bytes(root / PROTOCOL_PATH))
    if (
        spec.get("status") != "DECLARED_BEFORE_OKUTAMA_P1_PROBE_FITTING"
        or spec.get("protocol_version") != "1.0.0"
    ):
        raise RuntimeError("Unexpected P1 protocol/version")
    if (
        spec.get("primary_rows") != 4977
        or spec.get("seed") != 42
        or spec.get("fold_contract") != p0.FOLDS
    ):
        raise RuntimeError("P1 primary cohort, folds or seed changed")
    if spec["arms"] != ["vjepa21_real_clip", "vjepa21_repeated_center", "dinov2_native_frames"]:
        raise RuntimeError("P1 arms changed")
    if (
        spec["probes"]["linear"]["C_values"] != [0.01, 0.1, 1.0]
        or spec["probes"]["linear"]["inner_folds"] != 3
    ):
        raise RuntimeError("P1 linear candidate budget changed")
    expected_authorization = {
        "probe_fitting": True,
        "raw_image_extraction": False,
        "feature_extraction": False,
        "backbone_fitting": False,
        "protected_data_access": False,
    }
    if spec.get("authorization") != expected_authorization:
        raise RuntimeError("P1 authorizes only probe fitting on frozen eligible features")
    return spec


def receipt(root: Path, path: Path, expected: str | None = None) -> dict[str, Any]:
    selected = p0.safe_path(root, path)
    sha256, size = p0.common._sha256_file(selected)
    if expected is not None and sha256 != expected:
        raise RuntimeError(f"Artifact SHA256 mismatch before decoding: {selected}")
    return {"path": p0.common._relative_path(root, selected), "sha256": sha256, "size_bytes": size}


def bound_json(
    root: Path, path: Path, expected: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = p0.safe_path(root, path)
    raw = p0.common._read_bytes(selected)
    if expected is not None and p0.digest(raw) != expected:
        raise RuntimeError("JSON hash changed before decoding")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("Expected an artifact JSON object")
    return value, {
        "path": p0.common._relative_path(root, selected),
        "sha256": p0.digest(raw),
        "size_bytes": len(raw),
    }


def validate_p0_lineage(
    root: Path, extraction_path: Path, current_commit: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Validate an immutable P0 ancestor without relabelling its original commit.

    All P0 execution sources must still match their original committed blobs.
    New P1 files can be added in descendant commits; changing a P0 source fails.
    """
    extraction, extraction_raw = p0.read_lock(root, extraction_path)
    if (
        extraction.get("status") != p0.EXTRACTION_STATUS
        or extraction.get("authorization", {}).get("model_fitting") is not False
    ):
        raise RuntimeError("P1 requires the original non-fitting P0 extraction lock")
    source_path = p0.safe_path(root, root / extraction["source_lock"]["path"])
    p0.checked_bytes(root, source_path, extraction["source_lock"]["sha256"])
    source, source_raw = p0.read_lock(root, source_path)
    if source.get("status") != p0.MATERIALIZATION_STATUS:
        raise RuntimeError("P0 source-lock status changed")
    commit = source["repository_commit"]
    try:
        p0.common._git(root, "merge-base", "--is-ancestor", commit, current_commit)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("P0 source commit is not an ancestor of current P1 HEAD") from error
    if source["repository_tree"] != p0.common._git(root, "rev-parse", f"{commit}^{{tree}}"):
        raise RuntimeError("P0 retained Git tree changed")
    for name, item in source["sources"].items():
        observed = p0.common.git_source_receipt(root, item["path"], commit)
        if observed != item or extraction["sources"].get(name) != item:
            raise RuntimeError("P0 execution source changed after extraction")
    spec = p0.load_protocol(root)
    primary, inputs = p0.read_primary_cohort(root, spec)
    for name, observed in inputs.items():
        if source["inputs"].get(name) != observed:
            raise RuntimeError("P0 historical eligible input changed")
    for name, external in (
        ("archive", spec["archive"]),
        (
            "checkpoint",
            {
                "file_name": spec["upstream"]["checkpoint_name"],
                "sha256": spec["upstream"]["checkpoint_sha256"],
                "size_bytes": spec["upstream"]["checkpoint_size_bytes"],
            },
        ),
    ):
        item = source["inputs"][name]
        observed = p0.opaque_receipt(
            Path(item["path"]),
            name=external["file_name"],
            expected=external["sha256"],
            size=external["size_bytes"],
        )
        if observed != item:
            raise RuntimeError("P0 pinned external receipt changed")
    upstream = {
        **p0.upstream_receipt(Path(source["upstream"]["path"]), spec["upstream"]["commit"]),
        "url": spec["upstream"]["url"],
        "pretraining_overlap_status": spec["upstream"]["pretraining_overlap_status"],
    }
    if upstream != source["upstream"]:
        raise RuntimeError("P0 upstream state changed")
    snapshot = p0.snapshot_receipt(
        Path(source["inputs"]["dinov2_snapshot"]["path"]), spec["image_control"]
    )
    if snapshot != source["inputs"]["dinov2_snapshot"]:
        raise RuntimeError("P0 pinned DINO snapshot changed")
    for key in (
        "repository_commit",
        "repository_tree",
        "protocol_sha256",
        "sources",
        "inputs",
        "upstream",
        "primary_rows",
        "primary_scenarios",
        "sampling",
        "preprocessing",
        "encoder_cache",
        "image_control",
        "image_control_cache",
        "pilot",
        "environment",
    ):
        if extraction.get(key) != source.get(key):
            raise RuntimeError(f"P0 extraction/source chain differs: {key}")
    expected_source_hashes = {
        name: item["sha256"] for name, item in {**source["sources"], **source["inputs"]}.items()
    }
    if source["source_sha256"] != expected_source_hashes or extraction["source_sha256"] != {
        **expected_source_hashes,
        "materialization_lock": p0.digest(source_raw),
    }:
        raise RuntimeError("P0 source digest chain changed")
    if (
        source["authorization"] != spec["lock_stages"]["materialization"]["authorization"]
        or extraction["authorization"] != spec["lock_stages"]["extraction"]["authorization"]
    ):
        raise RuntimeError("P0 authorization changed")
    if source["primary_sample_ids_sha256"] != p0.canonical_digest(
        [row["sample_id"] for row in primary]
    ):
        raise RuntimeError("P0 primary identity receipt changed")
    if source["environment"] != p0.common.collect_environment():
        raise RuntimeError("Environment changed since P0; declare an explicit amendment before P1")
    summary_path = p0.safe_path(root, root / extraction["manifest"]["artifacts"]["summary"]["path"])
    manifest = p0.validate_manifest_artifacts(
        root, summary_path.parent, primary, p0.digest(source_raw), spec
    )
    if (
        manifest != extraction["manifest"]
        or manifest["requested_image_members_sha256"] != source["requested_image_members_sha256"]
    ):
        raise RuntimeError("P0 exact manifest/allowlist changed")
    return (
        extraction,
        {
            "path": p0.common._relative_path(root, extraction_path),
            "sha256": p0.digest(extraction_raw),
            "size_bytes": len(extraction_raw),
        },
        primary,
    )


def verified_npy(
    root: Path, directory: Path, item: dict[str, Any], shape: list[int], dtype: str
) -> tuple[np.ndarray, dict[str, Any]]:
    name = item.get("path", "")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npy"):
        raise RuntimeError("Cache arrays must use local declared .npy basenames")
    path = p0.safe_path(root, directory / name)
    observed = receipt(root, path, item["sha256"])
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype.hasobject or list(array.shape) != shape or str(array.dtype) != dtype:
        raise RuntimeError("Frozen array shape/dtype differs from protocol")
    return array, observed


def validate_feature_cache(
    root: Path,
    directory: Path,
    kind: str,
    spec: dict[str, Any],
    expected_ids: list[str],
    expected_validity: np.ndarray,
    extraction_digest: str,
    extractor_digest: str,
) -> dict[str, Any]:
    directory = p0.safe_path(root, directory)
    summary, summary_receipt = bound_json(root, directory / "summary.json")
    contract = spec["cache_contracts"][kind]
    if (
        summary.get("status") != contract["status"]
        or summary.get("mode") != "full"
        or summary.get("all_primary_rows") is not True
        or summary.get("rows") != len(expected_ids)
    ):
        raise RuntimeError(
            "P1 requires a completed FULL cache; pilot/incomplete caches are forbidden"
        )
    if summary.get("sample_ids") != expected_ids:
        raise RuntimeError("Frozen cache sample-ID order changed")
    if summary.get("extraction_lock_sha256") != extraction_digest:
        raise RuntimeError("Cache came from another extraction lock")
    for name in ("candidate_fits", "labels_used_for_fitting_or_selection", "protected_rows_read"):
        if summary.get(name) != 0:
            raise RuntimeError(f"Cache must declare zero {name}")
    request, request_receipt = bound_json(root, directory / "request.json")
    request_hash = p0.canonical_digest(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    if (
        request.get("request_sha256") != request_hash
        or summary.get("request_sha256") != request_hash
    ):
        raise RuntimeError("Cache request digest chain changed")
    if (
        request.get("mode") != "full"
        or request.get("sample_ids_sha256") != p0.canonical_digest(expected_ids)
        or request.get("extraction_lock_sha256") != extraction_digest
        or request.get("extractor_sha256") != extractor_digest
    ):
        raise RuntimeError("Cache request source/mode/identity changed")
    arms = list(contract["arms"])
    if list(summary.get("arms", {})) != arms or request.get("arms") != arms:
        raise RuntimeError("Unexpected cache arms or column order")
    validity, validity_receipt = verified_npy(
        root, directory, summary["validity"], [len(expected_ids), len(arms)], "bool"
    )
    completed, completed_receipt = verified_npy(
        root, directory, summary["completed"], [len(expected_ids)], "bool"
    )
    if not completed.all():
        raise RuntimeError("Cache contains incomplete original centers")
    if not np.array_equal(validity, expected_validity):
        raise RuntimeError("Frozen cache validity differs from native manifest")
    features = {}
    for column, name in enumerate(arms):
        item = summary["arms"][name]
        if (
            item.get("path") != name + ".npy"
            or item.get("shape") != contract["arms"][name]
            or item.get("dtype") != "float16"
            or item.get("valid_rows") != int(validity[:, column].sum())
        ):
            raise RuntimeError("Cache arm shape/dtype/valid-row declaration changed")
        array, observed = verified_npy(root, directory, item, contract["arms"][name], "float16")
        for start in range(0, len(expected_ids), 64):
            block = np.asarray(array[start : start + 64])
            mask = validity[start : start + 64, column]
            if not np.isfinite(block).all() or np.any(block[~mask] != 0):
                raise RuntimeError("Frozen features nonfinite or invalid-row features are nonzero")
        if receipt(root, root / observed["path"]) != observed:
            raise RuntimeError("Frozen cache array changed during validation")
        features[name] = {
            **observed,
            "shape": contract["arms"][name],
            "dtype": "float16",
            "valid_rows": int(validity[:, column].sum()),
            "validity": validity_receipt,
            "validity_path": validity_receipt["path"],
            "validity_column": column,
        }
    for item in (validity_receipt, completed_receipt):
        if receipt(root, root / item["path"]) != item:
            raise RuntimeError("Cache validity/completion changed during validation")
    return {
        "summary": summary_receipt,
        "request": request_receipt,
        "validity": validity_receipt,
        "completed": completed_receipt,
        "features": features,
    }


def validate_baseline(
    root: Path, spec: dict[str, Any], primary: list[dict[str, str]]
) -> dict[str, Any]:
    item = spec["baseline"]
    path = root / item["path"]
    raw = p0.checked_bytes(root, path, item["sha256"])
    if len(raw) != item["size_bytes"]:
        raise RuntimeError("Baseline byte count changed")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise RuntimeError("Baseline object dtype is forbidden")
    if (
        arrays["sample_ids"].tolist() != [row["sample_id"] for row in primary]
        or arrays["recording_ids"].tolist() != [row["recording_id"] for row in primary]
        or arrays["labels"].tolist() != [int(row["label_index"]) for row in primary]
    ):
        raise RuntimeError("Baseline identity/order/labels/scenarios differ from clip index")
    probabilities = arrays[item["probabilities_member"]]
    if (
        probabilities.shape != (len(primary), 3)
        or not np.issubdtype(probabilities.dtype, np.floating)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise RuntimeError("Baseline probabilities are invalid")
    labels = arrays["labels"].astype(int)
    confusion = np.bincount(labels * 3 + probabilities.argmax(1), minlength=9).reshape(3, 3)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(
        2 * np.diag(confusion), denominator, out=np.zeros(3), where=denominator > 0
    ).mean()
    if abs(float(f1) - item["macro_f1"]) > 1e-12:
        raise RuntimeError("Baseline macro-F1 differs from its declared historical result")
    return {
        "path": item["path"],
        "sha256": p0.digest(raw),
        "size_bytes": len(raw),
        "probabilities_member": item["probabilities_member"],
        "macro_f1": float(f1),
        "role": item["role"],
    }


def validate_cache_provenance(
    root: Path,
    video: dict[str, Any],
    dino: dict[str, Any],
    p0_lock: dict[str, Any],
    sources: dict[str, Any],
) -> None:
    video_request = json.loads(
        p0.checked_bytes(root, root / video["request"]["path"], video["request"]["sha256"])
    )
    dino_request = json.loads(
        p0.checked_bytes(root, root / dino["request"]["path"], dino["request"]["sha256"])
    )
    if video_request.get("checkpoint_sha256") != p0_lock["inputs"]["checkpoint"]["sha256"]:
        raise RuntimeError("Video cache checkpoint differs from the P0 lock")
    if (
        dino_request.get("shared_video_extractor_sha256") != sources["video_cache"]["sha256"]
        or dino_request.get("image_encoder_module_sha256")
        != sources["image_encoder_module"]["sha256"]
    ):
        raise RuntimeError("DINO cache used different preprocessing/encoder source")
    snapshot = dino_request.get("snapshot", {})
    expected = p0_lock["inputs"]["dinov2_snapshot"]
    if (
        snapshot.get("revision") != expected["revision"]
        or dino_request.get("revision") != expected["revision"]
        or snapshot.get("path") != expected["path"]
        or set(snapshot.get("files", {})) != set(expected["files"])
    ):
        raise RuntimeError("DINO cache snapshot identity differs from P0")
    for name, item in expected["files"].items():
        actual = snapshot["files"][name]
        if actual.get("sha256") != item["sha256"] or actual.get("size_bytes") != item["size_bytes"]:
            raise RuntimeError("DINO cache snapshot bytes differ from P0")


def make_fold_map(primary: list[dict[str, str]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    labels = np.array([int(row["label_index"]) for row in primary])
    groups = np.array([row["recording_id"] for row in primary])
    outer = np.array([int(row["fold"].split("-")[1]) for row in primary])
    result = [
        {
            "sample_id": row["sample_id"],
            "recording_id": row["recording_id"],
            "outer_fold": int(outer[index]),
            **{f"inner_fold_o{fold}": -1 for fold in range(5)},
        }
        for index, row in enumerate(primary)
    ]
    linear = spec["probes"]["linear"]
    for fold in range(5):
        training = np.flatnonzero(outer != fold)
        splitter = StratifiedGroupKFold(
            n_splits=linear["inner_folds"],
            shuffle=linear["inner_shuffle"],
            random_state=linear["inner_seed"],
        )
        for inner, (fit, held) in enumerate(
            splitter.split(np.zeros(len(training)), labels[training], groups[training])
        ):
            if set(groups[training[fit]]) & set(groups[training[held]]):
                raise RuntimeError("An inner scenario crosses fitting/held partitions")
            for index in training[held]:
                result[int(index)][f"inner_fold_o{fold}"] = inner
        if any(result[int(index)][f"inner_fold_o{fold}"] < 0 for index in training):
            raise RuntimeError("Incomplete nested fold assignment")
    return result


def build_p1_payload(
    root: Path, extraction_lock: Path, video_cache: Path, dino_cache: Path
) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = p0.require_clean_repository(root)
    spec = load_protocol(root)
    sources = {
        name: p0.common.git_source_receipt(root, relative, commit)
        for name, relative in spec["execution_sources"].items()
    }
    p0_lock, p0_receipt, primary = validate_p0_lineage(root, extraction_lock, commit)
    clip_receipt = p0_lock["manifest"]["artifacts"]["clip_index"]
    clips = p0.csv_rows(p0.checked_bytes(root, root / clip_receipt["path"], clip_receipt["sha256"]))
    ids = [row["sample_id"] for row in primary]
    if [row["sample_id"] for row in clips] != ids:
        raise RuntimeError("P0 clip index lost primary order")
    frame_receipt = p0_lock["manifest"]["artifacts"]["frame_manifest"]
    frames = p0.csv_rows(
        p0.checked_bytes(root, root / frame_receipt["path"], frame_receipt["sha256"])
    )
    if len(frames) != len(primary) * 16:
        raise RuntimeError("Native validity needs all sixteen frame rows per center")
    frame_validity = np.array(
        [p0.boolean(row["valid_frame"]) for row in frames], dtype=bool
    ).reshape(len(primary), 16)
    all_valid = frame_validity.all(1)
    center_valid = frame_validity[:, 8]
    if not np.array_equal(
        all_valid, [p0.boolean(row["all_frames_valid"]) for row in clips]
    ) or not np.array_equal(center_valid, [p0.boolean(row["center_frame_valid"]) for row in clips]):
        raise RuntimeError("Clip-level validity differs from the locked frame mask")
    video = validate_feature_cache(
        root,
        video_cache,
        "video",
        spec,
        ids,
        np.stack((all_valid, center_valid), axis=1),
        p0_receipt["sha256"],
        sources["video_cache"]["sha256"],
    )
    dino = validate_feature_cache(
        root,
        dino_cache,
        "dino",
        spec,
        ids,
        all_valid[:, None],
        p0_receipt["sha256"],
        sources["dino_cache"]["sha256"],
    )
    validate_cache_provenance(root, video, dino, p0_lock, sources)
    baseline = validate_baseline(root, spec, primary)
    fold_map = make_fold_map(primary, spec)
    labels = np.array([int(row["label_index"]) for row in primary])
    outer = np.array([row["outer_fold"] for row in fold_map])
    for arm, valid in (
        ("vjepa21_real_clip", all_valid),
        ("vjepa21_repeated_center", center_valid),
        ("dinov2_native_frames", all_valid),
    ):
        for fold in range(5):
            for inner in range(3):
                inner_values = np.array([row[f"inner_fold_o{fold}"] for row in fold_map])
                fit = valid & (outer != fold) & (inner_values != inner)
                held = valid & (outer != fold) & (inner_values == inner)
                if set(labels[fit]) != {0, 1, 2} or not held.any():
                    raise RuntimeError(
                        f"Insufficient valid inner train/held support for {arm}, outer{fold}, inner{inner}"
                    )
    statistics = spec["statistics"]
    draws = np.random.default_rng(statistics["bootstrap_seed"]).integers(
        0, 11, size=(statistics["bootstrap_resamples"], 11), dtype=np.int64
    )
    signs = (2 * ((np.arange(2048)[:, None] >> np.arange(11)) & 1) - 1).astype(np.int8)
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "p0_extraction_lock": p0_receipt,
        "inputs": {
            "clip_index": clip_receipt,
            "frame_manifest": p0_lock["manifest"]["artifacts"]["frame_manifest"],
            "baseline": baseline,
        },
        "baseline": baseline,
        "caches": {"video": video, "dino": dino},
        "features": {**video["features"], **dino["features"]},
        "primary_rows": len(primary),
        "fold_map_rows": fold_map,
        "fold_map_sha256": p0.canonical_digest(fold_map),
        "randomization": {
            "scenario_order": statistics["scenario_order"],
            "bootstrap_indices_shape": list(draws.shape),
            "bootstrap_indices_dtype": "int64",
            "bootstrap_indices_bytes_sha256": p0.digest(draws.tobytes()),
            "swap_signs_shape": list(signs.shape),
            "swap_signs_dtype": "int8",
            "swap_signs_bytes_sha256": p0.digest(signs.tobytes()),
        },
        "environment": p0.common.collect_environment(),
        "authorization": spec["authorization"],
        "access_accounting": {
            "raw_image_payloads_read": 0,
            "protected_rows_read": 0,
            "probe_fits": 0,
            "backbone_fits": 0,
            "role_safe_cache_values_validated": True,
        },
    }


def validate_p1_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    retained, _ = p0.read_lock(root, lock_path)
    if retained.get("status") != STATUS:
        raise RuntimeError("Expected a P1 pre-fit execution lock")
    p0_receipt = retained["p0_extraction_lock"]
    p0.checked_bytes(root, root / p0_receipt["path"], p0_receipt["sha256"])
    video_summary = root / retained["caches"]["video"]["summary"]["path"]
    dino_summary = root / retained["caches"]["dino"]["summary"]["path"]
    current = build_p1_payload(
        root, root / p0_receipt["path"], video_summary.parent, dino_summary.parent
    )
    p0.compare_lock(retained, current)
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-lock", type=Path)
    parser.add_argument("--video-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "check":
        payload = validate_p1_lock(root, args.output.resolve())
    else:
        if None in (args.extraction_lock, args.video_cache, args.dino_cache):
            parser.error("lock requires --extraction-lock, --video-cache, --dino-cache")
        payload = build_p1_payload(
            root,
            args.extraction_lock.resolve(),
            args.video_cache.resolve(),
            args.dino_cache.resolve(),
        )
        p0.write_lock(root, args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "rows": payload["primary_rows"],
                "authorization": payload["authorization"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
