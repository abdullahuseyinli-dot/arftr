"""Bind completed long caches and historical references before any P3 probe fitting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import run_okutama_video_probe as p1  # noqa: E402

from hac.video_multiscale import ARMS  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p3_probe_protocol.json"
PROTOCOL_STATUS = "DECLARED_BEFORE_OKUTAMA_P3_LONG_FEATURE_EXTRACTION_OR_PROBE_FITTING"
STATUS = "OKUTAMA_VIDEO_P3_PROBE_LOCKED_BEFORE_FITTING"
AUTHORIZATION = {
    "probe_fitting": True,
    "frozen_feature_read": True,
    "raw_image_access": False,
    "feature_extraction": False,
    "backbone_fitting": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p3_probe.py",
    "runner": "experiments/run_okutama_video_p3.py",
    "feature_module": "src/hac/video_multiscale.py",
    "p1_runner": "experiments/run_okutama_video_probe.py",
    "p2a_runner": "experiments/run_okutama_video_p2a.py",
    "p1_probe_module": "src/hac/video_probe.py",
    "requirements": "requirements-video-lock.txt",
}
CACHE_FILES = (
    "summary.json",
    "request.json",
    "model_receipt.json",
    "completed.npy",
    "validity.npy",
)


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P3 fitting")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    result = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, result)
    return result


def receipt(root: Path, path: Path) -> dict[str, Any]:
    path = _path(root, path)
    digest, size = common._sha256_file(path)
    return {"path": common._relative_path(root, path), "sha256": digest, "size_bytes": size}


def checked(root: Path, item: dict[str, Any]) -> tuple[Path, bytes]:
    path = _path(root, item["path"])
    raw = common._read_bytes(path)
    if len(raw) != item["size_bytes"] or p1.sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"P3 fitting input changed before decoding: {path}")
    return path, raw


def _ancestor(root: Path, old: str, new: str) -> None:
    process = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", old, new],
        check=False,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError("Historical P3/P2 execution commit is not an ancestor")


def comparison_name(item: dict[str, str]) -> str:
    return f"{item['candidate']}_vs_{item['reference']}"


def expected_comparisons() -> list[dict[str, str]]:
    return [
        *({"candidate": arm, "reference": "baseline"} for arm in ARMS),
        *({"candidate": arm, "reference": "p2a_best"} for arm in ARMS),
        {"candidate": "dual_scale_vjepa", "reference": "long_vjepa_mean"},
        {"candidate": "dual_scale_vjepa_motion", "reference": "dual_scale_vjepa"},
        {"candidate": "dual_scale_vjepa_dino", "reference": "dual_scale_vjepa"},
        {
            "candidate": "dual_scale_factorized",
            "reference": "dual_scale_vjepa_dino",
        },
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != ("1.0.0", PROTOCOL_STATUS):
        raise RuntimeError("Unexpected P3 probe protocol/version")
    if tuple(spec.get("arms", ())) != ARMS or spec.get("primary_rows") != 4977:
        raise RuntimeError("P3 arm or cohort contract changed")
    probe = spec.get("probe", {})
    if (
        probe.get("C_values") != [1e-5, 1e-4, 1e-3, 1e-2]
        or probe.get("solver") != "lbfgs"
        or probe.get("class_weight") != "balanced"
        or probe.get("inner_folds") != 3
        or probe.get("inner_seed") != 42
        or probe.get("maximum_unique_estimator_fits") != 455
    ):
        raise RuntimeError("P3 fitting/search budget changed")
    comparisons = spec.get("statistics", {}).get("comparisons", [])
    if comparisons != expected_comparisons() or len(
        {comparison_name(item) for item in comparisons}
    ) != 16:
        raise RuntimeError("P3 comparison family changed")
    valid_references = {"baseline", "p2a_best", *ARMS}
    if any(item.get("candidate") not in ARMS or item.get("reference") not in valid_references for item in comparisons):
        raise RuntimeError("P3 comparison target changed")
    return spec


def _historical_lock(
    root: Path, item: dict[str, Any], expected_status: str, current_commit: str
) -> tuple[dict[str, Any], bytes]:
    _, raw = checked(root, item)
    value = json.loads(raw)
    if value.get("status") != expected_status:
        raise RuntimeError("Historical execution lock status changed")
    _ancestor(root, value["repository_commit"], current_commit)
    for name, source in value["sources"].items():
        blob = common._git(
            root, "rev-parse", f"{value['repository_commit']}:{source['path']}"
        )
        if blob != source["git_blob_oid"]:
            raise RuntimeError(f"Historical execution source changed: {name}")
    return value, raw


def _validate_p2(
    root: Path, spec: dict[str, Any], current_commit: str
) -> tuple[dict[str, Any], p1.PrimaryData, dict[str, Any]]:
    reference = spec["references"]["p2a_best"]
    p2_lock, _ = _historical_lock(
        root,
        reference["execution_lock"],
        "OKUTAMA_VIDEO_P2A_LOCKED_BEFORE_PROBE_FITTING",
        current_commit,
    )
    if p2_lock.get("authorization", {}).get("probe_fitting") is not True:
        raise RuntimeError("P2a lineage never authorized its completed probe")
    data = p1.load_primary_data(root, p2_lock)
    if any(not mask.all() for mask in data.validity.values()):
        raise RuntimeError("P3 requires complete short-cache anchors")
    _, summary_raw = checked(root, reference["summary"])
    summary = json.loads(summary_raw)
    arm = reference["arm"]
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P2A_EXPLORATORY_CROSSFIT_COMPLETE"
        or summary.get("rows") != 4977
        or summary.get("reference_predictions_used_for_fitting") is not False
        or summary.get("models", {}).get(arm, {}).get("metrics", {}).get("macro_f1")
        != reference["macro_f1"]
    ):
        raise RuntimeError("P2a reference summary changed")
    # Hash the immutable comparison evidence, but deliberately do not decode it.
    # The runner opens these probabilities only after all 30 P3 workloads finish.
    checked(root, reference["oof"])
    return p2_lock, data, {
        "execution_lock": reference["execution_lock"],
        "summary": reference["summary"],
        "oof": reference["oof"],
        "arm": arm,
        "macro_f1": reference["macro_f1"],
        "prefit_validation": "sha256_and_size_only_no_probability_decode",
    }


def _validate_extraction(
    root: Path, extraction_lock_path: Path, current_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = receipt(root, extraction_lock_path)
    lock, _ = _historical_lock(
        root,
        item,
        "OKUTAMA_VIDEO_P3_EXTRACTION_LOCKED_BEFORE_IMAGE_ACCESS",
        current_commit,
    )
    authorization = lock.get("authorization", {})
    if (
        authorization.get("frozen_extraction") is not True
        or authorization.get("model_fitting") is not False
        or lock.get("manifest", {}).get("all_long_frames_valid_rows") != 4510
    ):
        raise RuntimeError("P3 extraction lineage changed")
    return lock, item


def _validate_cache(
    root: Path,
    directory: Path,
    *,
    encoder: str,
    arm: str,
    tail: tuple[int, ...],
    sample_ids: np.ndarray,
    extraction_sha: str,
) -> tuple[dict[str, Any], np.ndarray]:
    directory = _path(root, directory)
    feature_name = f"{arm}.npy"
    expected_names = {*CACHE_FILES, feature_name}
    if {path.name for path in directory.iterdir() if path.is_file()} != expected_names:
        raise RuntimeError(f"Unexpected file inventory in completed P3 {encoder} cache")
    items = {name: receipt(root, directory / name) for name in sorted(expected_names)}
    summary = json.loads(checked(root, items["summary.json"])[1])
    request = json.loads(checked(root, items["request.json"])[1])
    model = json.loads(checked(root, items["model_receipt.json"])[1])
    if (
        summary.get("status") != "OKUTAMA_P3_LONG_FROZEN_FEATURE_CACHE_COMPLETE"
        or summary.get("encoder") != encoder
        or summary.get("mode") != "full"
        or summary.get("rows") != 4977
        or summary.get("all_primary_rows") is not True
        or summary.get("candidate_fits") != 0
        or summary.get("labels_used_for_fitting_or_selection") != 0
        or summary.get("protected_rows_read") != 0
        or summary.get("extraction_lock_sha256") != extraction_sha
        or summary.get("sample_ids") != sample_ids.tolist()
        or request.get("request_sha256") != summary.get("request_sha256")
        or model.get("request_sha256") != summary.get("request_sha256")
    ):
        raise RuntimeError(f"Completed P3 {encoder} cache provenance changed")
    declared = summary.get("arms", {}).get(arm, {})
    if (
        declared.get("shape") != [4977, *tail]
        or declared.get("dtype") != "float16"
        or declared.get("valid_rows") != 4510
        or declared.get("invalid_rows") != 467
        or declared.get("sha256") != items[feature_name]["sha256"]
    ):
        raise RuntimeError(f"Completed P3 {encoder} feature declaration changed")
    completed = np.load(checked(root, items["completed.npy"])[0], allow_pickle=False)
    validity = np.load(checked(root, items["validity.npy"])[0], allow_pickle=False)
    feature_path = checked(root, items[feature_name])[0]
    features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
    if (
        completed.shape != (4977,)
        or completed.dtype != np.bool_
        or not completed.all()
        or validity.shape != (4977,)
        or validity.dtype != np.bool_
        or int(validity.sum()) != 4510
        or features.shape != (4977, *tail)
        or features.dtype != np.float16
    ):
        raise RuntimeError(f"Completed P3 {encoder} cache arrays changed")
    for start in range(0, 4977, 64):
        block = np.asarray(features[start : start + 64])
        mask = validity[start : start + 64]
        if not np.isfinite(block[mask]).all() or np.any(block[~mask] != 0):
            raise RuntimeError("P3 cache contains invalid valid rows or nonzero sentinels")
    return {"directory": common._relative_path(root, directory), "artifacts": items, "arm": arm, "shape": [4977, *tail]}, validity


def build_payload(
    root: Path, extraction_lock_path: Path, vjepa_cache: Path, dino_cache: Path
) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, relative, commit)
        for name, relative in SOURCES.items()
    }
    p2_lock, data, p2_reference = _validate_p2(root, spec, commit)
    extraction, extraction_item = _validate_extraction(root, extraction_lock_path, commit)
    vjepa, vj_valid = _validate_cache(
        root,
        vjepa_cache,
        encoder="vjepa21",
        arm="vjepa21_long16_real_clip",
        tail=(8, 9, 768),
        sample_ids=data.sample_ids,
        extraction_sha=extraction_item["sha256"],
    )
    dino, dino_valid = _validate_cache(
        root,
        dino_cache,
        encoder="dinov2",
        arm="dinov2_long16_native_frames",
        tail=(16, 1, 768),
        sample_ids=data.sample_ids,
        extraction_sha=extraction_item["sha256"],
    )
    if not np.array_equal(vj_valid, dino_valid):
        raise RuntimeError("P3 V-JEPA and DINO validity masks differ")
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "p2a_reference": p2_reference,
        "p2a_lock_protocol": p2_lock["protocol"],
        "p3_extraction_lock": extraction_item,
        "p3_extraction_repository_commit": extraction["repository_commit"],
        "long_caches": {"vjepa21": vjepa, "dinov2": dino},
        "long_valid_rows": int(vj_valid.sum()),
        "long_validity_sha256": p1.canonical_digest(vj_valid.astype(int).tolist()),
        "sample_ids_sha256": p1.canonical_digest(data.sample_ids.tolist()),
        "fold_map_rows": p2_lock["fold_map_rows"],
        "fold_map_sha256": p2_lock["fold_map_sha256"],
        "randomization": p2_lock["randomization"],
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "raw_images_read": 0,
            "feature_extractions": 0,
            "model_fits": 0,
            "protected_rows_read": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P3 fitting lock changed: " + ", ".join(changed))


def validate_probe_lock(root: Path, path: Path) -> dict[str, Any]:
    path = _path(root, path)
    retained = json.loads(path.read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P3 pre-fit execution lock")
    extraction = _path(root, retained["p3_extraction_lock"]["path"])
    vjepa = _path(root, retained["long_caches"]["vjepa21"]["directory"])
    dino = _path(root, retained["long_caches"]["dinov2"]["directory"])
    compare(retained, build_payload(root, extraction, vjepa, dino))
    return retained


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P3 fitting lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-lock", type=Path)
    parser.add_argument("--vjepa-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.mode == "check":
        payload = validate_probe_lock(root, output)
    else:
        if None in (args.extraction_lock, args.vjepa_cache, args.dino_cache):
            parser.error("prepare/lock require --extraction-lock, --vjepa-cache and --dino-cache")
        payload = build_payload(
            root,
            args.extraction_lock.resolve(),
            args.vjepa_cache.resolve(),
            args.dino_cache.resolve(),
        )
        if args.mode == "lock":
            write_lock(root, output, payload)
    print(json.dumps({"mode": args.mode, "status": payload["status"], "rows": 4977, "long_valid_rows": payload["long_valid_rows"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
