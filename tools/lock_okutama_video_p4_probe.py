"""Bind P4 materialized features and P3 lineage before any P4 estimator fit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, _ROOT / "experiments"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import materialize_okutama_video_p4 as materializer  # noqa: E402
import run_okutama_video_probe as p1  # noqa: E402

from tools import lock_hac_continuation_protocols as common  # noqa: E402
from tools import lock_okutama_video_p4_materialization as materialization  # noqa: E402

STATUS = "OKUTAMA_VIDEO_P4_PROBE_LOCKED_BEFORE_FITTING"
AUTHORIZATION = {
    "probe_fitting": True,
    "frozen_feature_read": True,
    "feature_materialization": False,
    "raw_image_access": False,
    "backbone_fitting": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": materialization.PROTOCOL_PATH,
    "materialization_locker": "tools/lock_okutama_video_p4_materialization.py",
    "materializer": "experiments/materialize_okutama_video_p4.py",
    "probe_locker": "tools/lock_okutama_video_p4_probe.py",
    "runner": "experiments/run_okutama_video_p4.py",
    "feature_module": "src/hac/video_kinematic.py",
    "p3_feature_module": "src/hac/video_multiscale.py",
    "p3_runner": "experiments/run_okutama_video_p3.py",
    "p2_runner": "experiments/run_okutama_video_p2a.py",
    "p1_runner": "experiments/run_okutama_video_probe.py",
    "requirements": "requirements-video-lock.txt",
}


def _path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def receipt(root: Path, value: str | Path) -> dict[str, Any]:
    path = _path(root, value)
    digest, size = common._sha256_file(path)
    return {
        "path": common._relative_path(root, path),
        "sha256": digest,
        "size_bytes": size,
    }


def checked(root: Path, item: dict[str, Any]) -> tuple[Path, bytes]:
    path = _path(root, item["path"])
    raw = common._read_bytes(path)
    if len(raw) != int(item["size_bytes"]) or p1.sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"P4 fitting input changed before decoding: {path}")
    return path, raw


def verified(root: Path, item: dict[str, Any]) -> Path:
    path = _path(root, item["path"])
    observed = receipt(root, path)
    if (
        observed["sha256"] != item["sha256"]
        or observed["size_bytes"] != int(item["size_bytes"])
    ):
        raise RuntimeError(f"P4 fitting input changed before value access: {path}")
    return path


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P4 fitting")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _ancestor(root: Path, old: str, new: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", old, new],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("Historical P4/P3 execution commit is not an ancestor")


def _historical_lock(
    root: Path,
    item: dict[str, Any],
    *,
    status: str,
    current_commit: str,
) -> dict[str, Any]:
    _, raw = checked(root, item)
    lock = json.loads(raw)
    if lock.get("status") != status:
        raise RuntimeError("Historical execution-lock status changed")
    _ancestor(root, lock["repository_commit"], current_commit)
    for name, source in lock["sources"].items():
        blob = common._git(
            root, "rev-parse", f"{lock['repository_commit']}:{source['path']}"
        )
        if blob != source["git_blob_oid"]:
            raise RuntimeError(f"Historical source blob changed: {name}")
    return lock


def _validate_p3(
    root: Path, spec: dict[str, Any], current_commit: str
) -> tuple[dict[str, Any], p1.PrimaryData, dict[str, Any]]:
    reference = spec["references"]["p3_best"]
    p3_lock = _historical_lock(
        root,
        reference["probe_lock"],
        status="OKUTAMA_VIDEO_P3_PROBE_LOCKED_BEFORE_FITTING",
        current_commit=current_commit,
    )
    if p3_lock.get("authorization", {}).get("probe_fitting") is not True:
        raise RuntimeError("P3 lineage did not authorize fitting")
    p2_path, p2_raw = checked(root, p3_lock["p2a_reference"]["execution_lock"])
    p2_lock = json.loads(p2_raw)
    data = p1.load_primary_data(root, p2_lock)
    p1._validate_inner_map(data, p2_lock)
    p1.validate_randomization(p2_lock)
    if len(data.labels) != 4977 or tuple(sorted(np.unique(data.folds))) != tuple(range(5)):
        raise RuntimeError("P4 primary data/fold contract changed")

    _, summary_raw = checked(root, reference["summary"])
    summary = json.loads(summary_raw)
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P3_EXPLORATORY_CROSSFIT_COMPLETE"
        or summary.get("rows") != 4977
        or summary.get("reference_predictions_used_for_fitting") is not False
        or summary.get("models", {})
        .get(reference["arm"], {})
        .get("metrics", {})
        .get("macro_f1")
        != reference["macro_f1"]
    ):
        raise RuntimeError("P3 reference summary changed")
    # Byte-verify only. The P3 comparison probabilities are decoded after all P4 fits.
    verified(root, reference["oof"])
    for encoder, cache in p3_lock["long_caches"].items():
        expected = {"summary.json", "request.json", "model_receipt.json", "completed.npy", "validity.npy", f"{cache['arm']}.npy"}
        if set(cache["artifacts"]) != expected:
            raise RuntimeError(f"P3 long-cache inventory changed: {encoder}")
        for artifact in cache["artifacts"].values():
            verified(root, artifact)
    return p3_lock, data, {
        "probe_lock": reference["probe_lock"],
        "summary": reference["summary"],
        "oof": reference["oof"],
        "arm": reference["arm"],
        "macro_f1": reference["macro_f1"],
        "prefit_validation": "sha256_and_size_only_no_probability_decode",
        "p2_execution_lock": receipt(root, p2_path),
    }


def _validate_materialized(
    root: Path,
    lock_path: Path,
    feature_dir: Path,
    current_commit: str,
    sample_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_item = receipt(root, lock_path)
    source_lock = _historical_lock(
        root,
        lock_item,
        status=materialization.STATUS,
        current_commit=current_commit,
    )
    if source_lock.get("authorization") != materialization.AUTHORIZATION:
        raise RuntimeError("P4 materialization authority changed")
    directory = _path(root, feature_dir)
    if {path.name for path in directory.iterdir() if path.is_file()} != {
        "request.json",
        "features.npz",
        "summary.json",
    }:
        raise RuntimeError("P4 materialized feature inventory changed")
    items = {name: receipt(root, directory / name) for name in ("request.json", "features.npz", "summary.json")}
    summary = json.loads(checked(root, items["summary.json"])[1])
    request = json.loads(checked(root, items["request.json"])[1])
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P4_PRIMARY_FEATURE_CACHE_COMPLETE"
        or summary.get("rows") != 4977
        or summary.get("feature_sha256") != items["features.npz"]["sha256"]
        or summary.get("request_sha256") != p1.canonical_digest(request)
        or request.get("materialization_lock_sha256") != lock_item["sha256"]
        or summary.get("access_accounting", {}).get("protected_rows_read") != 0
        or summary.get("access_accounting", {}).get("model_fits") != 0
    ):
        raise RuntimeError("P4 materialized feature provenance changed")
    feature_path = verified(root, items["features.npz"])
    with np.load(feature_path, allow_pickle=False) as source:
        if set(source.files) != materializer.OUTPUT_MEMBERS:
            raise RuntimeError("P4 materialized feature members changed")
        if not np.array_equal(source["sample_ids"], sample_ids):
            raise RuntimeError("P4 materialized feature identity/order changed")
        for name, shape in materializer.SHAPES.items():
            values = source[name]
            if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
                raise RuntimeError(f"P4 materialized array changed: {name}")
        camera = source["camera_quality"]
        if np.any((camera < 0) | (camera > 1)):
            raise RuntimeError("P4 materialized camera quality changed")
    return source_lock, {
        "directory": common._relative_path(root, directory),
        "artifacts": items,
    }


def build_payload(
    root: Path, materialization_lock: Path, feature_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = materialization.load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit)
        for name, path in SOURCES.items()
    }
    p3_lock, data, p3_reference = _validate_p3(root, spec, commit)
    source_lock, features = _validate_materialized(
        root,
        materialization_lock,
        feature_dir,
        commit,
        data.sample_ids,
    )
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "p3_reference": p3_reference,
        "p3_long_caches": p3_lock["long_caches"],
        "p3_fold_map_rows": p3_lock["fold_map_rows"],
        "p3_fold_map_sha256": p3_lock["fold_map_sha256"],
        "p3_randomization": p3_lock["randomization"],
        "materialization_lock": receipt(root, materialization_lock),
        "materialization_repository_commit": source_lock["repository_commit"],
        "features": features,
        "sample_ids_sha256": p1.canonical_digest(data.sample_ids.tolist()),
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "raw_images_read": 0,
            "feature_extractions": 0,
            "model_fits": 0,
            "protected_rows_read": 0,
            "p3_reference_probabilities_decoded": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P4 fitting lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P4 fitting lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_probe_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P4 pre-fit execution lock")
    source_lock = _path(root, retained["materialization_lock"]["path"])
    feature_dir = _path(root, retained["features"]["directory"])
    compare(retained, build_payload(root, source_lock, feature_dir))
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialization-lock", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.mode == "check":
        payload = validate_probe_lock(root, output)
    else:
        if args.materialization_lock is None or args.feature_dir is None:
            parser.error("prepare/lock require --materialization-lock and --feature-dir")
        payload = build_payload(
            root, args.materialization_lock.resolve(), args.feature_dir.resolve()
        )
        if args.mode == "lock":
            write_lock(root, output, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "rows": 4977,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
