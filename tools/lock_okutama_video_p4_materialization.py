"""Bind role-safe P4 rows and mixed feature files before selective value access."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, _ROOT / "experiments"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import run_okutama_video_probe as p1  # noqa: E402

from hac.video_kinematic import ARMS  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p4_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_P3_ANALYSIS_BEFORE_P4_FEATURE_VALUE_ACCESS_OR_FITTING"
STATUS = "OKUTAMA_VIDEO_P4_MATERIALIZATION_LOCKED_BEFORE_FEATURE_VALUE_ACCESS"
AUTHORIZATION = {
    "opaque_source_hashing": True,
    "selective_primary_feature_materialization": True,
    "model_fitting": False,
    "raw_image_access": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p4_materialization.py",
    "materializer": "experiments/materialize_okutama_video_p4.py",
    "feature_module": "src/hac/video_kinematic.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "requirements": "requirements-video-lock.txt",
}
INDEX_COLUMNS = (
    "scope",
    "development_role",
    "sample_id",
    "recording_id",
    "provider_recording_id",
    "track_id",
    "provider_track_id",
    "center_frame",
    "feature_index",
    "fold",
    "label_index",
    "label",
    "transition_window",
    "window_any_occluded",
)
MOTION_ARRAYS = {
    "camera_quality": ("7bf1e2ded903d09bc806c1dfa9e6b10d44a1abf405787f3313a203e28b9217c8", 567180),
    "compensated_sequence": ("29286b95adfca89bbecaacde00bfd3316f71e4f46f9219731cccd6e4b8b41f81", 11908220),
    "compensated_summary": ("84baecf1db517ed730c76601685d14bdc45bb2de66fe522f8ad289eca3f08bdd", 1934776),
    "raw_sequence": ("38dd3af97946b232380e0d06465a6549216b6fe848a76922486d351ace5f1f6e", 11908220),
    "raw_summary": ("9726f9aeafcaee3c713a262aff6cb5b138253dc11be0e7c6713df9270045a014", 1934776),
    "to_centre_homography": ("a2dcfb49a953a624e094e3b50e6067541048c66c8b28e5ebaa7c8b482355b24e", 5103596),
}
MOTION_OTHER = {
    "camera_estimation_by_recording.csv": ("d3517bab43537af17da47b5e7fce7edf2604f76e3905537ffdc012143329781b", 4113),
    "completed.npy": ("facd0bdbaf494c19200602d55c0df55ed8a8b991af3a42afae5d8ae93cd267dc", 8467),
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
        raise RuntimeError(f"P4 source changed before decoding: {path}")
    return path, raw


def verified(root: Path, item: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _path(root, item["path"])
    observed = receipt(root, path)
    if (
        observed["sha256"] != item["sha256"]
        or observed["size_bytes"] != int(item["size_bytes"])
    ):
        raise RuntimeError(f"P4 source changed before value access: {path}")
    return path, observed


def _declared(root: Path, item: dict[str, Any]) -> tuple[Path, bytes, dict[str, Any]]:
    path, raw = checked(root, item)
    return path, raw, receipt(root, path)


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P4 feature access")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _csv(raw: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != INDEX_COLUMNS:
        raise RuntimeError("P4 eligible-index schema changed")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError("Malformed P4 eligible-index row")
    return rows


def expected_comparisons() -> list[dict[str, str]]:
    return [
        *({"candidate": arm, "reference": "p3_best"} for arm in ARMS),
        *(
            {"candidate": arm, "reference": "visual_factorized_refit"}
            for arm in ARMS[1:]
        ),
        {
            "candidate": "visual_comp_track_factorized",
            "reference": "visual_raw_track_factorized",
        },
        {
            "candidate": "visual_dual_track_factorized",
            "reference": "visual_comp_track_factorized",
        },
        {
            "candidate": "camera_reference_marginalized",
            "reference": "visual_comp_track_factorized",
        },
        {
            "candidate": "camera_reference_marginalized",
            "reference": "visual_dual_track_factorized",
        },
        {
            "candidate": "visual_dual_track_multinomial",
            "reference": "visual_dual_track_factorized",
        },
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != (
        "1.0.0",
        PROTOCOL_STATUS,
    ):
        raise RuntimeError("Unexpected P4 protocol/version")
    if tuple(spec.get("arms", ())) != ARMS or spec.get("primary_rows") != 4977:
        raise RuntimeError("P4 arm/cohort contract changed")
    if spec.get("statistics", {}).get("comparisons") != expected_comparisons():
        raise RuntimeError("P4 comparison family changed")
    probe = spec.get("probe", {})
    if (
        probe.get("C_values") != [1e-5, 1e-4, 1e-3, 1e-2]
        or probe.get("maximum_unique_estimator_fits") != 910
        or probe.get("inner_folds") != 3
        or probe.get("inner_seed") != 42
    ):
        raise RuntimeError("P4 fitting budget changed")
    if spec.get("authorization", {}).get("protected_data_access") is not False:
        raise RuntimeError("P4 protected-data boundary changed")
    return spec


def _declared_receipt(item: dict[str, Any], *, path_key: str = "path") -> dict[str, Any]:
    return {
        "path": item[path_key],
        "sha256": item["sha256"],
        "size_bytes": int(item["size_bytes"]),
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit)
        for name, path in SOURCES.items()
    }
    inputs = spec["materialization_inputs"]
    index_path, index_raw, index_receipt = _declared(
        root, _declared_receipt(inputs["eligible_index"])
    )
    rows = _csv(index_raw)
    primary = [row for row in rows if row["scope"] == "grouped_crossfit_oof"]
    if (
        len(rows) != 6360
        or len(primary) != 4977
        or len({row["sample_id"] for row in primary}) != 4977
        or len({row["recording_id"] for row in primary}) != 11
        or any(row["fold"] not in {f"fold-{value}" for value in range(5)} for row in primary)
    ):
        raise RuntimeError("P4 primary row contract changed")
    indices = [int(row["feature_index"]) for row in primary]
    if len(set(indices)) != 4977 or min(indices) < 0 or max(indices) >= 8339:
        raise RuntimeError("P4 source feature-index selection changed")

    bundle = inputs["role_safe_bundle"]
    _, bundle_receipt = verified(
        root,
        {
            "path": bundle["path"],
            "sha256": bundle["sha256"],
            "size_bytes": bundle["size_bytes"],
        },
    )
    _, bundle_summary_raw, bundle_summary_receipt = _declared(
        root,
        {
            "path": bundle["summary_path"],
            "sha256": bundle["summary_sha256"],
            "size_bytes": bundle["summary_size_bytes"],
        },
    )
    bundle_summary = json.loads(bundle_summary_raw)
    if (
        bundle_summary.get("status") != "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE"
        or bundle_summary.get("rows") != 6360
        or bundle_summary.get("excluded_feature_identities_or_values_read") != 0
        or bundle_summary.get("artifact_sha256", {}).get("eligible_feature_bundle.npz")
        != bundle["sha256"]
    ):
        raise RuntimeError("Historical role-safe bundle declaration changed")

    motion = inputs["role_mixed_motion_store"]
    directory = _path(root, motion["directory"])
    expected_inventory = {
        "store.json",
        "request.json",
        *MOTION_OTHER,
        *(f"{name}.npy" for name in MOTION_ARRAYS),
    }
    if {path.name for path in directory.iterdir() if path.is_file()} != expected_inventory:
        raise RuntimeError("Historical motion-store inventory changed")
    store_item = {
        "path": common._relative_path(root, directory / "store.json"),
        "sha256": motion["store_sha256"],
        "size_bytes": motion["store_size_bytes"],
    }
    request_item = {
        "path": common._relative_path(root, directory / "request.json"),
        "sha256": motion["request_sha256"],
        "size_bytes": motion["request_size_bytes"],
    }
    _, store_raw, store_receipt = _declared(root, store_item)
    _, request_raw, request_receipt = _declared(root, request_item)
    store, request = json.loads(store_raw), json.loads(request_raw)
    if (
        store.get("status") != "OKUTAMA_CPTR_FEATURE_STORE_COMPLETE"
        or store.get("samples") != 8339
        or store.get("frames_per_sample") != 17
        or store.get("trajectory_sequence_dim") != 21
        or store.get("trajectory_summary_dim") != 58
        or request.get("status") != "OKUTAMA_CPTR_MOTION_CACHE_REQUEST"
        or request.get("samples") != 8339
        or set(store.get("arrays", {})) != set(MOTION_ARRAYS)
    ):
        raise RuntimeError("Historical motion-store declaration changed")
    motion_receipts = {}
    for name, (digest, size) in MOTION_ARRAYS.items():
        declared = store["arrays"][name]
        if declared != {"path": f"{name}.npy", "sha256": digest}:
            raise RuntimeError(f"Historical motion declaration changed: {name}")
        item = receipt(root, directory / declared["path"])
        if (item["sha256"], item["size_bytes"]) != (digest, size):
            raise RuntimeError(f"Historical motion array changed: {name}")
        motion_receipts[name] = item
    for name, (digest, size) in MOTION_OTHER.items():
        item = receipt(root, directory / name)
        if (item["sha256"], item["size_bytes"]) != (digest, size):
            raise RuntimeError(f"Historical motion completion artifact changed: {name}")
        motion_receipts[name.removesuffix(".npy").removesuffix(".csv")] = item

    p3 = spec["references"]["p3_best"]
    p3_receipts = {}
    for name in ("probe_lock", "summary", "oof"):
        _, raw, item = _declared(root, _declared_receipt(p3[name]))
        p3_receipts[name] = item
        if name == "summary":
            summary = json.loads(raw)
            if (
                summary.get("status")
                != "OKUTAMA_VIDEO_P3_EXPLORATORY_CROSSFIT_COMPLETE"
                or summary.get("rows") != 4977
                or summary.get("models", {})
                .get(p3["arm"], {})
                .get("metrics", {})
                .get("macro_f1")
                != p3["macro_f1"]
            ):
                raise RuntimeError("P3 result declaration changed")

    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "inputs": {
            "eligible_index": index_receipt,
            "role_safe_bundle": bundle_receipt,
            "role_safe_bundle_summary": bundle_summary_receipt,
            "motion_store": store_receipt,
            "motion_request": request_receipt,
            "motion_artifacts": motion_receipts,
            "p3": p3_receipts,
        },
        "primary_sample_ids_sha256": p1.canonical_digest(
            [row["sample_id"] for row in primary]
        ),
        "primary_source_indices_sha256": p1.canonical_digest(indices),
        "primary_source_index_min": min(indices),
        "primary_source_index_max": max(indices),
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "role_safe_bundle_array_values_read": 0,
            "role_mixed_motion_array_values_read": 0,
            "protected_rows_read": 0,
            "raw_images_read": 0,
            "model_fits": 0,
        },
        "selected_primary_rows": 4977,
        "excluded_source_rows": 3362,
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P4 materialization lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P4 materialization lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P4 pre-materialization lock")
    compare(retained, build_payload(root))
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.mode == "check":
        payload = validate_lock(root, output)
    else:
        payload = build_payload(root)
        if args.mode == "lock":
            write_lock(root, output, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "rows": payload["selected_primary_rows"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
