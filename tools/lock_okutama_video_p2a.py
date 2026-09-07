"""Bind completed P1 evidence and unchanged frozen caches before any P2a fitting."""

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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import lock_okutama_video_p1 as p1  # noqa: E402

p0 = p1.p0
STATUS = "OKUTAMA_VIDEO_P2A_LOCKED_BEFORE_PROBE_FITTING"
PROTOCOL_PATH = "experiments/okutama_video_p2a_protocol.json"
ARMS = [
    "video_mean_multinomial",
    "video_dino_mean_multinomial",
    "video_dino_motion_multinomial",
    "factorized_specialist",
]
SOURCE_ARMS = ["vjepa21_real_clip", "vjepa21_repeated_center", "dinov2_native_frames"]
AUTHORIZATION = {
    "probe_fitting": True,
    "raw_image_extraction": False,
    "feature_extraction": False,
    "backbone_fitting": False,
    "protected_data_access": False,
}


def expected_comparisons() -> list[dict[str, str]]:
    pairs = [(arm, reference) for reference in ("baseline", "p1_real_linear") for arm in ARMS]
    pairs.extend((arm, ARMS[0]) for arm in ARMS[1:])
    pairs.append((ARMS[3], ARMS[2]))
    return [
        {"name": f"{candidate}_vs_{reference}", "candidate": candidate, "reference": reference}
        for candidate, reference in pairs
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads(p0.common._read_bytes(root / PROTOCOL_PATH))
    if (
        spec.get("protocol_version") != "1.0.0"
        or spec.get("status") != "DECLARED_BEFORE_OKUTAMA_P2A_PROBE_FITTING"
        or spec.get("arms") != ARMS
        or spec.get("primary_rows") != 4977
        or spec.get("seed") != 42
        or spec.get("fold_contract") != p0.FOLDS
        or spec.get("class_order") != ["sitting", "standing", "walking_running"]
    ):
        raise RuntimeError("Unexpected P2a protocol/version/cohort/arms")
    linear = spec["probes"]["linear"]
    numeric_contract = {
        "C_values": [1e-5, 1e-4, 1e-3, 1e-2],
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 2000,
        "tolerance": 0.0001,
        "standardize": True,
        "inner_folds": 3,
        "inner_splitter": "StratifiedGroupKFold",
        "inner_shuffle": True,
        "inner_seed": 42,
    }
    if any(linear.get(key) != value for key, value in numeric_contract.items()):
        raise RuntimeError("P2a prespecified fitting budget changed")
    if spec.get("authorization") != AUTHORIZATION:
        raise RuntimeError("P2a authorizes frozen-feature probe fitting only")
    rules = spec["fitting_rules"]
    if (
        rules.get("all_rows_required_valid") is not True
        or rules.get("baseline_probabilities_in_fitting_or_selection") is not False
        or rules.get("p1_probabilities_in_fitting_or_selection") is not False
        or rules.get("outer_held_baseline_fallback") is not False
    ):
        raise RuntimeError("P2a forbids comparison probabilities in fitting or fallback")
    stats = spec["statistics"]
    if (
        stats.get("bootstrap_resamples") != 10000
        or stats.get("bootstrap_seed") != 20260907
        or stats.get("exact_scenario_swaps") != 2048
        or stats.get("comparisons") != expected_comparisons()
        or stats.get("scenario_order") != sorted(sum(p0.FOLDS.values(), []))
    ):
        raise RuntimeError("P2a randomization/comparison family changed")
    return spec


def verified_receipt(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    observed = p1.receipt(root, root / item["path"], item["sha256"])
    if observed["size_bytes"] != item["size_bytes"]:
        raise RuntimeError("Artifact byte count changed before decoding")
    return observed


def checked_json(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    raw = p0.checked_bytes(root, root / item["path"], item["sha256"])
    if len(raw) != item["size_bytes"]:
        raise RuntimeError("JSON byte count changed before decoding")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("Expected an artifact JSON object")
    return result


def validate_p1_lineage(
    root: Path, spec: dict[str, Any], commit: str, environment: dict[str, Any]
) -> dict[str, Any]:
    """Validate anchored history without re-extracting or reopening raw image archives.

    P1's exact pinned bytes attest the earlier extraction validation. Current P2a
    inputs are independently rehashed below. Every retained P1 source must still
    match its historical Git blob and current bytes; only new sources may differ.
    """
    retained = checked_json(root, spec["p1_artifacts"]["execution_lock"])
    if retained.get("status") != p1.STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the original P1 pre-fit execution lock")
    historical_commit = retained["repository_commit"]
    try:
        p0.common._git(root, "merge-base", "--is-ancestor", historical_commit, commit)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("P1 commit is not an ancestor of current P2a HEAD") from error
    if retained["repository_tree"] != p0.common._git(
        root, "rev-parse", f"{historical_commit}^{{tree}}"
    ):
        raise RuntimeError("P1 retained Git tree changed")
    if retained["protocol"] != p1.load_protocol(root):
        raise RuntimeError("P1 retained protocol changed")
    for item in retained["sources"].values():
        if p0.common.git_source_receipt(root, item["path"], historical_commit) != item:
            raise RuntimeError("P1 committed execution source changed")
    if (
        retained["source_sha256"]
        != {name: item["sha256"] for name, item in retained["sources"].items()}
        or retained["protocol_sha256"] != retained["sources"]["protocol"]["sha256"]
    ):
        raise RuntimeError("P1 execution source digest chain changed")
    if environment != retained["environment"]:
        raise RuntimeError("P2a environment differs from the pinned P1 environment")
    extraction = checked_json(root, retained["p0_extraction_lock"])
    if (
        extraction.get("status") != p0.EXTRACTION_STATUS
        or extraction["authorization"].get("model_fitting") is not False
        or extraction["manifest"]["artifacts"]["clip_index"] != retained["inputs"]["clip_index"]
        or extraction["manifest"]["artifacts"]["frame_manifest"]
        != retained["inputs"]["frame_manifest"]
    ):
        raise RuntimeError("P0/P1 immutable extraction chain changed")
    return retained


def validate_primary(
    root: Path, retained: dict[str, Any], spec: dict[str, Any]
) -> list[dict[str, str]]:
    primary, _ = p0.read_primary_cohort(root, p0.load_protocol(root))
    item = retained["inputs"]["clip_index"]
    clips = p0.csv_rows(p0.checked_bytes(root, root / item["path"], item["sha256"]))
    verified_receipt(root, item)
    verified_receipt(root, retained["inputs"]["frame_manifest"])
    if len(clips) != spec["primary_rows"] or len(primary) != len(clips):
        raise RuntimeError("P2a requires the unchanged 4977 primary rows")
    for source, clip in zip(primary, clips, strict=True):
        for key in ("sample_id", "recording_id", "label_index", "fold", "center_frame"):
            if source[key] != clip[key]:
                raise RuntimeError("Clip identity/order/labels/fold changed")
        if not p0.boolean(clip["all_frames_valid"]) or not p0.boolean(clip["center_frame_valid"]):
            raise RuntimeError("P2a requires every original row valid; no probability fallback")
    folds = p1.make_fold_map(primary, spec)
    if (
        retained["primary_rows"] != len(primary)
        or retained["fold_map_rows"] != folds
        or retained["fold_map_sha256"] != p0.canonical_digest(folds)
    ):
        raise RuntimeError("P1/P2a nested fold assignments changed")
    labels = np.array([int(row["label_index"]) for row in primary])
    outer = np.array([row["outer_fold"] for row in folds])
    for fold in range(5):
        inner = np.array([row[f"inner_fold_o{fold}"] for row in folds])
        for split in range(3):
            train = (outer != fold) & (inner != split)
            held = (outer != fold) & (inner == split)
            if set(labels[train]) != {0, 1, 2} or not held.any():
                raise RuntimeError("An inner partition lacks required three-class support")
    return primary


def validate_caches(root: Path, retained: dict[str, Any], primary: list[dict[str, str]]) -> dict:
    ids = [row["sample_id"] for row in primary]
    caches = {}
    for kind, columns, source in (("video", 2, "video_cache"), ("dino", 1, "dino_cache")):
        historical = retained["caches"][kind]
        # Pin summary and request before their contents can nominate arrays.
        verified_receipt(root, historical["summary"])
        verified_receipt(root, historical["request"])
        cache = p1.validate_feature_cache(
            root,
            (root / historical["summary"]["path"]).parent,
            kind,
            retained["protocol"],
            ids,
            np.ones((len(primary), columns), dtype=bool),
            retained["p0_extraction_lock"]["sha256"],
            retained["sources"][source]["sha256"],
        )
        if cache != historical:
            raise RuntimeError("Original frozen cache receipts changed")
        caches[kind] = cache
    features = {**caches["video"]["features"], **caches["dino"]["features"]}
    if features != retained["features"]:
        raise RuntimeError("Original P1 feature mapping changed")
    return caches


def probability_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if (
        probabilities.shape != (len(labels), 3)
        or probabilities.dtype != np.float64
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(1), 1, atol=2e-6, rtol=0)
    ):
        raise RuntimeError("P1 OOF probabilities have invalid shape/dtype/normalization")
    matrix = np.bincount(labels * 3 + probabilities.argmax(1), minlength=9).reshape(3, 3)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(
        np.divide(2 * np.diag(matrix), denominator, out=np.zeros(3), where=denominator > 0).mean()
    )


def validate_oof(
    root: Path, item: dict[str, Any], primary: list[dict[str, str]], summary: dict[str, Any]
) -> dict[str, np.ndarray]:
    raw = p0.checked_bytes(root, root / item["path"], item["sha256"])
    if len(raw) != item["size_bytes"]:
        raise RuntimeError("P1 OOF byte count changed before decoding")
    expected_models = {
        f"{arm}__{probe}" for arm in SOURCE_ARMS for probe in ("linear", "attentive")
    }
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        expected = expected_models | {
            "sample_ids",
            "recording_ids",
            "labels",
            "folds",
            "baseline_probabilities",
        }
        if set(archive.files) != expected:
            raise RuntimeError("P1 OOF members differ from the six completed arms")
        arrays = {name: archive[name] for name in archive.files}
    labels = np.array([int(row["label_index"]) for row in primary], dtype=np.int64)
    folds = np.array([int(row["fold"].split("-")[1]) for row in primary], dtype=np.int64)
    if (
        any(value.dtype.hasobject for value in arrays.values())
        or arrays["sample_ids"].dtype.kind != "U"
        or arrays["recording_ids"].dtype.kind != "U"
        or arrays["labels"].dtype != np.int64
        or arrays["folds"].dtype != np.int64
        or arrays["sample_ids"].tolist() != [row["sample_id"] for row in primary]
        or arrays["recording_ids"].tolist() != [row["recording_id"] for row in primary]
        or not np.array_equal(arrays["labels"], labels)
        or not np.array_equal(arrays["folds"], folds)
    ):
        raise RuntimeError("P1 OOF identity/order/labels/folds changed")
    for name in expected_models | {"baseline_probabilities"}:
        score = probability_f1(labels, arrays[name])
        expected_score = (
            summary["baseline"]["macro_f1"]
            if name == "baseline_probabilities"
            else summary["models"][name]["metrics"]["macro_f1"]
        )
        if abs(score - expected_score) > 1e-12:
            raise RuntimeError("P1 summary and OOF macro-F1 disagree")
    return arrays


def validate_p1_results(
    root: Path, spec: dict[str, Any], retained: dict[str, Any], primary: list[dict[str, str]]
) -> dict[str, Any]:
    inputs = spec["p1_artifacts"]
    summary = checked_json(root, inputs["summary"])
    directory = (root / inputs["summary"]["path"]).parent
    expected_models = {
        f"{arm}__{probe}" for arm in SOURCE_ARMS for probe in ("linear", "attentive")
    }
    if (
        summary.get("status") != "OKUTAMA_VIDEO_P1_EXPLORATORY_CROSSFIT_COMPLETE"
        or summary.get("rows") != len(primary)
        or summary.get("seed") != 42
        or summary.get("scenarios") != 11
        or set(summary.get("models", {})) != expected_models
        or any(summary["models"][name]["fallback_rows"] != 0 for name in expected_models)
    ):
        raise RuntimeError("P2a requires the complete original six-arm P1 results")
    for key in (
        "inner_selection_baseline_access",
        "protected_manifest_reads",
        "target_images_read",
    ):
        if summary.get(key) != 0:
            raise RuntimeError(f"P1 access accounting must be zero: {key}")
    request, request_receipt = p1.bound_json(root, directory / "request.json")
    request_hash = p0.canonical_digest(request)
    if (
        request_hash != summary["request_sha256"]
        or request.get("status") != "P1_REQUEST_BEFORE_FITTING"
        or request.get("lock_sha256") != inputs["execution_lock"]["sha256"]
        or request.get("protocol") != retained["protocol"]
        or request.get("source_sha256") != retained["sources"]["runner"]["sha256"]
        or request.get("sample_ids_sha256")
        != p0.canonical_digest([r["sample_id"] for r in primary])
    ):
        raise RuntimeError("P1 result/request/lock chain changed")
    required = {"metrics.csv", "paired_statistics.json", "oof_probabilities.npz"}
    for arm in SOURCE_ARMS:
        for probe in ("linear", "attentive"):
            for fold in range(5):
                folder = f"workloads/{arm}/{probe}/fold-{fold}/"
                checkpoint = "checkpoint.npz" if probe == "linear" else "checkpoint.pt"
                required.update(
                    folder + name for name in (checkpoint, "predictions.npz", "receipt.json")
                )
    if set(summary.get("artifacts", {})) != required:
        raise RuntimeError("P1 complete artifact inventory changed")
    artifacts = {}
    for name in sorted(required):
        path = p0.safe_path(root, directory / name)
        if not path.is_relative_to(directory.resolve()):
            raise RuntimeError("P1 artifact path escaped the retained result directory")
        artifacts[name] = p1.receipt(root, path, summary["artifacts"][name])
        if name.endswith("/receipt.json"):
            workload = checked_json(root, artifacts[name])
            _, arm, probe, fold_name, _ = name.split("/")
            fold = int(fold_name.split("-")[1])
            expected_request = {
                "arm": arm,
                "probe": probe,
                "outer_fold": fold,
                "seed": 42,
                "request_sha256": request_hash,
                "held_sample_ids": [r["sample_id"] for r in primary if r["fold"] == fold_name],
            }
            if (
                workload.get("status") != "P1_WORKLOAD_COMPLETE"
                or workload.get("request") != expected_request
            ):
                raise RuntimeError("P1 workload receipt is incomplete or has different folds")
            for artifact, digest in workload["artifacts"].items():
                relative = str(Path(name).parent / artifact).replace("\\", "/")
                if summary["artifacts"].get(relative) != digest:
                    raise RuntimeError("P1 workload artifact digest chain changed")
    if artifacts["oof_probabilities.npz"] != inputs["oof"]:
        raise RuntimeError("P1 declared OOF receipt differs from summary")
    arrays = validate_oof(root, inputs["oof"], primary, summary)
    reference = spec["p1_reference"]
    if (
        abs(
            probability_f1(arrays["labels"], arrays[reference["probabilities_member"]])
            - reference["macro_f1"]
        )
        > 1e-12
    ):
        raise RuntimeError("Pinned P1 reference result changed")
    baseline_item = retained["baseline"]
    raw = p0.checked_bytes(root, root / baseline_item["path"], baseline_item["sha256"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if not np.array_equal(
            arrays["baseline_probabilities"], archive[baseline_item["probabilities_member"]]
        ):
            raise RuntimeError("P1 retained historical baseline probabilities changed")
    return {**inputs, "request": request_receipt, "artifacts": artifacts, "workloads_completed": 30}


def randomization_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    stats = spec["statistics"]
    draws = np.random.default_rng(stats["bootstrap_seed"]).integers(
        0, 11, size=(stats["bootstrap_resamples"], 11), dtype=np.int64
    )
    signs = (2 * ((np.arange(2048)[:, None] >> np.arange(11)) & 1) - 1).astype(np.int8)
    return {
        "scenario_order": stats["scenario_order"],
        "bootstrap_indices_shape": list(draws.shape),
        "bootstrap_indices_dtype": "int64",
        "bootstrap_indices_bytes_sha256": p0.digest(draws.tobytes()),
        "swap_signs_shape": list(signs.shape),
        "swap_signs_dtype": "int8",
        "swap_signs_bytes_sha256": p0.digest(signs.tobytes()),
    }


def build_p2a_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = p0.require_clean_repository(root)
    spec = load_protocol(root)
    sources = {
        name: p0.common.git_source_receipt(root, relative, commit)
        for name, relative in spec["execution_sources"].items()
    }
    environment = p0.common.collect_environment()
    retained = validate_p1_lineage(root, spec, commit, environment)
    primary = validate_primary(root, retained, spec)
    caches = validate_caches(root, retained, primary)
    baseline = p1.validate_baseline(root, retained["protocol"], primary)
    if baseline != retained["baseline"]:
        raise RuntimeError("Original historical baseline receipt changed")
    baseline = {**baseline, "role": "comparison only; never fitting, selection or fallback"}
    evidence = validate_p1_results(root, spec, retained, primary)
    randomization = randomization_receipt(spec)
    if randomization != retained["randomization"]:
        raise RuntimeError("P1/P2a scenario randomization changed")
    # Recheck cheap source/receipt state after the expensive cache checks.
    if p0.require_clean_repository(root) != (commit, tree):
        raise RuntimeError("Repository changed during P2a locking")
    for item in sources.values():
        if p0.common.git_source_receipt(root, item["path"], commit) != item:
            raise RuntimeError("Execution source changed during P2a locking")
    for item in evidence["artifacts"].values():
        verified_receipt(root, item)
    for item in spec["p1_artifacts"].values():
        verified_receipt(root, item)
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "p1_evidence": evidence,
        "inputs": {
            **retained["inputs"],
            "baseline": baseline,
            "p1_oof": spec["p1_artifacts"]["oof"],
        },
        "baseline": baseline,
        "p1_reference": {**spec["p1_artifacts"]["oof"], **spec["p1_reference"]},
        "caches": caches,
        "features": retained["features"],
        "primary_rows": len(primary),
        "fold_map_rows": retained["fold_map_rows"],
        "fold_map_sha256": retained["fold_map_sha256"],
        "randomization": randomization,
        "environment": environment,
        "authorization": spec["authorization"],
        "access_accounting": {
            "raw_image_payloads_read": 0,
            "protected_rows_read": 0,
            "probe_fits": 0,
            "backbone_fits": 0,
            "comparison_probabilities_used_for_fitting_or_selection": 0,
            "role_safe_cache_values_validated": True,
            "historical_checkpoint_payloads_decoded": 0,
        },
    }


def validate_p2a_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    retained, _ = p0.read_lock(root, lock_path)
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected a P2a pre-fit execution lock")
    current = build_p2a_payload(root)
    p0.compare_lock(retained, current)
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("lock", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.mode == "check":
        payload = validate_p2a_lock(root, output)
    else:
        if output.exists():
            raise RuntimeError("Refusing to overwrite a retained P2a lock")
        payload = build_p2a_payload(root)
        p0.write_lock(root, output, payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload["status"],
                "rows": payload["primary_rows"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
