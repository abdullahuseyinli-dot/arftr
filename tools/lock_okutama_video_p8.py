"""Bind completed CCAC extraction and the prospectively fixed supervised P8 trial."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.ccac_features import (  # noqa: E402
    ARMS,
    COMPENSATED_TRANSLATION_COLUMNS,
    EXPECTED_WIDTHS,
    QUALITY_COLUMNS,
    RAW_TRANSLATION_COLUMNS,
    WITHIN_ACTOR_COLUMNS,
)
from tools import lock_hac_continuation_protocols as common  # noqa: E402
from tools import lock_okutama_video_p7 as previous  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p8_protocol.json"
PROTOCOL_STATUS = "DECLARED_AFTER_CCAC_EXTRACTION_BEFORE_P8_SUPERVISED_FITTING"
STATUS = "OKUTAMA_VIDEO_P8_LOCKED_BEFORE_SUPERVISED_FITTING"
AUTHORIZATION = {
    "read_named_p0_p3_frozen_caches": True,
    "read_completed_label_free_ccac_features": True,
    "probe_fitting": True,
    "read_named_p3_p5_p6_oof_only_after_all_p8_fits": True,
    "deterministic_post_fit_probability_fusion": True,
    "raw_image_access": False,
    "feature_extraction": False,
    "backbone_fitting": False,
    "learned_fusion": False,
    "protected_data_access": False,
}
SOURCES = {
    **previous.SOURCES,
    "protocol": PROTOCOL_PATH,
    "locker": "tools/lock_okutama_video_p8.py",
    "runner": "experiments/run_okutama_video_p8.py",
    "ccac_features": "src/hac/ccac_features.py",
    "p7_protocol": "experiments/okutama_video_p7_protocol.json",
    "p7_locker": "tools/lock_okutama_video_p7.py",
    "p7_runner": "experiments/run_okutama_video_p7.py",
    "p8_decision": "docs/HAC_P7_RESULTS_AND_P8_DECISION_20260907.md",
    "tests": "tests/test_okutama_video_p8.py",
}
FULL_NAMES = {"execution_lock", "request", "summary", "completion", "feature_manifest", "features"}
_path = previous._path
_artifact_receipt = previous._artifact_receipt


def expected_comparisons() -> list[dict[str, str]]:
    return [
        {"candidate": candidate, "reference": reference}
        for candidate, reference in (
            ("ccac_replacement_triad", "p6_reference"),
            ("ccac_full", "spatial_refit"),
            ("ccac_full", "ccac_compensated_translation"),
            ("ccac_compensated_translation", "ccac_raw_translation"),
            ("ccac_full", "ccac_reliability"),
            ("ccac_raw_translation", "spatial_refit"),
        )
    ]


def validate_spec(spec: dict[str, Any], *, require_bound: bool = True) -> None:
    probe, statistics = spec["probe"], spec["statistics"]
    if (
        spec.get("protocol_version") != "1.0.0"
        or spec.get("status") != PROTOCOL_STATUS
        or spec.get("declared_on") != "2026-09-07"
        or spec.get("seed") != 42
        or spec.get("primary_rows") != 4977
        or spec.get("primary_scenarios") != 11
        or spec.get("class_order") != ["sitting", "standing", "walking_running"]
        or tuple(spec.get("arms", ())) != ARMS
        or spec.get("primary_expert") != "ccac_full"
        or probe.get("C_values") != [1e-5, 1e-4, 1e-3, 1e-2]
        or probe.get("solver") != "lbfgs"
        or probe.get("class_weight") != "balanced"
        or probe.get("max_iter") != 2000
        or probe.get("tolerance") != 0.0001
        or probe.get("inner_folds") != 3
        or probe.get("inner_seed") != 42
        or probe.get("inner_splitter") != "StratifiedGroupKFold"
        or probe.get("inner_shuffle") is not True
        or probe.get("standardize") is not True
        or probe.get("threshold_tuning") is not False
        or probe.get("backbones_frozen") is not True
        or probe.get("maximum_unique_estimator_fits") != 470
        or probe.get("maximum_estimator_fit_invocations") != 470
        or probe.get("maximum_workloads") != 25
        or statistics.get("comparisons") != expected_comparisons()
        or statistics.get("bootstrap_resamples") != 10000
        or statistics.get("bootstrap_seed") != 20260907
        or statistics.get("exact_scenario_swaps") != 2048
    ):
        raise RuntimeError("P8 prospective fitting/statistical contract changed")
    expected_contracts = {
        arm: {
            "kind": "factorized",
            "posture_width": widths[0],
            "motion_width": widths[1],
            "fits_per_fold": 26 if arm == "spatial_refit" else 17,
        }
        for arm, widths in EXPECTED_WIDTHS.items()
    }
    if spec["arm_contracts"] != expected_contracts:
        raise RuntimeError("P8 arm widths or fit counts changed")
    system = spec["systems"]["ccac_replacement_triad"]
    if (
        set(spec["systems"]) != {"p6_reference", "ccac_replacement_triad"}
        or system["components"]
        != ["P3.long_vjepa_mean", "P3.dual_scale_vjepa_dino", "P8.ccac_full"]
        or system["learned_parameters"] != 0
        or system["fitted_weights"] is not False
        or system["primary_system"] is not True
    ):
        raise RuntimeError("P8 fixed system changed")
    if set(spec.get("full_extraction", {})) != FULL_NAMES:
        raise RuntimeError("Incomplete P8 full-extraction receipt inventory")
    if require_bound:
        for name, item in spec["full_extraction"].items():
            if (
                re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
                or item.get("size_bytes", -1) <= 0
            ):
                raise RuntimeError(f"Unset or invalid full extraction receipt: {name}")


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    validate_spec(spec)
    old = previous.load_protocol(root)
    expected_refs = json.loads(json.dumps(old["references"]))
    expected_refs["p5"]["a0_reproduction_policy"] = (
        "require exact float64 OOF array equality before interpreting any P8 result"
    )
    if spec["references"] != expected_refs or spec["fold_contract"] != old["fold_contract"]:
        raise RuntimeError("P8 historical references or scenario folds changed")
    expected_guards = {
        key: value
        for key, value in old["engineering_guardrails"].items()
        if key != "mechanism_claim"
    }
    actual_guards = {
        key: value
        for key, value in spec["engineering_guardrails"].items()
        if key != "mechanism_claim"
    }
    if actual_guards != expected_guards:
        raise RuntimeError("P8 engineering guardrails changed")
    expected_authorization = {
        "read_named_p0_p3_frozen_caches_after_separate_lock": True,
        "read_completed_label_free_ccac_features_after_separate_lock": True,
        "probe_fitting_after_separate_lock": True,
        "read_named_p3_p5_p6_oof_only_after_all_p8_fits": True,
        "deterministic_post_fit_probability_fusion": True,
        "raw_image_access": False,
        "feature_extraction": False,
        "backbone_fitting": False,
        "learned_fusion": False,
        "protected_data_access": False,
        "external_confirmation_claim": False,
    }
    if spec["authorization"] != expected_authorization:
        raise RuntimeError("P8 authorization changed")
    return spec


def _full_extraction(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    declared = spec["full_extraction"]
    receipts = {name: _artifact_receipt(root, item) for name, item in declared.items()}
    payloads = {
        name: json.loads(_path(root, declared[name]["path"]).read_text(encoding="utf-8"))
        for name in FULL_NAMES - {"features"}
    }
    full_lock, request = payloads["execution_lock"], payloads["request"]
    summary, completion, manifest = (
        payloads["summary"],
        payloads["completion"],
        payloads["feature_manifest"],
    )
    request_sha = common._sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    if (
        full_lock.get("status") != "OKUTAMA_CCAC_FULL_LOCKED_BEFORE_NONPILOT_PIXEL_ACCESS"
        or summary.get("status") != "OKUTAMA_CCAC_FULL_EXTRACTION_COMPLETE"
        or manifest.get("status") != "OKUTAMA_CCAC_FULL_FEATURES_COMPLETE"
        or summary.get("rows") != 4977
        or manifest.get("rows") != 4977
        or summary.get("requested_pairs") != 74655
        or summary.get("pilot_reused_workloads") != 128
        or summary.get("new_pixel_workloads") != 4849
        or summary.get("execution_lock_sha256") != receipts["execution_lock"]["sha256"]
        or summary.get("request_sha256") != request_sha
        or request.get("lock_sha256") != receipts["execution_lock"]["sha256"]
        or completion
        != {
            "status": "OKUTAMA_CCAC_FULL_PUBLICATION_COMPLETE",
            "request_sha256": request_sha,
            "summary_sha256": receipts["summary"]["sha256"],
            "features_sha256": receipts["features"]["sha256"],
        }
        or any(
            summary.get(key) != 0
            for key in (
                "action_labels_read",
                "prediction_rows_read",
                "classifier_fits",
                "protected_rows_read",
            )
        )
        or manifest.get("labels_present") is not False
        or manifest.get("predictions_present") is not False
        or manifest.get("sample_ids_sha256") != full_lock.get("selected_sample_ids_sha256")
        or manifest.get("measurement_source_sha256") != full_lock.get("measurement_source_sha256")
    ):
        raise RuntimeError("Full CCAC extraction has not completed with the declared lineage")
    replay = summary.get("nonpilot_exact_replay", [])
    if len(replay) != 2 or any(
        not row.get(key)
        for row in replay
        for key in ("pair_metrics_exact", "clip_metrics_exact", "correspondence_bytes_exact")
    ):
        raise RuntimeError("Full CCAC exact replay evidence missing")
    expected_columns = {
        "quality": list(QUALITY_COLUMNS),
        "raw_translation": list(RAW_TRANSLATION_COLUMNS),
        "compensated_translation": list(COMPENSATED_TRANSLATION_COLUMNS),
        "within_actor": list(WITHIN_ACTOR_COLUMNS),
    }
    if (
        manifest.get("columns") != expected_columns
        or manifest.get("quantiles") != [0.1, 0.5, 0.9]
        or manifest.get("quantile_method") != "linear"
    ):
        raise RuntimeError("Full CCAC feature columns/quantiles changed")
    output = _path(root, declared["summary"]["path"]).parent
    for name, digest in summary.get("artifacts", {}).items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or common._sha256_file(path)[0] != digest:
            raise RuntimeError(f"Full CCAC publication artifact changed: {name}")
    for name in ("features", "feature_manifest"):
        filename = _path(root, declared[name]["path"]).name
        if summary.get("artifacts", {}).get(filename) != receipts[name]["sha256"]:
            raise RuntimeError("Full CCAC summary does not bind its features")
    observed = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if observed != set(summary["artifacts"]) | {"request.json", "summary.json", "completion.json"}:
        raise RuntimeError("Full CCAC publication inventory changed")
    # Reused pilot evidence remains a live provenance dependency, not just copied hashes.
    pilot_declared = full_lock["protocol"]["pilot_authorization"]
    pilot_receipts = {
        name: _artifact_receipt(root, pilot_declared[name])
        for name in ("execution_lock", "request", "summary", "completion")
    }
    pilot_output = _path(root, pilot_declared["summary"]["path"]).parent
    pilot_summary = json.loads((pilot_output / "summary.json").read_text(encoding="utf-8"))
    pilot_completion = json.loads((pilot_output / "completion.json").read_text(encoding="utf-8"))
    if (
        pilot_summary.get("status") != "OKUTAMA_CCAC_PILOT_COMPLETE"
        or pilot_completion.get("status") != "OKUTAMA_CCAC_PILOT_PUBLICATION_COMPLETE"
        or pilot_completion.get("summary_sha256") != pilot_receipts["summary"]["sha256"]
    ):
        raise RuntimeError("Reused CCAC pilot publication changed")
    for name, digest in pilot_summary.get("artifacts", {}).items():
        path = (pilot_output / name).resolve()
        if not path.is_relative_to(pilot_output) or common._sha256_file(path)[0] != digest:
            raise RuntimeError(f"Reused CCAC pilot artifact changed: {name}")
    # Validate archival source receipts at their own commit, never rebuild the old HEAD lock.
    for name, receipt in full_lock["sources"].items():
        if (
            common.git_source_receipt(root, receipt["path"], full_lock["repository_commit"])
            != receipt
        ):
            raise RuntimeError(f"Full CCAC archival source changed: {name}")
    if (
        full_lock["sources"]["feature_module"]["sha256"]
        != common._sha256_file(root / "src/hac/ccac_features.py")[0]
    ):
        raise RuntimeError("Supervised CCAC schema differs from full extraction")
    return {
        "receipts": receipts,
        "feature_columns": expected_columns,
        "sample_ids_sha256": manifest["sample_ids_sha256"],
        "extraction_repository_commit": full_lock["repository_commit"],
        "artifact_count": len(summary["artifacts"]),
        "pilot_artifact_count": len(pilot_summary["artifacts"]),
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = previous._clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit) for name, path in SOURCES.items()
    }
    upstream_receipt, upstream = previous._upstream(root, spec)
    full = _full_extraction(root, spec)
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: entry["sha256"] for name, entry in sources.items()},
        "execution_locks": {
            "p3_probe": _artifact_receipt(root, spec["references"]["p3"]["probe_lock"]),
            "p5": upstream_receipt,
            "p6": _artifact_receipt(root, spec["references"]["p6"]["execution_lock"]),
        },
        "postfit_references": previous._postfit_reference_receipts(root, spec),
        "full_extraction": full,
        **{
            name: upstream[name]
            for name in (
                "p3_long_caches",
                "p3_reference",
                "p3_fold_map_sha256",
                "p3_randomization",
                "sample_ids_sha256",
            )
        },
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "model_fits": 0,
            "ccac_feature_rows_decoded": 0,
            "postfit_probability_rows_decoded": 0,
            "raw_images_read": 0,
            "protected_rows_read": 0,
        },
    }


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected P8 supervised pre-fit lock")
    previous.compare(retained, build_payload(root))
    return retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = validate_lock(_ROOT, args.output) if args.check else build_payload(_ROOT)
    if not args.check:
        previous.write_lock(_ROOT, args.output, payload)
    print(json.dumps({"status": payload["status"], "model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
