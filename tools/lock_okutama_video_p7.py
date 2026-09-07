"""Bind the committed CAPE protocol, frozen caches, and post-fit references."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hac.video_center import ARMS, EXPECTED_WIDTHS  # noqa: E402
from tools import lock_hac_continuation_protocols as common  # noqa: E402

PROTOCOL_PATH = "experiments/okutama_video_p7_protocol.json"
PROTOCOL_STATUS = (
    "DECLARED_AFTER_POST_P6_REVIEW_BEFORE_P7_FEATURE_DERIVATION_OR_FITTING"
)
STATUS = "OKUTAMA_VIDEO_P7_LOCKED_BEFORE_CENTER_FEATURE_DERIVATION_OR_FITTING"
AUTHORIZATION = {
    "read_named_p0_p3_frozen_caches": True,
    "derive_fixed_cape_features": True,
    "probe_fitting": True,
    "read_named_p3_p5_p6_oof_only_after_all_p7_fits": True,
    "deterministic_post_fit_probability_fusion": True,
    "feature_extraction": False,
    "backbone_fitting": False,
    "learned_fusion": False,
    "raw_image_access": False,
    "protected_data_access": False,
}
SOURCES = {
    "protocol": PROTOCOL_PATH,
    "research_plan": "docs/HAC_RESEARCH_MAP_AND_NEXT_PHASE_20260907.md",
    "locker": "tools/lock_okutama_video_p7.py",
    "runner": "experiments/run_okutama_video_p7.py",
    "center_module": "src/hac/video_center.py",
    "moment_module": "src/hac/video_token_moments.py",
    "multiscale_module": "src/hac/video_multiscale.py",
    "kinematic_types": "src/hac/video_kinematic.py",
    "consensus_module": "src/hac/video_consensus.py",
    "p5_runner": "experiments/run_okutama_video_p5.py",
    "p4_runner": "experiments/run_okutama_video_p4.py",
    "p3_runner": "experiments/run_okutama_video_p3.py",
    "p2_runner": "experiments/run_okutama_video_p2a.py",
    "p1_runner": "experiments/run_okutama_video_probe.py",
    "common_locker": "tools/lock_hac_continuation_protocols.py",
    "requirements": "requirements-video-lock.txt",
}


def expected_comparisons() -> list[dict[str, str]]:
    return [
        {"candidate": "center_signed_replacement_triad", "reference": "p6_reference"},
        {"candidate": "center_signed_factorized", "reference": "center_posture_factorized"},
        {"candidate": "center_posture_factorized", "reference": "spatial_refit"},
        {"candidate": "center_signed_factorized", "reference": "offcenter_signed_factorized"},
        {"candidate": "center_signed_factorized", "reference": "center_unsigned_factorized"},
        {"candidate": "center_signed_factorized", "reference": "center_signed_multinomial"},
    ]


def load_protocol(root: Path) -> dict[str, Any]:
    spec = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if (spec.get("protocol_version"), spec.get("status")) != (
        "1.0.0",
        PROTOCOL_STATUS,
    ):
        raise RuntimeError("Unexpected P7 protocol/version")
    probe = spec.get("probe", {})
    contracts = spec.get("arm_contracts", {})
    expected_contracts = {
        arm: (
            {"kind": "multinomial", "feature_width": width[0]}
            if width[1] is None
            else {
                "kind": "factorized",
                "posture_width": width[0],
                "motion_width": width[1],
            }
        )
        for arm, width in EXPECTED_WIDTHS.items()
    }
    recipe = spec.get("feature_recipe", {})
    signed = recipe.get("signed_change_formula", {})
    center = recipe.get("center", {})
    offcenter = recipe.get("offcenter_control", {})
    systems = spec.get("systems", {})
    triad = systems.get("center_signed_replacement_triad", {})
    references = spec.get("references", {})
    guardrails = spec.get("engineering_guardrails", {})
    expected_folds = {
        "fold-0": ["1.4", "2.2", "2.5"],
        "fold-1": ["1.5", "2.11"],
        "fold-2": ["1.10", "1.2"],
        "fold-3": ["2.7", "2.8"],
        "fold-4": ["1.11", "1.3"],
    }
    if (
        tuple(spec.get("arms", ())) != ARMS
        or spec.get("primary_expert") != "center_signed_factorized"
        or spec.get("primary_rows") != 4977
        or spec.get("primary_scenarios") != 11
        or spec.get("class_order") != ["sitting", "standing", "walking_running"]
        or spec.get("fold_contract") != expected_folds
        or contracts != expected_contracts
        or probe.get("C_values") != [1e-5, 1e-4, 1e-3, 1e-2]
        or probe.get("inner_folds") != 3
        or probe.get("inner_seed") != 42
        or probe.get("maximum_unique_estimator_fits") != 715
        or probe.get("maximum_workloads") != 30
        or spec.get("statistics", {}).get("comparisons") != expected_comparisons()
        or spec.get("statistics", {}).get("bootstrap_resamples") != 10000
        or spec.get("statistics", {}).get("bootstrap_seed") != 20260907
        or spec.get("statistics", {}).get("exact_scenario_swaps") != 2048
        or recipe.get("arithmetic_dtype") != "float32"
        or recipe.get("source_fps") != 30.0
        or recipe.get("long_source_stride_frames") != 4
        or recipe.get("short_source_stride_frames") != 1
        or recipe.get("long_fallback")
        != (
            "For each of the 467 incomplete long rows, substitute the entire exact "
            "corresponding short tensor before every CAPE operation; retain all rows "
            "and add no missingness feature."
        )
        or center.get("dino_anchor") != 8
        or center.get("dino_neighbors") != [7, 9]
        or center.get("vjepa_tubelet") != 4
        or offcenter.get("dino_anchor") != 4
        or offcenter.get("dino_neighbors") != [3, 5]
        or offcenter.get("vjepa_tubelet") != 2
        or recipe.get("anchor_formula")
        != (
            "concat(D[k]-mean_time(D), "
            "mean_region(V[q])-mean_time_region(V))"
        )
        or recipe.get("anchor_width") != 1536
        or recipe.get("signed_change_width") != 1536
        or recipe.get("direct_union_width") != 16128
        or recipe.get("direct_union")
        != (
            "concat(base_visual_posture, base_visual_motion columns after its "
            "duplicate short_v/long_v prefix, spatial_contrast, center_anchor, "
            "signed_change)"
        )
        or recipe.get("unsigned_control")
        != (
            "elementwise absolute value of both signed-change blocks after their "
            "signed derivation; preserve signed center-anchor features"
        )
        or recipe.get("spatial_contrast")
        != "reuse the exact P5 time-averaged 3x3 V-JEPA spatial-contrast derivation"
        or recipe.get("labels_used_in_feature_derivation") is not False
        or recipe.get("learned_feature_operations") is not False
        or signed.get("slope") != "(D[k+1]-D[k-1])/(2*h)"
        or signed.get("center_curvature")
        != "(D[k]-(D[k-1]+D[k+1])/2)/(h*h)"
        or signed.get("h_seconds")
        != "4/30 for long-valid rows and 1/30 for exact-short-fallback rows"
        or recipe.get("base_visual_posture")
        != ["short_v", "long_v", "short_d", "long_d"]
        or recipe.get("base_visual_motion")
        != [
            "short_v",
            "long_v",
            "v_scale",
            "short_motion",
            "long_motion",
            "d_scale",
        ]
        or references.get("p3", {}).get("post_fit_arrays")
        != ["long_vjepa_mean", "dual_scale_vjepa_dino"]
        or references.get("p5", {}).get("post_fit_arrays")
        != ["spatial_contrast_factorized", "orthogonal_moments_factorized"]
        or references.get("p5", {}).get("a0_reproduction_policy")
        != "require exact float64 OOF array equality before interpreting any P7 result"
        or references.get("p6", {}).get("reference_array")
        != "ocvc_uniform_diverse_triad"
        or references.get("p6", {}).get("reference_macro_f1")
        != 0.8258300963800421
        or references.get("p6", {}).get("reference_nll")
        != 0.41657000476156425
        or references.get("p6", {}).get("reference_brier")
        != 0.2409015437227207
        or triad.get("components")
        != [
            "P3.long_vjepa_mean",
            "P3.dual_scale_vjepa_dino",
            "P7.center_signed_factorized",
        ]
        or triad.get("learned_parameters") != 0
        or triad.get("fitted_weights") is not False
        or spec.get("authorization", {}).get("raw_image_access") is not False
        or spec.get("authorization", {}).get("protected_data_access") is not False
        or guardrails.get("primary_system_macro_f1_at_least") != 0.835
        or guardrails.get("milestone_macro_f1") != 0.84
        or guardrails.get("stretch_macro_f1") != 0.85
        or guardrails.get("primary_system_nll_strictly_below_p6") is not True
        or guardrails.get("primary_system_brier_strictly_below_p6") is not True
        or guardrails.get("minimum_scenarios_improved_over_p6") != 7
        or guardrails.get("maximum_scenario_macro_f1_decline") != 0.02
        or guardrails.get("positive_net_correct_change_overall") is not True
        or guardrails.get("positive_net_correct_change_clear_stable") is not True
    ):
        raise RuntimeError("P7 feature, fitting, or comparison contract changed")
    return spec


def _clean(root: Path) -> tuple[str, str]:
    if common._git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean committed repository is required before P7")
    commit = common._git(root, "rev-parse", "HEAD")
    return commit, common._git(root, "rev-parse", "HEAD^{tree}")


def _path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    common.assert_role_safe_input_path(root, path)
    return path


def _artifact_receipt(root: Path, declared: dict[str, Any]) -> dict[str, Any]:
    path = _path(root, declared["path"])
    digest, size = common._sha256_file(path)
    if digest != declared["sha256"] or size != int(declared["size_bytes"]):
        raise RuntimeError(f"P7 source artifact receipt changed: {declared['path']}")
    return {
        "path": common._relative_path(root, path),
        "sha256": digest,
        "size_bytes": size,
    }


def _postfit_reference_receipts(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in ("p3", "p5", "p6"):
        declared = spec["references"][phase]
        summary_receipt = _artifact_receipt(root, declared["summary"])
        oof_receipt = _artifact_receipt(root, declared["oof"])
        summary = json.loads(
            _path(root, declared["summary"]["path"]).read_text(encoding="utf-8")
        )
        if summary.get("status") != declared["summary"]["status"]:
            raise RuntimeError(f"Unexpected {phase.upper()} reference status")
        if summary.get("artifacts", {}).get("oof_probabilities.npz") != oof_receipt["sha256"]:
            raise RuntimeError(f"{phase.upper()} summary does not bind its OOF artifact")
        output[phase] = {
            "summary": summary_receipt,
            "oof": oof_receipt,
            "post_fit_arrays": declared.get("post_fit_arrays", []),
            "reference_array": declared.get("reference_array"),
            "status": declared["summary"]["status"],
        }
    return output


def _upstream(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    declared = spec["references"]["p5"]["execution_lock"]
    receipt = _artifact_receipt(root, declared)
    retained = json.loads(_path(root, declared["path"]).read_text(encoding="utf-8"))
    if retained.get("status") != "OKUTAMA_VIDEO_P5_LOCKED_BEFORE_MOMENT_DERIVATION_OR_FITTING":
        raise RuntimeError("Expected the retained P5 pre-fit execution lock")
    required = {
        "p3_long_caches",
        "p3_reference",
        "p3_fold_map_sha256",
        "p3_randomization",
        "sample_ids_sha256",
    }
    if not required.issubset(retained):
        raise RuntimeError("Retained P5 lineage is incomplete")
    return receipt, retained


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit, tree = _clean(root)
    spec = load_protocol(root)
    sources = {
        name: common.git_source_receipt(root, path, commit) for name, path in SOURCES.items()
    }
    upstream_receipt, upstream = _upstream(root, spec)
    execution_locks = {
        "p3_probe": _artifact_receipt(
            root, spec["references"]["p3"]["probe_lock"]
        ),
        "p5": upstream_receipt,
        "p6": _artifact_receipt(root, spec["references"]["p6"]["execution_lock"]),
    }
    p6_lock = json.loads(
        _path(root, spec["references"]["p6"]["execution_lock"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if p6_lock.get("status") != "OKUTAMA_VIDEO_P6_LOCKED_BEFORE_DETERMINISTIC_REPLAY":
        raise RuntimeError("Expected the retained P6 execution lock")
    return {
        "status": STATUS,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "repository_tree": tree,
        "protocol": spec,
        "protocol_sha256": sources["protocol"]["sha256"],
        "sources": sources,
        "source_sha256": {name: item["sha256"] for name, item in sources.items()},
        "execution_locks": execution_locks,
        "postfit_references": _postfit_reference_receipts(root, spec),
        "p3_long_caches": upstream["p3_long_caches"],
        "p3_reference": upstream["p3_reference"],
        "p3_fold_map_sha256": upstream["p3_fold_map_sha256"],
        "p3_randomization": upstream["p3_randomization"],
        "sample_ids_sha256": upstream["sample_ids_sha256"],
        "authorization": AUTHORIZATION.copy(),
        "environment": common.collect_environment(),
        "access_accounting": {
            "center_feature_derivations": 0,
            "model_fits": 0,
            "postfit_probability_rows_decoded": 0,
            "postfit_probability_fusions": 0,
            "raw_images_read": 0,
            "protected_rows_read": 0,
        },
    }


def compare(retained: dict[str, Any], current: dict[str, Any]) -> None:
    left = {key: value for key, value in retained.items() if key != "locked_at_utc"}
    right = {key: value for key, value in current.items() if key != "locked_at_utc"}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise RuntimeError("Retained P7 lock changed: " + ", ".join(changed))


def write_lock(root: Path, path: Path, payload: dict[str, Any]) -> None:
    path = _path(root, path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing P7 lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_lock(root: Path, path: Path) -> dict[str, Any]:
    retained = json.loads(_path(root, path).read_text(encoding="utf-8"))
    if retained.get("status") != STATUS or retained.get("authorization") != AUTHORIZATION:
        raise RuntimeError("Expected the P7 pre-fit execution lock")
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
                "rows": 4977,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
