"""Read-only, label-blind body-witness dependency planning.

This is not a runner. In particular, historical StratifiedGroupKFold ancestors
cannot be chosen without labels. The exact *policy* exclusion DAG is enumerated;
historical fit counts are conservative ceilings and structural feasibility is
checked without loading labels, scores, supports, or error-bank membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path("experiments/okutama_body_witness_protocol.json")
SAFE_METADATA_FIELDS = ("sample_ids", "scenarios", "folds", "recordings", "tracks")
POLICY_PREFIX = "hac-body-policy-v1|"
PILOT_PREFIX = "hac-body-witness-v1|"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate protocol JSON key: {key}")
        result[key] = value
    return result


def load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        protocol = json.load(stream, object_pairs_hook=_unique_json_object)
    if protocol["nested_integration"]["policy_block_prefix"] != POLICY_PREFIX:
        raise ValueError("Policy salt differs from the prospective v1 contract")
    if protocol["nested_integration"]["policy_blocks"] != 3:
        raise ValueError("The prospective v1 policy uses exactly three blocks")
    if protocol["pilot"]["sample_hash_prefix"] != PILOT_PREFIX:
        raise ValueError("Pilot salt differs from the prospective v1 contract")
    if protocol["pilot"]["selection"]["salt"] != PILOT_PREFIX:
        raise ValueError("Pilot selection salt alias differs from the prospective v1 contract")
    if protocol["observations"]["spatial_masks"]["excluded_band_fraction"] != protocol["verifier"]["excluded_middle_band_height_fraction"]:
        raise ValueError("Observation and verifier mask definitions differ")
    if protocol["population"]["classes"] != ["sitting", "standing", "walking_running"]:
        raise ValueError("Canonical class order changed")
    return protocol


def _string_set(values: Iterable[str]) -> tuple[str, ...]:
    values = tuple(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Scenario identifiers must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("Scenario populations must not contain duplicate identifiers")
    return tuple(sorted(values))


def policy_blocks(scenarios: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Recompute fixed hash-order round-robin blocks on this exact population."""
    values = _string_set(scenarios)
    if len(values) < 3:
        raise ValueError("Three nonempty policy blocks require at least three scenarios")
    ranked = sorted(
        values,
        key=lambda scenario: (
            hashlib.sha256((POLICY_PREFIX + scenario).encode("utf-8")).hexdigest(),
            scenario,
        ),
    )
    return tuple(tuple(ranked[index::3]) for index in range(3))


def load_safe_metadata(path: Path) -> dict[str, np.ndarray]:
    """Only these five members are indexed; NPZ labels are never read."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in SAFE_METADATA_FIELDS}


def validate_metadata(metadata: Mapping[str, np.ndarray], protocol: Mapping) -> None:
    rows = protocol["population"]["rows"]
    for key in SAFE_METADATA_FIELDS:
        value = np.asarray(metadata[key])
        if value.shape != (rows,):
            raise ValueError(f"Canonical {key} shape mismatch: {value.shape}")
        if key != "folds" and value.dtype.kind not in "US":
            raise ValueError(f"Canonical {key} must be strings, never numeric coercions")
    ids = metadata["sample_ids"].tolist()
    if len(set(ids)) != rows or any(not value for value in ids):
        raise ValueError("Canonical sample IDs are empty or duplicated")
    if canonical_hash(ids) != protocol["population"]["sample_ids_sha256"]:
        raise ValueError("Canonical sample ID order/hash mismatch")
    expected = protocol["population"]["outer_scenarios"]
    all_scenarios = [value for fold in expected.values() for value in fold]
    if len(all_scenarios) != len(set(all_scenarios)):
        raise ValueError("A scenario occurs in multiple outer folds")
    if set(metadata["scenarios"].tolist()) != set(all_scenarios):
        raise ValueError("Canonical scenario set differs from the pinned outer folds")
    if len(set(all_scenarios)) != protocol["population"]["scenario_count"]:
        raise ValueError("Scenario count mismatch")
    if metadata["folds"].dtype.kind not in "iu":
        raise ValueError("Fold IDs must be integers")
    if set(metadata["folds"].tolist()) != {int(fold) for fold in expected}:
        raise ValueError("Fold set differs from the pinned outer folds")
    for fold, scenarios in expected.items():
        observed = set(metadata["scenarios"][metadata["folds"] == int(fold)].tolist())
        if observed != set(scenarios):
            raise ValueError(f"Outer fold {fold} has incorrect scenario membership")


def select_pilot(metadata: Mapping[str, np.ndarray], protocol: Mapping) -> dict:
    """Return exact label-blind IDs, with track diversity on the first pass."""
    ids = metadata["sample_ids"]
    scenarios = metadata["scenarios"]
    settings = protocol["pilot"]
    quota_settings = settings["quotas"]
    selected: list[int] = []
    records = []
    for index, scenario in enumerate(sorted(set(scenarios.tolist()))):
        quota = (
            quota_settings["first_quota"]
            if index < quota_settings["first_scenarios"]
            else quota_settings["remaining_quota"]
        )
        ranked = sorted(
            np.flatnonzero(scenarios == scenario).tolist(),
            key=lambda row: (
                hashlib.sha256((PILOT_PREFIX + str(ids[row])).encode("utf-8")).hexdigest(),
                str(ids[row]),
            ),
        )
        if len(ranked) < quota:
            raise ValueError(f"Scenario {scenario} has fewer than {quota} pilot centers")
        first_pass: list[int] = []
        seen_tracks: set[tuple[str, str]] = set()
        for row in ranked:
            track = (str(metadata["recordings"][row]), str(metadata["tracks"][row]))
            if track not in seen_tracks:
                first_pass.append(row)
                seen_tracks.add(track)
                if len(first_pass) == quota:
                    break
        chosen = list(first_pass)
        chosen_set = set(chosen)
        for row in ranked:
            if len(chosen) == quota:
                break
            if row not in chosen_set:
                chosen.append(row)
                chosen_set.add(row)
        selected.extend(chosen)
        records.append({"scenario": scenario, "quota": quota, "first_pass_unique_tracks": len(first_pass), "sample_ids": ids[chosen].tolist()})
    if len(selected) != settings["rows"] or len(set(selected)) != len(selected):
        raise ValueError("Pilot size or uniqueness differs from the prospective contract")
    timing = sorted(
        selected,
        key=lambda row: (
            hashlib.sha256((PILOT_PREFIX + str(ids[row])).encode("utf-8")).hexdigest(),
            str(ids[row]),
        ),
    )[: settings["timing_centers"]]
    return {
        "selection_uses_labels_or_predictions": False,
        "sample_ids": ids[selected].tolist(),
        "sample_ids_sha256": canonical_hash(ids[selected].tolist()),
        "by_scenario": records,
        "timing_first_16_global_sample_hash_ids": ids[timing].tolist(),
        "human_review_and_all_quality_gates": "not_executed",
    }


def population_summary(scenarios: Iterable[str], metadata: Mapping[str, np.ndarray]) -> dict:
    population = _string_set(scenarios)
    rows = np.flatnonzero(np.isin(metadata["scenarios"], population))
    return {
        "scenarios": list(population),
        "scenario_count": len(population),
        "canonical_row_count": len(rows),
        "canonical_row_indices_sha256": canonical_hash(rows.tolist()),
        "sample_ids_sha256": canonical_hash(metadata["sample_ids"][rows].tolist()),
        "body_reader_posture_training_rows": {"minimum_label_blind": 0, "maximum_label_blind": len(rows), "exact": None},
    }


def validate_ancestor_exclusions(plan: Mapping) -> None:
    """Reject target/outer/calibration contamination, including policy ancestors."""
    populations = {node["id"]: node for node in plan["F_populations"]}
    requests = {node["id"]: node for node in plan["prediction_requests"]}
    for request in requests.values():
        train = set(populations[request["F_population"]]["scenarios"])
        if train & set(request["forbidden_scenarios"]):
            raise ValueError(f"Forbidden scenario in ancestor of {request['id']}")
        if train & set(request["prediction_scenarios"]):
            raise ValueError(f"Prediction target was in training: {request['id']}")
    for procedure in plan["policy_training_procedures"]:
        forbidden = set(procedure["forbidden_scenarios"])
        targets: list[str] = []
        for request_id in procedure["oof_request_ids"]:
            request = requests[request_id]
            ancestor = set(populations[request["F_population"]]["scenarios"])
            if ancestor & forbidden or set(request["prediction_scenarios"]) & forbidden:
                raise ValueError(f"Calibration/outer contamination in {procedure['id']}")
            targets.extend(request["prediction_scenarios"])
        if len(targets) != len(set(targets)) or set(targets) != set(procedure["training_scenarios"]):
            raise ValueError(f"Policy cross-fit coverage fails: {procedure['id']}")


def build_dependency_dag(metadata: Mapping[str, np.ndarray], protocol: Mapping) -> dict:
    all_scenarios = set(metadata["scenarios"].tolist())
    populations: dict[str, dict] = {}
    requests: list[dict] = []
    procedures: list[dict] = []
    outer_nodes: list[dict] = []

    def add_population(scenarios: Iterable[str]) -> str:
        summary = population_summary(scenarios, metadata)
        key = "F_" + canonical_hash(summary["scenarios"])[:16]
        if key not in populations:
            summary["id"] = key
            summary["historical_ARFTR_structurally_infeasible"] = summary["scenario_count"] < protocol["nested_integration"]["historical_structural_min_scenarios_necessary"]
            summary["historical_label_class_feasibility"] = "unknown_not_inspected"
            populations[key] = summary
        elif populations[key]["scenarios"] != summary["scenarios"]:
            raise ValueError("Population key collision")
        return key

    def add_request(identifier: str, role: str, fold: int, train: set[str], target: set[str], forbidden: set[str]) -> str:
        summary = population_summary(target, metadata)
        requests.append({
            "id": identifier, "role": role, "outer_fold": fold,
            "F_population": add_population(train),
            "prediction_scenarios": summary["scenarios"],
            "prediction_row_count": summary["canonical_row_count"],
            "prediction_sample_ids_sha256": summary["sample_ids_sha256"],
            "forbidden_scenarios": sorted(forbidden),
        })
        return identifier

    for fold_string, held_list in sorted(protocol["population"]["outer_scenarios"].items(), key=lambda item: int(item[0])):
        fold = int(fold_string)
        held = set(held_list)
        train = all_scenarios - held
        blocks = policy_blocks(train)
        final_request = add_request(f"outer{fold}_final_F", "final_outer_prediction", fold, train, held, held)
        top_policy = {"id": f"outer{fold}_final_policy", "outer_fold": fold, "role": "final_policy_training", "training_scenarios": sorted(train), "forbidden_scenarios": sorted(held), "oof_request_ids": []}
        calibration_procedures = []
        calibration_requests = []
        for block_index, block in enumerate(blocks):
            excluded = set(block)
            reduced = train - excluded
            top_policy["oof_request_ids"].append(add_request(f"outer{fold}_policy_I{block_index}", "outer_policy_cross_fit", fold, reduced, excluded, held | excluded))
            calibration_requests.append(add_request(f"outer{fold}_calibration_J{block_index}_F", "policy_calibration_prediction", fold, reduced, excluded, held | excluded))
            procedure = {"id": f"outer{fold}_calibration_J{block_index}_policy", "outer_fold": fold, "role": "calibration_policy_training", "training_scenarios": sorted(reduced), "forbidden_scenarios": sorted(held | excluded), "oof_request_ids": []}
            for inner_index, inner_block in enumerate(policy_blocks(reduced)):
                inner_held = set(inner_block)
                procedure["oof_request_ids"].append(add_request(f"outer{fold}_calibration_J{block_index}_policy_I{inner_index}", "calibration_policy_cross_fit", fold, reduced - inner_held, inner_held, held | excluded | inner_held))
            procedures.append(procedure)
            calibration_procedures.append(procedure["id"])
        procedures.append(top_policy)
        outer_nodes.append({"outer_fold": fold, "held_scenarios": sorted(held), "training_scenarios": sorted(train), "blocks_recomputed_in_S": [list(block) for block in blocks], "final_F_request": final_request, "final_policy_procedure": top_policy["id"], "calibration_F_requests": calibration_requests, "calibration_policy_procedures": calibration_procedures})

    result = {"outer_folds": outer_nodes, "F_populations": sorted(populations.values(), key=lambda node: (node["scenario_count"], node["scenarios"])), "prediction_requests": requests, "policy_training_procedures": procedures}
    validate_ancestor_exclusions(result)
    result["ancestor_exclusions_checked"] = True
    return result


def fit_budget(dag: Mapping, protocol: Mapping, route: str) -> dict:
    """Ceilings describe requests, not feasible work or certified reuse."""
    populations = dag["F_populations"]
    unique_F = len(populations)
    unique_policy = len({tuple(node["training_scenarios"]) for node in dag["policy_training_procedures"]})
    recipe = protocol["nested_integration"]["historical_recipe_budget"]
    m4_per_F = recipe["M4_configs"] * recipe["selection_folds_max"] + recipe["refit_seeds"]
    a3_per_F = recipe["A3_configs"] * recipe["selection_folds_max"] + recipe["refit_seeds"]
    base_per_population = recipe["P6_estimator_streams"] * (recipe["P6_C_candidates"] * recipe["P6_inner_folds_max"] + 1)
    base_population_calls = unique_F * recipe["P6_training_population_calls_per_F_max"]
    reader_arms = ["V2", "V3"] if route == "V3" else ["V2"]
    policy_arms = protocol["nested_integration"]["routes"][route]
    original_outer_sets = {tuple(node["training_scenarios"]) for node in dag["outer_folds"]}
    original_outer_count = sum(tuple(node["scenarios"]) in original_outer_sets for node in populations)
    return {
        "route": route,
        "exact_policy_layer_counts": {
            "F_prediction_requests": len(dag["prediction_requests"]),
            "F_population_occurrences_after_within_outer_dedup": len({(request["outer_fold"], request["F_population"]) for request in dag["prediction_requests"]}),
            "distinct_F_training_scenario_populations": unique_F,
            "F_population_size_histogram": dict(sorted(Counter(str(node["scenario_count"]) for node in populations).items(), key=lambda item: int(item[0]))),
            "policy_training_procedure_occurrences": len(dag["policy_training_procedures"]),
            "distinct_policy_training_scenario_populations": unique_policy,
            "reader_arms": reader_arms,
            "policy_arms": policy_arms,
            "policy_estimator_fits_without_reuse_max": len(dag["policy_training_procedures"]) * len(policy_arms),
            "policy_estimator_fits_after_exact_population_reuse_max": unique_policy * len(policy_arms),
        },
        "conditional_recipe_upper_bounds_not_executable_counts": {
            "M4_neural_fits_per_F": m4_per_F,
            "A3_neural_fits_per_F": a3_per_F,
            "M4_and_A3_neural_fits": unique_F * (m4_per_F + a3_per_F),
            "P6_population_requests_before_transitive_dedup": base_population_calls,
            "P6_distinct_population_count_max": base_population_calls,
            "P6_estimator_fits_per_population_max": base_per_population,
            "P6_estimator_fits_max": base_population_calls * base_per_population,
            "body_reader_fits": unique_F * len(reader_arms) * recipe["refit_seeds"],
            "ARFTR_coefficient_evaluations_not_training_fits": unique_F * recipe["ARFTR_coefficient_candidates_per_F"],
            "historical_inner_exact_fit_count": None,
            "reason_exact_inner_count_is_unavailable": "Historical stratified inner populations depend on labels; planner does not read labels. Required F populations are already structurally infeasible.",
        },
        "reuse": {
            "original_outer_F_population_matches": original_outer_count,
            "existing_anchor_original_outer_results_potentially_reusable": original_outer_count,
            "confirmed_screen_reader_fits_potentially_reusable": original_outer_count * len(reader_arms) * recipe["refit_seeds"],
            "task_fits_certified_reusable_by_this_planner": 0,
            "reuse_requirements": protocol["nested_integration"]["reuse"],
            "frozen_encoder_reuse_does_not_validate_task_trained_ancestors": True,
            "same_training_set_different_prediction_batch": "Reuse weights only after exact receipt checks; regenerate and independently audit the required prediction batch. Identical scenario sets do not promise byte-identical cached predictions.",
        },
        "screen_fit_counts": {
            "initial_all_four": 20,
            "initial_target_sensitivity_excludes_V3": 15,
            "V3_confirmation_additional": 40,
            "V3_confirmed_screen_total": 60,
            "V3_mechanism_control_additional": 30,
            "V3_screen_and_mechanism_total": 90,
            "V2_confirmation_additional": 30,
            "V2_total_if_V3_initially_fitted": 50,
            "V2_total_if_V3_not_initially_fitted": 45,
        },
    }


def handoff_decision(projected_seconds: float | None, protocol: Mapping) -> dict:
    boundary = protocol["compute"]["training_job_max_seconds"]
    if projected_seconds is not None and (not np.isfinite(projected_seconds) or projected_seconds < 0):
        raise ValueError("Projected job seconds must be finite and nonnegative")
    return {
        "training_job_boundary_seconds": boundary,
        "projected_job_seconds": projected_seconds,
        "handoff_required": projected_seconds is None or projected_seconds > boundary,
        "projection_status": "unmeasured_no_full_launch" if projected_seconds is None else ("above_boundary_prepare_and_handoff" if projected_seconds > boundary else "within_time_boundary_not_execution_authority"),
        "next_if_above_boundary": protocol["compute"]["above_boundary"],
        "this_planner_can_launch_training": False,
        "per_job_boundary_is_not_total_study_budget": True,
        "nested_integration_cannot_launch_even_if_time_is_small": "hard structural ancestor failure takes precedence",
    }


def build_plan(metadata: Mapping[str, np.ndarray], protocol: Mapping, *, route: str = "V3", projected_job_seconds: float | None = None) -> dict:
    if route not in ("V3", "V2_only"):
        raise ValueError("Route must be V3 or V2_only; V1 cannot be promoted")
    validate_metadata(metadata, protocol)
    dag = build_dependency_dag(metadata, protocol)
    infeasible = [node for node in dag["F_populations"] if node["historical_ARFTR_structurally_infeasible"]]
    blocker = {
        "code": "HISTORICAL_ARFTR_NESTING_INSUFFICIENT_SCENARIOS",
        "scope": "nested_integration_only; does not itself fail the prospective observation pilot or fixed outer screen",
        "unique_infeasible_F_populations": len(infeasible),
        "population_ids": [node["id"] for node in infeasible],
        "minimum_scenario_count_observed": min(node["scenario_count"] for node in dag["F_populations"]),
        "necessary_scenario_count": protocol["nested_integration"]["historical_structural_min_scenarios_necessary"],
        "proof": "For g=3 or4 scenarios, a three-way nonempty historical M4 selection partition necessarily has an inner training set of at most2 groups (g=2 leaves1). P6 meta cross-fitting then asks for a base trained on at most1 group. NestedBaseCache.prepare calls group_splits on that base population, which rejects fewer than2 groups. Empty or missing-class partitions also fail. Thus g<=4 is necessarily infeasible; g>=5 is not sufficient proof of feasibility.",
        "evidence": ["src/hac/actor_memory_base.py:57", "src/hac/actor_memory_base.py:86", "src/hac/actor_memory_base.py:316", "src/hac/actor_memory_base.py:368", "experiments/okutama_evidence_memory_protocol.json"],
        "action": "stop integration; no relaxed held/calibration exclusion, no silent historical-recipe change; new protocol authority or additional scenario data required",
    }
    return {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "integration_protocol_infeasible" if infeasible else "label_blind_plan_requires_label_and_execution_validation",
        "read_only": True,
        "labels_loaded": False,
        "predictions_loaded": False,
        "support_or_error_bank_loaded": False,
        "NPZ_members_read": list(SAFE_METADATA_FIELDS),
        "selected_route_is_hypothetical_not_a_gate_result": route,
        "protocol_canonical_sha256": canonical_hash(protocol),
        "canonical_sample_ids_sha256": canonical_hash(metadata["sample_ids"].tolist()),
        "pilot_selection": select_pilot(metadata, protocol),
        "dependency_DAG": dag,
        "fit_budget": fit_budget(dag, protocol, route),
        "blockers": [blocker] if infeasible else [],
        "execution_pending": {
            "source_model_lock": protocol["observations"]["source_model_lock"],
            "additional_unverified": ["historical label-dependent split and class feasibility", "generalized F(S) historical runner, presently original-fold-specific", "all exact transitive artifact reuse receipts", "actual extraction and optimizer throughput", "all prospective observation and reader gates"],
            "training_authorized_by_plan": False,
        },
        "handoff": handoff_decision(projected_job_seconds, protocol),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--route", choices=("V3", "V2_only"), default="V3", help="Hypothetical dependency branch, not a claim of qualification")
    parser.add_argument("--projected-job-seconds", type=float, default=None, help="Measured projection for one proposed training job; never launches it")
    parser.add_argument("--output", type=Path, default=None, help="Optional new JSON receipt; stdout is always retained")
    args = parser.parse_args(argv)
    try:
        repo = args.repo_root.resolve()
        protocol_path = args.protocol if args.protocol.is_absolute() else repo / args.protocol
        protocol = load_protocol(protocol_path)
        hash_checks = {}
        for name, pin in {"contract": protocol["contract"], **protocol["pinned_inputs"]}.items():
            path = (repo / pin["path"]).resolve()
            actual = file_sha256(path)
            hash_checks[name] = {"path": pin["path"], "expected_sha256": pin["sha256"], "actual_sha256": actual, "matches": actual == pin["sha256"]}
        mismatches = [name for name, receipt in hash_checks.items() if not receipt["matches"]]
        if mismatches:
            print(json.dumps({"status": "input_pin_mismatch", "read_only": True, "mismatched_inputs": mismatches, "hash_checks": hash_checks}, indent=2, allow_nan=False))
            return 2
        metadata = load_safe_metadata(repo / protocol["pinned_inputs"]["canonical_data"]["path"])
        plan = build_plan(metadata, protocol, route=args.route, projected_job_seconds=args.projected_job_seconds)
        plan["provenance"] = {"protocol_path": str(protocol_path), "protocol_file_sha256": file_sha256(protocol_path), "planner_file_sha256": file_sha256(Path(__file__)), "hash_checks": hash_checks}
        if args.output is not None:
            output = args.output if args.output.is_absolute() else repo / args.output
            write_new_json(output.resolve(), plan)
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "invalid_plan_inputs", "read_only": True, "error": str(exc)}, indent=2, allow_nan=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
