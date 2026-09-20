"""Focused synthetic tests: no action labels, cached predictions, or fits."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "body_witness_dependency_plan", ROOT / "experiments/plan_okutama_nested_intervention.py"
)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)


@pytest.fixture
def protocol():
    return planner.load_protocol(ROOT / "experiments/okutama_body_witness_protocol.json")


@pytest.fixture
def synthetic(protocol):
    """String-preserving canonical fold topology, no labels anywhere."""
    columns = {key: [] for key in planner.SAFE_METADATA_FIELDS}
    for fold, scenarios in protocol["population"]["outer_scenarios"].items():
        for scenario in scenarios:
            for index in range(24):
                columns["sample_ids"].append(f"{scenario}_sample_{index:03d}")
                columns["scenarios"].append(scenario)
                columns["folds"].append(int(fold))
                columns["recordings"].append(f"{scenario}_recording")
                columns["tracks"].append(str(index % 6))
    metadata = {key: np.asarray(value) for key, value in columns.items()}
    protocol["population"]["rows"] = len(metadata["sample_ids"])
    protocol["population"]["sample_ids_sha256"] = planner.canonical_hash(metadata["sample_ids"].tolist())
    return metadata, protocol


def test_protocol_pins_class_order_anchor_recipe_and_no_oracle(protocol):
    assert protocol["population"]["classes"] == ["sitting", "standing", "walking_running"]
    assert protocol["pinned_inputs"]["anchor"]["sha256"] == "ec9957a9393e6f1803d274d281549c20290ed59343b308c5be3446be37cec720"
    assert protocol["population"]["rows"] == 4977
    assert protocol["anchor_metrics"]["errors"] == 702
    assert protocol["adoption_gates"]["macro_f1_min"] - protocol["anchor_metrics"]["macro_f1"] == pytest.approx(0.005)
    assert not protocol["adoption_gates"]["oracle_qualifies"]
    assert protocol["training"]["updates"] == 800
    assert protocol["training"]["initial_seeds"] == [42]
    assert protocol["training"]["confirmation_seeds"] == [43, 44]
    assert protocol["candidate"]["class2_brier_invariant"] is False
    assert protocol["reader_architecture"]["modality_dims"]["pose"] == 2 * 8 * 193 + 16
    assert protocol["pilot"]["target_sensitivity_gate"]["metric_name"] == "mean_landmark_total_variation_including_outside_bin"


def test_protocol_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"same": 1, "same": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        planner.load_protocol(path)


def test_policy_blocks_are_fixed_hash_round_robin_and_recomputed(protocol):
    all_scenarios = {scenario for fold in protocol["population"]["outer_scenarios"].values() for scenario in fold}
    train = all_scenarios - set(protocol["population"]["outer_scenarios"]["0"])
    expected = (("2.8", "1.10", "2.7"), ("1.2", "1.11", "1.3"), ("1.5", "2.11"))
    assert planner.policy_blocks(train) == expected
    assert planner.policy_blocks(reversed(sorted(train))) == expected
    restricted = train - set(expected[0])
    recomputed = planner.policy_blocks(restricted)
    assert set().union(*map(set, recomputed)) == restricted
    assert sorted(map(len, recomputed)) == [1, 2, 2]
    assert "1.10" in train and "1.1" not in train
    with pytest.raises(ValueError, match="nonempty strings"):
        planner.policy_blocks([1.10, "1.2", "1.3"])
    with pytest.raises(ValueError, match="duplicate"):
        planner.policy_blocks(["1.10", "1.10", "1.2"])
    with pytest.raises(ValueError, match="three scenarios"):
        planner.policy_blocks(["1.10", "1.2"])


def test_safe_loader_never_indexes_labels_scores_or_support(monkeypatch):
    touched = []

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            assert key in planner.SAFE_METADATA_FIELDS
            touched.append(key)
            return np.asarray([0] if key == "folds" else ["safe"])

    def fake_load(path, *, allow_pickle):
        assert allow_pickle is False
        return Archive()

    monkeypatch.setattr(planner.np, "load", fake_load)
    result = planner.load_safe_metadata(Path("not_opened.npz"))
    assert tuple(touched) == planner.SAFE_METADATA_FIELDS
    assert tuple(result) == planner.SAFE_METADATA_FIELDS


def test_validation_rejects_numeric_scenarios_bad_folds_and_id_order(synthetic):
    metadata, protocol = synthetic
    planner.validate_metadata(metadata, protocol)
    bad = {key: value.copy() for key, value in metadata.items()}
    bad["scenarios"] = bad["scenarios"].astype(float)
    with pytest.raises(ValueError, match="strings"):
        planner.validate_metadata(bad, protocol)
    bad = {key: value.copy() for key, value in metadata.items()}
    bad["folds"][0] = 4
    with pytest.raises(ValueError, match="membership"):
        planner.validate_metadata(bad, protocol)
    bad = {key: value.copy() for key, value in metadata.items()}
    bad["sample_ids"][[0, 1]] = bad["sample_ids"][[1, 0]]
    with pytest.raises(ValueError, match="order/hash"):
        planner.validate_metadata(bad, protocol)


def test_pilot_128_quotas_track_first_pass_and_order_invariance(synthetic):
    metadata, protocol = synthetic
    pilot = planner.select_pilot(metadata, protocol)
    assert len(pilot["sample_ids"]) == len(set(pilot["sample_ids"])) == 128
    assert [row["quota"] for row in pilot["by_scenario"]] == [12] * 7 + [11] * 4
    assert all(row["first_pass_unique_tracks"] == 6 for row in pilot["by_scenario"])
    assert len(pilot["timing_first_16_global_sample_hash_ids"]) == 16
    reversed_metadata = {key: value[::-1].copy() for key, value in metadata.items()}
    assert planner.select_pilot(reversed_metadata, protocol) == pilot
    assert not pilot["selection_uses_labels_or_predictions"]


def test_exact_dependency_population_counts_and_hard_stop(synthetic):
    metadata, protocol = synthetic
    plan = planner.build_plan(metadata, protocol)
    assert plan["status"] == "integration_protocol_infeasible"
    assert plan["labels_loaded"] is False
    assert plan["predictions_loaded"] is False
    assert plan["support_or_error_bank_loaded"] is False
    counts = plan["fit_budget"]["exact_policy_layer_counts"]
    assert counts["F_prediction_requests"] == 80
    assert counts["F_population_occurrences_after_within_outer_dedup"] == 65
    assert counts["distinct_F_training_scenario_populations"] == 59
    assert counts["F_population_size_histogram"] == {"3": 4, "4": 36, "5": 2, "6": 12, "8": 1, "9": 4}
    assert counts["policy_training_procedure_occurrences"] == 20
    assert counts["distinct_policy_training_scenario_populations"] == 19
    assert counts["policy_estimator_fits_without_reuse_max"] == 60
    assert counts["policy_estimator_fits_after_exact_population_reuse_max"] == 57
    assert plan["blockers"][0]["unique_infeasible_F_populations"] == 40
    assert plan["blockers"][0]["minimum_scenario_count_observed"] == 3
    assert plan["dependency_DAG"]["ancestor_exclusions_checked"] is True
    for population in plan["dependency_DAG"]["F_populations"]:
        assert population["body_reader_posture_training_rows"]["exact"] is None
        assert population["canonical_row_count"] == population["scenario_count"] * 24
    json.dumps(plan, allow_nan=False)


def test_each_policy_ancestor_excludes_outer_and_calibration(synthetic):
    metadata, protocol = synthetic
    dag = planner.build_dependency_dag(metadata, protocol)
    populations = {node["id"]: node for node in dag["F_populations"]}
    requests = {node["id"]: node for node in dag["prediction_requests"]}
    for procedure in dag["policy_training_procedures"]:
        for request_id in procedure["oof_request_ids"]:
            request = requests[request_id]
            train = set(populations[request["F_population"]]["scenarios"])
            assert train.isdisjoint(procedure["forbidden_scenarios"])
            assert train.isdisjoint(request["prediction_scenarios"])
    contaminated = copy.deepcopy(dag)
    first = contaminated["prediction_requests"][0]
    ancestor = populations[first["F_population"]]["scenarios"][0]
    first["forbidden_scenarios"].append(ancestor)
    with pytest.raises(ValueError, match="Forbidden scenario"):
        planner.validate_ancestor_exclusions(contaminated)
    contaminated = copy.deepcopy(dag)
    procedure = contaminated["policy_training_procedures"][0]
    request = requests[procedure["oof_request_ids"][0]]
    procedure["forbidden_scenarios"].append(populations[request["F_population"]]["scenarios"][0])
    with pytest.raises(ValueError, match="Calibration/outer contamination"):
        planner.validate_ancestor_exclusions(contaminated)


def test_fit_upper_bounds_and_no_unvalidated_reuse(synthetic):
    metadata, protocol = synthetic
    plan = planner.build_plan(metadata, protocol)
    bounds = plan["fit_budget"]["conditional_recipe_upper_bounds_not_executable_counts"]
    assert bounds["M4_neural_fits_per_F"] == bounds["A3_neural_fits_per_F"] == 15
    assert bounds["M4_and_A3_neural_fits"] == 1770
    assert bounds["P6_population_requests_before_transitive_dedup"] == 767
    assert bounds["P6_estimator_fits_max"] == 39884
    assert bounds["body_reader_fits"] == 354
    assert bounds["historical_inner_exact_fit_count"] is None
    reuse = plan["fit_budget"]["reuse"]
    assert reuse["original_outer_F_population_matches"] == 5
    assert reuse["confirmed_screen_reader_fits_potentially_reusable"] == 30
    assert reuse["task_fits_certified_reusable_by_this_planner"] == 0
    direct = planner.build_plan(metadata, protocol, route="V2_only")
    assert direct["fit_budget"]["conditional_recipe_upper_bounds_not_executable_counts"]["body_reader_fits"] == 177
    assert direct["fit_budget"]["exact_policy_layer_counts"]["policy_estimator_fits_without_reuse_max"] == 40
    assert direct["fit_budget"]["exact_policy_layer_counts"]["policy_estimator_fits_after_exact_population_reuse_max"] == 38
    assert direct["fit_budget"]["screen_fit_counts"]["V2_total_if_V3_not_initially_fitted"] == 45
    with pytest.raises(ValueError, match="cannot be promoted"):
        planner.build_plan(metadata, protocol, route="V1")


@pytest.mark.parametrize("seconds,expected", [(None, True), (0, False), (1200, False), (1200.1, True), (3600, True)])
def test_twenty_minute_boundary_never_authorizes_execution(protocol, seconds, expected):
    decision = planner.handoff_decision(seconds, protocol)
    assert decision["handoff_required"] is expected
    assert decision["this_planner_can_launch_training"] is False


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_invalid_runtime_rejected(protocol, seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        planner.handoff_decision(seconds, protocol)


def test_main_json_only_and_no_output_file(monkeypatch, capsys, synthetic):
    metadata, protocol = synthetic
    pins = {"contract": protocol["contract"], **protocol["pinned_inputs"]}
    expected_hashes = {str((ROOT / pin["path"]).resolve()): pin["sha256"] for pin in pins.values()}
    monkeypatch.setattr(planner, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(planner, "file_sha256", lambda path: expected_hashes.get(str(path.resolve()), "synthetic_script_hash"))
    monkeypatch.setattr(planner, "load_safe_metadata", lambda path: metadata)
    assert planner.main(["--repo-root", str(ROOT)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["status"] == "integration_protocol_infeasible"
    assert all(pin["matches"] for pin in report["provenance"]["hash_checks"].values())
    assert report["read_only"] is True


def test_main_pin_mismatch_stops_before_metadata(monkeypatch, capsys, protocol):
    monkeypatch.setattr(planner, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(planner, "file_sha256", lambda path: "wrong_hash")

    def forbid_metadata(path):
        raise AssertionError("Metadata must not be read after a pin failure")

    monkeypatch.setattr(planner, "load_safe_metadata", forbid_metadata)
    assert planner.main(["--repo-root", str(ROOT)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "input_pin_mismatch"
    assert "anchor" in report["mismatched_inputs"]
