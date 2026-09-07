from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import run_okutama_temporal_controls as controls
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "experiments/okutama_cptr_continuation_protocol.json"


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_arithmetic_pooling_is_permutation_invariant_and_all_valid_only():
    torch.manual_seed(9)
    model = controls.ArithmeticMeanFactorizedHead(12, model_dim=16, dropout=0.0).eval()
    values = torch.randn(5, 8, 12)
    permutation = torch.tensor([7, 0, 5, 2, 3, 6, 1, 4])
    valid = torch.ones(5, 8, dtype=torch.bool)

    with torch.inference_mode():
        original = model(values, valid)
        shuffled = model(values[:, permutation], valid)

    assert torch.allclose(original.pooled_features, shuffled.pooled_features, atol=1e-6, rtol=0)
    assert torch.allclose(original.posture_logits, shuffled.posture_logits, atol=1e-6, rtol=0)
    assert torch.allclose(original.motion_logits, shuffled.motion_logits, atol=1e-6, rtol=0)
    assert torch.allclose(original.probabilities, shuffled.probabilities, atol=1e-6, rtol=0)
    assert torch.equal(original.attention_weights, torch.full((5, 8), 1 / 8))
    invalid = valid.clone()
    invalid[0, 0] = False
    with pytest.raises(ValueError, match="all-valid"):
        model(values, invalid)


def test_repeated_and_distinct_sampler_mapping_and_center_are_exact():
    distinct = np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3)
    repeated = distinct[:, controls.LEGACY_FROM_DISTINCT]
    controls.validate_sampler_arrays(
        repeated,
        distinct,
        short_centre_index=4,
        distinct_centre_index=4,
        distinct_indices=controls.DISTINCT_INDICES,
    )

    assert controls.LEGACY_INDICES.tolist() == [4, 6, 6, 8, 8, 10, 10, 12]
    assert controls.DISTINCT_INDICES.tolist() == [4, 5, 6, 7, 8, 10, 11, 12]
    assert controls.LEGACY_INDICES[4] == controls.DISTINCT_INDICES[4] == 8
    damaged = repeated.copy()
    damaged[0, 2, 0] += 1
    with pytest.raises(RuntimeError, match="not an exact view"):
        controls.validate_sampler_arrays(
            damaged,
            distinct,
            short_centre_index=4,
            distinct_centre_index=4,
            distinct_indices=controls.DISTINCT_INDICES,
        )


def test_t2_arms_are_equal_capacity_and_identically_initialized():
    torch.manual_seed(42)
    repeated = controls.build_model("t2_legacy_repeated", 30)
    torch.manual_seed(42)
    distinct = controls.build_model("t2_fixed_distinct", 30)

    assert sum(parameter.numel() for parameter in repeated.parameters()) == sum(
        parameter.numel() for parameter in distinct.parameters()
    )
    for name, value in repeated.state_dict().items():
        assert torch.equal(value, distinct.state_dict()[name])


def _valid_fold_frame() -> pd.DataFrame:
    group_to_fold = controls.expected_fold_for_group(protocol())
    groups = sorted(group_to_fold)
    recording_ids = np.asarray(
        [groups[index % len(groups)] for index in range(controls.EXPECTED_PRIMARY_ROWS)]
    )
    return pd.DataFrame(
        {
            "bundle_row_index": np.arange(controls.EXPECTED_PRIMARY_ROWS),
            "sample_id": [f"sample-{index:05d}" for index in range(controls.EXPECTED_PRIMARY_ROWS)],
            "recording_id": recording_ids,
            "fold": [group_to_fold[group] for group in recording_ids],
        }
    )


def test_fold_map_enforces_historical_scenario_isolation():
    frame = _valid_fold_frame()
    controls.validate_fold_frame(frame, protocol())
    broken = frame.copy()
    broken.loc[0, "fold"] = next(
        fold for fold in controls.EXPECTED_FOLDS if fold != broken.loc[0, "fold"]
    )
    with pytest.raises(RuntimeError, match="crosses"):
        controls.validate_fold_frame(broken, protocol())


def test_shared_resampling_and_statistics_are_identity_preserving(tmp_path):
    group_order = np.asarray([f"scenario-{index:02d}" for index in range(11)])
    first = controls.make_resampling(group_order)
    second = controls.make_resampling(group_order)
    for name in first:
        assert np.array_equal(first[name], second[name])
    assert first["group_draw_indices"].shape == (10_000, 11)
    assert first["group_draw_indices"].dtype == np.int16
    assert first["exact_swap_assignments"].shape == (2**11, 11)
    assert first["exact_swap_assignments"].dtype == bool
    path = tmp_path / "scenario_resampling.npz"
    controls.write_npz_atomic(path, **first)
    controls.validate_resampling(path, group_order)

    groups = np.repeat(group_order, 3)
    labels = np.tile(np.arange(3), 11)
    probabilities = np.eye(3, dtype=np.float64)[labels] * 0.9 + 0.1 / 3
    bootstrap = controls.bootstrap_contrast(
        labels=labels,
        reference=probabilities,
        candidate=probabilities.copy(),
        groups=groups,
        group_order=group_order,
        selector=np.ones(len(labels), dtype=bool),
        draws=first["group_draw_indices"],
    )
    swap = controls.exact_group_swap(
        labels=labels,
        reference=probabilities,
        candidate=probabilities.copy(),
        groups=groups,
        group_order=group_order,
        assignments=first["exact_swap_assignments"],
    )
    assert bootstrap["observed_macro_f1_delta"] == 0.0
    assert bootstrap["macro_f1_delta_two_sided_95pct"] == [0.0, 0.0]
    assert bootstrap["observed_nll_delta"] == 0.0
    assert swap["one_sided_pvalue"] == 1.0


def test_completed_workload_resume_is_hash_bound(tmp_path):
    paths = controls.workload_paths(tmp_path, "t1_arithmetic_mean", "fold-0", 42)
    paths["root"].mkdir(parents=True)
    request_core = {"status": "synthetic", "source_sha256": {"runner": "a" * 64}}
    request_sha256 = controls.canonical_json_sha256(request_core)
    controls.write_json_atomic(
        paths["request"], {**request_core, "request_sha256": request_sha256}
    )
    controls.write_npz_atomic(
        paths["predictions"],
        sample_ids=np.asarray(["a", "b", "c"]),
        recording_ids=np.asarray(["g0", "g0", "g0"]),
        bundle_row_indices=np.arange(3),
        primary_row_indices=np.arange(3),
        labels=np.arange(3),
        posture_logits=np.zeros((3, 2), dtype=np.float32),
        motion_logits=np.zeros((3, 2), dtype=np.float32),
        probabilities=np.full((3, 3), 1 / 3, dtype=np.float32),
    )
    paths["checkpoint"].write_bytes(b"checkpoint")
    paths["history"].write_text('{"epochs": []}\n', encoding="utf-8")
    summary = {
        "status": controls.RUN_STATUS,
        "request_sha256": request_sha256,
        "resource_measurement": {
            "scope": "synthetic",
            "initial_cuda_memory_allocated_bytes": 0,
            "initial_cuda_memory_reserved_bytes": 0,
            "peak_cuda_memory_allocated_bytes": 0,
            "peak_cuda_memory_reserved_bytes": 0,
        },
        "artifact_sha256": {
            paths[name].name: controls.sha256_file(paths[name])
            for name in ("predictions", "checkpoint", "history")
        },
    }
    controls.write_json_atomic(paths["summary"], summary)
    assert (
        controls.validate_existing_workload(
            paths=paths, request_sha256=request_sha256, expected_rows=3
        )["status"]
        == controls.RUN_STATUS
    )
    paths["history"].write_text('{"epochs": [1]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed closed"):
        controls.validate_existing_workload(
            paths=paths, request_sha256=request_sha256, expected_rows=3
        )


def test_cli_dry_run_validates_contract_without_fitting(capsys):
    controls.main(["--mode", "dry-run", "--protocol", str(PROTOCOL_PATH)])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "OKUTAMA_TEMPORAL_CONTROLS_DRY_RUN_VALID"
    assert output["workloads"] == 75
    assert output["model_fits"] == 0
    assert output["feature_values_read"] == 0


def test_cli_prepare_builds_primary_fold_map_and_shared_randomization(tmp_path, capsys):
    source_lock = tmp_path / "source_lock.json"
    controls.write_json_atomic(
        source_lock,
        {
            "status": controls.MATERIALIZATION_LOCK_STATUS,
            "authorization": {"temporal_controls_artifact_construction": True},
        },
    )
    bundle = tmp_path / "eligible_feature_bundle.npz"
    frame = _valid_fold_frame()
    validation_rows = 1_383
    controls.write_npz_atomic(
        bundle,
        sample_ids=np.concatenate(
            (
                frame["sample_id"].to_numpy(dtype=str),
                np.asarray([f"validation-{i}" for i in range(validation_rows)]),
            )
        ),
        recording_ids=np.concatenate(
            (
                frame["recording_id"].to_numpy(dtype=str),
                np.repeat("validation", validation_rows),
            )
        ),
        scope=np.concatenate(
            (
                np.repeat("grouped_crossfit_oof", len(frame)),
                np.repeat("fixed_development_validation", validation_rows),
            )
        ),
        fold=np.concatenate((frame["fold"].to_numpy(dtype=str), np.repeat("", validation_rows))),
    )
    bundle_summary = tmp_path / "bundle_summary.json"
    controls.write_json_atomic(
        bundle_summary,
        {
            "status": "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE",
            "artifact_sha256": {bundle.name: controls.sha256_file(bundle)},
            "source_sha256": {"protocol_lock": controls.sha256_file(source_lock)},
        },
    )
    fold_map = tmp_path / "fold_map.csv"
    resampling = tmp_path / "scenario_resampling.npz"
    prepare_summary = tmp_path / "prepare_summary.json"
    arguments = [
        "--mode",
        "prepare",
        "--protocol",
        str(PROTOCOL_PATH),
        "--source-lock",
        str(source_lock),
        "--feature-bundle",
        str(bundle),
        "--bundle-summary",
        str(bundle_summary),
        "--fold-map",
        str(fold_map),
        "--scenario-resampling",
        str(resampling),
        "--prepare-summary",
        str(prepare_summary),
    ]
    controls.main(arguments)
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == controls.PREPARE_STATUS
    assert first["model_fits"] == first["protected_rows_read"] == 0
    assert len(pd.read_csv(fold_map)) == controls.EXPECTED_PRIMARY_ROWS
    with np.load(resampling, allow_pickle=False) as arrays:
        assert arrays["group_draw_indices"].shape == (10_000, 11)
        assert arrays["exact_swap_assignments"].shape == (2**11, 11)

    controls.main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert second["resume_action"] == "verified_existing_preparation_reused"


def test_primary_loader_requires_exact_retained_teacher_schema(tmp_path):
    fold_frame = _valid_fold_frame()
    fold_map = tmp_path / "fold_map.csv"
    fold_frame.to_csv(fold_map, index=False)
    validation_rows = 1_383
    rows = controls.EXPECTED_PRIMARY_ROWS + validation_rows
    distinct = np.arange(rows * 8 * 2, dtype=np.float32).reshape(rows, 8, 2)
    repeated = distinct[:, controls.LEGACY_FROM_DISTINCT]
    sample_ids = np.concatenate(
        (
            fold_frame["sample_id"].to_numpy(dtype=str),
            np.asarray([f"validation-{index}" for index in range(validation_rows)]),
        )
    )
    teacher = np.full(
        (controls.EXPECTED_PRIMARY_ROWS, len(controls.EXPECTED_SEEDS), 3),
        1 / 3,
        dtype=np.float32,
    )

    def save_bundle(path: Path, retained: np.ndarray) -> None:
        controls.write_npz_atomic(
            path,
            sample_ids=sample_ids,
            recording_ids=np.concatenate(
                (
                    fold_frame["recording_id"].to_numpy(dtype=str),
                    np.repeat("validation", validation_rows),
                )
            ),
            scope=np.concatenate(
                (
                    np.repeat("grouped_crossfit_oof", controls.EXPECTED_PRIMARY_ROWS),
                    np.repeat("fixed_development_validation", validation_rows),
                )
            ),
            fold=np.concatenate(
                (fold_frame["fold"].to_numpy(dtype=str), np.repeat("", validation_rows))
            ),
            labels=np.arange(rows, dtype=np.int64) % 3,
            transition_targets=np.zeros(rows, dtype=bool),
            occlusion_targets=np.ones(rows, dtype=bool),
            short_features=repeated,
            short_valid_mask=np.ones((rows, 8), dtype=bool),
            short_centre_index=np.asarray(4, dtype=np.int64),
            distinct_short_features=distinct,
            distinct_short_valid_mask=np.ones((rows, 8), dtype=bool),
            distinct_short_indices=controls.DISTINCT_INDICES,
            distinct_short_centre_index=np.asarray(4, dtype=np.int64),
            primary_retained_teacher_probabilities=retained,
            primary_retained_teacher_sample_ids=fold_frame["sample_id"].to_numpy(dtype=str),
            retained_teacher_seeds=np.asarray(controls.EXPECTED_SEEDS, dtype=np.int64),
        )

    bundle = tmp_path / "eligible_feature_bundle.npz"
    save_bundle(bundle, teacher)
    summary = tmp_path / "bundle_summary.json"
    controls.write_json_atomic(
        summary,
        {
            "status": "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE",
            "artifact_sha256": {bundle.name: controls.sha256_file(bundle)},
        },
    )
    loaded = controls.load_primary_data(
        bundle_path=bundle,
        bundle_summary_path=summary,
        fold_map_path=fold_map,
        protocol=protocol(),
    )
    assert loaded.retained_teacher_probabilities.shape == (4_977, 5, 3)

    bad_bundle = tmp_path / "bad_bundle.npz"
    save_bundle(bad_bundle, teacher.astype(np.float64))
    bad_summary = tmp_path / "bad_summary.json"
    controls.write_json_atomic(
        bad_summary,
        {
            "status": "OKUTAMA_CPTR_ROLE_SAFE_FEATURE_BUNDLE_COMPLETE",
            "artifact_sha256": {bad_bundle.name: controls.sha256_file(bad_bundle)},
        },
    )
    with pytest.raises(RuntimeError, match="must remain float32"):
        controls.load_primary_data(
            bundle_path=bad_bundle,
            bundle_summary_path=bad_summary,
            fold_map_path=fold_map,
            protocol=protocol(),
        )


def test_benchmark_workload_is_reused_by_full_schedule_without_refit(tmp_path):
    rows = 30
    folds = np.repeat(np.asarray(controls.EXPECTED_FOLDS), 6)
    groups = np.asarray([f"group-{fold}" for fold in folds])
    features = np.arange(rows * 8 * 4, dtype=np.float32).reshape(rows, 8, 4) / 100
    data = controls.PrimaryData(
        sample_ids=np.asarray([f"sample-{index}" for index in range(rows)]),
        recording_ids=groups,
        labels=np.arange(rows, dtype=np.int64) % 3,
        folds=folds,
        bundle_row_indices=np.arange(rows),
        short_features=features,
        distinct_features=features.copy(),
        occluded=np.ones(rows, dtype=bool),
        transition=np.zeros(rows, dtype=bool),
        retained_teacher_probabilities=np.full((rows, 5, 3), 1 / 3),
    )
    training = {
        "fixed_epochs_by_seed": {"42": 0},
        "batch_size": 8,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.02,
    }
    first = controls.run_workload(
        arm="t1_arithmetic_mean",
        fold="fold-0",
        seed=42,
        data=data,
        training=training,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        source_sha256={"execution_lock": "a" * 64},
        latency_warmup=1,
        latency_iterations=20,
    )
    second = controls.run_workload(
        arm="t1_arithmetic_mean",
        fold="fold-0",
        seed=42,
        data=data,
        training=training,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        source_sha256={"execution_lock": "a" * 64},
        latency_warmup=1,
        latency_iterations=20,
    )
    assert first["status"] == controls.RUN_STATUS
    assert second["resume_action"] == "verified_existing_workload_reused"
    assert second["artifact_sha256"] == first["artifact_sha256"]
    model, loaded_summary, held = controls.load_completed_workload_model(
        arm="t1_arithmetic_mean",
        fold="fold-0",
        seed=42,
        data=data,
        training=training,
        output_dir=tmp_path,
        source_sha256={"execution_lock": "a" * 64},
        latency_warmup=1,
        latency_iterations=20,
        device=torch.device("cpu"),
    )
    assert isinstance(model, controls.ArithmeticMeanFactorizedHead)
    assert loaded_summary["request_sha256"] == first["request_sha256"]
    assert len(held) == 6

    checkpoint = controls.workload_paths(
        tmp_path, "t1_arithmetic_mean", "fold-0", 42
    )["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["unexpected"] = torch.tensor(1)
    torch.save(payload, checkpoint)
    summary_path = controls.workload_paths(
        tmp_path, "t1_arithmetic_mean", "fold-0", 42
    )["summary"]
    tampered_summary = controls.load_json(summary_path)
    tampered_summary["artifact_sha256"]["checkpoint.pt"] = controls.sha256_file(checkpoint)
    controls.write_json_atomic(summary_path, tampered_summary)
    with pytest.raises(RuntimeError, match="top-level schema"):
        controls.load_completed_workload_model(
            arm="t1_arithmetic_mean",
            fold="fold-0",
            seed=42,
            data=data,
            training=training,
            output_dir=tmp_path,
            source_sha256={"execution_lock": "a" * 64},
            latency_warmup=1,
            latency_iterations=20,
            device=torch.device("cpu"),
        )


def test_joint_latency_uses_identical_batch_and_exact_balanced_order():
    torch.manual_seed(1)
    candidate = controls.ArithmeticMeanFactorizedHead(4, model_dim=8, dropout=0.0)
    torch.manual_seed(2)
    reference = controls.ArithmeticMeanFactorizedHead(4, model_dim=8, dropout=0.0)
    batch = np.arange(6 * 8 * 4, dtype=np.float32).reshape(6, 8, 4)
    seen: dict[str, list[torch.Tensor]] = {"candidate": [], "reference": []}
    candidate.register_forward_pre_hook(
        lambda _module, arguments: seen["candidate"].append(arguments[0].detach().cpu().clone())
    )
    reference.register_forward_pre_hook(
        lambda _module, arguments: seen["reference"].append(arguments[0].detach().cpu().clone())
    )
    measured = controls.measure_interleaved_latency(
        candidate,
        reference,
        batch,
        device=torch.device("cpu"),
        warmup_paired_rounds=2,
        timed_paired_rounds=4,
    )
    design = {"timed_paired_rounds": 4}
    controls.validate_joint_measurement_design(measured, design)
    assert measured["order"] == ["AB", "BA", "AB", "BA"]
    assert len(measured["candidate_batch_latency_ms"]) == 4
    assert len(measured["reference_batch_latency_ms"]) == 4
    expected = torch.from_numpy(batch)
    assert all(torch.equal(value, expected) for value in seen["candidate"])
    assert all(torch.equal(value, expected) for value in seen["reference"])

    broken = {**measured, "order": ["AB", "AB", "BA", "BA"]}
    with pytest.raises(RuntimeError, match="frozen schedule"):
        controls.validate_joint_measurement_design(broken, design)
