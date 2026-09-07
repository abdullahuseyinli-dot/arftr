from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from tools import lock_hac_continuation_protocols as locks

ROOT = Path(__file__).resolve().parents[1]


def load_spec(name: str) -> dict:
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def test_okutama_protocol_closes_replay_scope_and_order() -> None:
    spec = load_spec("okutama_cptr_continuation_protocol.json")

    locks.validate_protocol_spec("okutama", spec)
    assert spec["data_contract"]["eligible_total_rows"] == 4977 + 1383
    assert spec["data_contract"]["eligible_total_scenarios"] == 11 + 3
    assert spec["data_contract"]["feature_index_derivation"]["source_order"] == [
        "scenario_id_lexicographic",
        "source_recording_id_lexicographic",
        "provider_track_id_lexicographic",
        "center_frame_numeric",
    ]
    assert spec["execution_stages"]["R1a"]["seeds"] == [43]
    assert spec["execution_stages"]["R1b"]["requires_exact_role_specific_window_masks"]
    assert spec["interventions"]["F3"].startswith("direct_return_exact_teacher")
    assert spec["replay_reproduction_contract"] == {
        "current_gpu_teacher_max_abs_tolerance": 0.0005,
        "F3_statistical_probabilities": "exact_retained_teacher_probabilities",
        "current_gpu_direct_teacher_role": "reproduction_check_only_not_statistical_input",
        "decoded_output_identity": "F3_decodes_from_exact_retained_teacher_probabilities",
    }
    assert spec["source_archive_contract"] == {
        "file_name": "TrainSetFrames.zip",
        "expected_bytes": 5770432522,
        "historical_sha256_attestation": (
            "c021ce8a12c84e083f359023ffd41c145561aaedb48b118e7c5416d5ddcecb73"
        ),
        "source_lock_policy": (
            "Hash the supplied archive as an opaque byte stream at materialization lock time "
            "and require its filename, byte count, and SHA-256 to match the historical audit "
            "receipt. Do not parse any unselected member."
        ),
        "mask_extraction_policy": (
            "Open annotation members only for the 14 eligible scenarios proven complete by "
            "the eligible-ID/count invariant; emit a role-specific 17-frame Boolean mask "
            "artifact before any feature-array access."
        ),
    }


def test_okutama_fold_contract_is_exact_and_disjoint() -> None:
    spec = load_spec("okutama_cptr_continuation_protocol.json")
    folds = spec["fold_contract"]
    scenarios = [scenario for held in folds.values() for scenario in held]

    assert list(folds) == ["fold-0", "fold-1", "fold-2", "fold-3", "fold-4"]
    assert len(scenarios) == 11
    assert len(set(scenarios)) == 11
    assert set(scenarios) == {
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.10",
        "1.11",
        "2.2",
        "2.5",
        "2.7",
        "2.8",
        "2.11",
    }


def test_okutama_temporal_controls_are_fully_predeclared() -> None:
    spec = load_spec("okutama_cptr_continuation_protocol.json")
    controls = spec["temporal_controls"]

    assert controls["data_scope"] == {
        "rows": 4977,
        "scenarios": 11,
        "scope_value": "grouped_crossfit_oof",
        "folds": 5,
        "seeds": [42, 43, 44, 45, 46],
    }
    assert controls["T1_architecture"]["positional_or_timestamp_signal"] is False
    assert controls["T2_architecture"]["model_dim"] == 256
    assert controls["T2_architecture"]["distinct_indices"] == [4, 5, 6, 7, 8, 10, 11, 12]
    assert controls["training"]["fixed_epochs_by_seed"] == {
        "42": 5,
        "43": 5,
        "44": 5,
        "45": 5,
        "46": 3,
    }
    assert (
        spec["decision_rules"]["T2_sampler_benefit"][
            "aggregate_macro_f1_two_sided_95pct_lower_bound_must_exceed"
        ]
        == 0.0
    )


def test_vcoco_protocol_has_equal_v0_budget_and_shared_folds() -> None:
    spec = load_spec("vcoco_continuation_protocol.json")

    locks.validate_protocol_spec("vcoco", spec)
    assert len(spec["V0"]["families"]) == 4
    assert spec["V0"]["candidate_selection"]["ranking"] == [
        {"metric": "macro_f1", "direction": "descending"},
        {"metric": "locomotion_f1", "direction": "descending"},
        {"metric": "log_loss", "direction": "ascending"},
        {"metric": "candidate_id", "direction": "ascending"},
    ]
    assert spec["V0"]["shared_geometry_features"] == {
        "source": "shared_pooled_feature_rows_csv",
        "source_rows_sha256": ("fc0772cdf1f1bce3786cfdf3c9571bcf456cfaa00cb12ea0d26212a7ff0504cb"),
        "dimensions": 6,
        "names": [
            "log_bbox_area_fraction",
            "log_bbox_aspect_ratio",
            "bbox_center_x_fraction",
            "bbox_center_y_fraction",
            "log_person_pixel_height",
            "bbox_edge_distance_fraction",
        ],
        "preprocessing": "hac.vcoco_v3_models.geometry_features",
        "identical_across_all_families": True,
    }
    logistic = spec["V0"]["probability_stack_grid"]
    svm = spec["V0"]["linear_svm_grid"]
    logistic_budget = (
        len(logistic["component_C"]) * len(logistic["meta_C"]) * len(logistic["class_weight"])
    )
    svm_budget = len(svm["C"]) * len(svm["class_weight"])
    assert logistic_budget == svm_budget == spec["V0"]["candidate_selection_budget_per_family"]
    columns = spec["cross_validation"]["fold_map"]["columns"]
    assert columns[:5] == ["row_index", "person_id", "image_id", "label_index", "outer_fold"]
    assert columns[5:15] == [
        value for fold in range(5) for value in (f"inner_fold_o{fold}", f"stack_fold_o{fold}")
    ]
    assert columns[15:] == [
        f"selection_stack_fold_o{outer}_i{inner}" for outer in range(5) for inner in range(3)
    ]
    assert spec["cross_validation"]["selection_stack_seed_by_outer_and_inner_fold"] == (
        "20260827 + 200000 + 1000 * outer_fold + inner_fold"
    )
    assert spec["statistics"]["bootstrap_indices_artifact"]["shape"] == [10000, 4123]
    assert spec["statistics"]["swap_signs_artifact"]["shape"] == [100000, 516]
    assert spec["statistics"]["pre_fit_lock_required"] is True
    assert spec["V1"]["training_seeds"] == [42, 43, 44, 45, 46]


@pytest.mark.parametrize(
    "relative",
    (
        ".runs/vcoco_v3/temporal/development_manifest.csv",
        ".runs/vcoco_v3/okutama/features/dinov2_base/development_metadata.csv",
        ".runs/polar_v2/locked_protocol/vcoco_test_clean.csv",
        ".runs/calibration/values.npz",
        ".runs/confirmation/values.npz",
        ".runs/test/values.npz",
    ),
)
def test_protected_input_paths_are_rejected_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"must not be hashed")
    called = False

    def forbidden_hash(_: Path) -> tuple[str, int]:
        nonlocal called
        called = True
        raise AssertionError("protected bytes were hashed")

    monkeypatch.setattr(locks, "_sha256_file", forbidden_hash)
    with pytest.raises(RuntimeError, match="Forbidden|Protected"):
        locks._artifact_receipt(tmp_path, relative, "0" * 64)
    assert not called


def test_safe_input_receipt_binds_exact_captured_bytes(tmp_path: Path) -> None:
    path = tmp_path / ".runs" / "development" / "eligible.npz"
    path.parent.mkdir(parents=True)
    content = b"eligible-development-only"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    receipt = locks._artifact_receipt(tmp_path, ".runs/development/eligible.npz", expected)

    assert receipt == {
        "path": ".runs/development/eligible.npz",
        "sha256": expected,
        "size_bytes": len(content),
    }
    with pytest.raises(RuntimeError, match="Locked SHA-256 mismatch"):
        locks._artifact_receipt(tmp_path, ".runs/development/eligible.npz", "f" * 64)


def test_okutama_source_lock_opaque_hashes_all_declared_feature_arrays(
    tmp_path: Path,
) -> None:
    receipts = {}
    declarations = {
        "base_store_declaration": {
            "directory": ".runs/base",
            "arrays": ("tight", "context", "geometry"),
        },
        "part_store_declaration": {
            "directory": ".runs/part",
            "arrays": ("part_tokens", "part_confidence"),
        },
    }
    for receipt_name, contract in declarations.items():
        directory = tmp_path / contract["directory"]
        directory.mkdir(parents=True)
        arrays = {}
        for name in contract["arrays"]:
            content = f"opaque-{name}".encode()
            path = directory / f"{name}.npy"
            path.write_bytes(content)
            arrays[name] = {
                "path": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        declaration_path = directory / "store.json"
        declaration_path.write_text(json.dumps({"arrays": arrays}), encoding="utf-8")
        receipts[receipt_name] = {
            "path": declaration_path.relative_to(tmp_path).as_posix()
        }

    observed = locks._opaque_okutama_feature_array_receipts(
        tmp_path, {"receipts": receipts}
    )

    assert set(observed) == {
        "base_tight",
        "base_context",
        "base_geometry",
        "part_part_tokens",
        "part_part_confidence",
    }
    assert all(
        item["verification"] == "opaque_byte_stream_no_array_decoding_or_indexing"
        for item in observed.values()
    )
    (tmp_path / ".runs/base/tight.npy").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        locks._opaque_okutama_feature_array_receipts(tmp_path, {"receipts": receipts})


def test_git_source_receipt_requires_committed_blob(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "source"], cwd=tmp_path, check=True)
    commit = locks._git(tmp_path, "rev-parse", "HEAD")

    receipt = locks.git_source_receipt(tmp_path, "source.py", commit)

    assert receipt["path"] == "source.py"
    assert receipt["git_blob_oid"] == locks._git(tmp_path, "rev-parse", f"{commit}:source.py")
    assert (
        locks.committed_source_preconditions(
            tmp_path, {"sources": {"test_source": "source.py"}}, commit
        )
        == []
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from committed blob"):
        locks.git_source_receipt(tmp_path, "source.py", commit)
    assert locks.committed_source_preconditions(
        tmp_path, {"sources": {"test_source": "source.py"}}, commit
    ) == [
        {
            "name": "test_source",
            "path": "source.py",
            "issue": "differs_from_committed_blob",
        }
    ]


def test_inventory_digest_accepts_named_or_file_artifact() -> None:
    digest = "a" * 64
    assert (
        locks._inventory_digest(
            {"artifacts": {"eligible_index.csv": {"sha256": digest}}},
            "eligible_index",
            "eligible_index.csv",
        )
        == digest
    )
    assert locks._inventory_digest({}, "missing") is None


def test_lock_contract_paths_and_statuses_are_stable() -> None:
    okutama = locks.DATASET_CONFIG["okutama"]
    vcoco = locks.DATASET_CONFIG["vcoco"]

    assert okutama["source_status"] == (
        "OKUTAMA_CPTR_REPLAY_MATERIALIZATION_LOCKED_BEFORE_FEATURE_ACCESS"
    )
    assert okutama["execution_status"] == "OKUTAMA_CPTR_FROZEN_REPLAY_LOCKED_BEFORE_EXECUTION"
    assert okutama["default_source_output"].endswith("okutama_materialization_lock.json")
    assert okutama["temporal_controls_execution_output"].endswith(
        "okutama_temporal_controls_execution_lock.json"
    )
    assert okutama["temporal_controls_execution_status"] == (
        "OKUTAMA_TEMPORAL_CONTROLS_LOCKED_BEFORE_FITTING"
    )
    assert okutama["sources"]["temporal_controls_runner"] == (
        "experiments/run_okutama_temporal_controls.py"
    )
    assert vcoco["source_status"] == "VCOCO_CONTINUATION_SOURCE_LOCKED_BEFORE_V0_FITTING"
    assert vcoco["execution_status"] == "VCOCO_CONTINUATION_EXECUTION_LOCKED_BEFORE_FITTING"
    assert vcoco["sources"]["v0_runner"] == "experiments/run_vcoco_continuation_v0.py"


def test_execution_stage_validation_fails_before_unrelated_access(tmp_path: Path) -> None:
    arguments = Namespace(stage="v1", source_lock=None)

    with pytest.raises(ValueError, match="source-lock"):
        locks._execution_payload(tmp_path, "vcoco", arguments)


def test_okutama_r1a_uses_the_upfront_r1b_lock(tmp_path: Path) -> None:
    arguments = Namespace(stage="r1a", source_lock=None)

    with pytest.raises(ValueError, match="stage-r1b lock up front"):
        locks._execution_payload(tmp_path, "okutama", arguments)
