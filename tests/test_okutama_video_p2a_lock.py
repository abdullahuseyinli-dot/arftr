from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import lock_okutama_video_p2a as locks

ROOT = Path(__file__).resolve().parents[1]


def protocol() -> dict:
    return locks.load_protocol(ROOT)


def write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {
        "path": path.name,
        "sha256": locks.p0.digest(path.read_bytes()),
        "size_bytes": path.stat().st_size,
    }


def pin(root: Path, path: Path) -> dict:
    return locks.p1.receipt(root, path)


def primary_rows() -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"{scenario}-{label}",
            "recording_id": scenario,
            "label_index": str(label),
            "fold": fold,
            "center_frame": "30",
        }
        for fold, scenarios in locks.p0.FOLDS.items()
        for scenario in scenarios
        for label in range(3)
    ]


@pytest.fixture
def results(tmp_path: Path):
    spec, primary = protocol(), primary_rows()
    directory = tmp_path / "results"
    directory.mkdir()
    labels = np.array([int(row["label_index"]) for row in primary], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)[labels]
    arrays = {
        "sample_ids": np.array([row["sample_id"] for row in primary]),
        "recording_ids": np.array([row["recording_id"] for row in primary]),
        "labels": labels,
        "folds": np.array([int(row["fold"][-1]) for row in primary], dtype=np.int64),
        "baseline_probabilities": probabilities,
    }
    models = {}
    for arm in locks.SOURCE_ARMS:
        for probe in ("linear", "attentive"):
            name = f"{arm}__{probe}"
            arrays[name] = probabilities.copy()
            models[name] = {"fallback_rows": 0, "metrics": {"macro_f1": 1.0}}
    np.savez(directory / "oof_probabilities.npz", **arrays)
    np.savez(tmp_path / "baseline.npz", historical=probabilities)
    baseline = {**pin(tmp_path, tmp_path / "baseline.npz"), "probabilities_member": "historical"}
    retained = {
        "protocol": {"tag": "p1"},
        "sources": {"runner": {"sha256": "a" * 64}},
        "baseline": baseline,
    }
    request = {
        "status": "P1_REQUEST_BEFORE_FITTING",
        "lock_sha256": "b" * 64,
        "protocol": retained["protocol"],
        "source_sha256": "a" * 64,
        "sample_ids_sha256": locks.p0.canonical_digest([r["sample_id"] for r in primary]),
    }
    write_json(directory / "request.json", request)
    request_hash = locks.p0.canonical_digest(request)
    artifacts = {}
    for name in ("metrics.csv", "paired_statistics.json"):
        (directory / name).write_bytes(b"{}")
    for arm in locks.SOURCE_ARMS:
        for probe in ("linear", "attentive"):
            for fold in range(5):
                folder = directory / "workloads" / arm / probe / f"fold-{fold}"
                folder.mkdir(parents=True)
                checkpoint = "checkpoint.npz" if probe == "linear" else "checkpoint.pt"
                local = {}
                for name in (checkpoint, "predictions.npz"):
                    (folder / name).write_bytes(b"opaque-never-unpickle")
                    local[name] = locks.p0.digest((folder / name).read_bytes())
                write_json(
                    folder / "receipt.json",
                    {
                        "status": "P1_WORKLOAD_COMPLETE",
                        "artifacts": local,
                        "request": {
                            "arm": arm,
                            "probe": probe,
                            "outer_fold": fold,
                            "seed": 42,
                            "request_sha256": request_hash,
                            "held_sample_ids": [
                                r["sample_id"] for r in primary if r["fold"] == f"fold-{fold}"
                            ],
                        },
                    },
                )
                for name in (checkpoint, "predictions.npz", "receipt.json"):
                    path = folder / name
                    artifacts[path.relative_to(directory).as_posix()] = locks.p0.digest(
                        path.read_bytes()
                    )
    for name in ("metrics.csv", "paired_statistics.json", "oof_probabilities.npz"):
        artifacts[name] = locks.p0.digest((directory / name).read_bytes())
    summary = {
        "status": "OKUTAMA_VIDEO_P1_EXPLORATORY_CROSSFIT_COMPLETE",
        "rows": len(primary),
        "seed": 42,
        "scenarios": 11,
        "models": models,
        "baseline": {"macro_f1": 1.0},
        "request_sha256": request_hash,
        "inner_selection_baseline_access": False,
        "protected_manifest_reads": 0,
        "target_images_read": 0,
        "artifacts": artifacts,
    }
    write_json(directory / "summary.json", summary)
    spec["p1_artifacts"] = {
        "execution_lock": {"path": "execution.json", "sha256": "b" * 64, "size_bytes": 0},
        "summary": pin(tmp_path, directory / "summary.json"),
        "oof": pin(tmp_path, directory / "oof_probabilities.npz"),
    }
    spec["p1_reference"]["macro_f1"] = 1.0
    return spec, primary, retained, summary, arrays, directory


def repin_summary(root: Path, results) -> None:
    spec, _, _, summary, _, directory = results
    write_json(directory / "summary.json", summary)
    spec["p1_artifacts"]["summary"] = pin(root, directory / "summary.json")


def test_protocol_freezes_four_arms_budget_and_twelve_comparisons() -> None:
    spec = protocol()
    assert spec["authorization"] == locks.AUTHORIZATION
    assert spec["arms"] == locks.ARMS
    assert spec["statistics"]["comparisons"] == locks.expected_comparisons()
    assert len(spec["statistics"]["comparisons"]) == 12
    assert spec["probes"]["linear"]["C_values"] == [1e-5, 1e-4, 1e-3, 1e-2]
    assert spec["probes"]["factorized"]["C_pair_candidates"] == 16
    assert spec["fitting_rules"]["maximum_unique_estimator_fits"] == 325
    assert not spec["fitting_rules"]["p1_probabilities_in_fitting_or_selection"]
    assert not spec["fitting_rules"]["outer_held_baseline_fallback"]
    assert spec["feature_recipes"]["factorized_motion"] == ["vmean", "motion"]


@pytest.mark.parametrize("change", ["family", "budget", "authorization", "fallback"])
def test_protocol_drift_fails_closed(tmp_path: Path, change: str) -> None:
    spec = protocol()
    if change == "family":
        spec["statistics"]["comparisons"].pop()
    elif change == "budget":
        spec["probes"]["linear"]["C_values"].append(0.1)
    elif change == "authorization":
        spec["authorization"]["backbone_fitting"] = True
    else:
        spec["fitting_rules"]["outer_held_baseline_fallback"] = True
    write_json(tmp_path / locks.PROTOCOL_PATH, spec)
    with pytest.raises(RuntimeError):
        locks.load_protocol(tmp_path)


def test_complete_p1_evidence_and_opaque_checkpoints(tmp_path: Path, results) -> None:
    spec, primary, retained, _, _, _ = results
    evidence = locks.validate_p1_results(tmp_path, spec, retained, primary)
    assert evidence["workloads_completed"] == 30
    assert len(evidence["artifacts"]) == 93


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "RUNNING"),
        ("rows", 10),
        ("seed", 43),
        ("inner_selection_baseline_access", True),
        ("protected_manifest_reads", 1),
    ],
)
def test_incomplete_or_leaking_p1_rejected_before_npz(
    tmp_path: Path, results, monkeypatch, field: str, value
) -> None:
    spec, primary, retained, summary, _, _ = results
    summary[field] = value
    repin_summary(tmp_path, results)
    monkeypatch.setattr(
        locks.np, "load", lambda *_args, **_kwargs: pytest.fail("Premature NPZ read")
    )
    with pytest.raises(RuntimeError):
        locks.validate_p1_results(tmp_path, spec, retained, primary)


def test_p1_artifact_hash_changed_before_numpy_decode(tmp_path: Path, results, monkeypatch) -> None:
    spec, primary, retained, _, _, directory = results
    (directory / "oof_probabilities.npz").write_bytes(b"changed")
    monkeypatch.setattr(locks.np, "load", lambda *_args, **_kwargs: pytest.fail("Unverified NPZ"))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        locks.validate_p1_results(tmp_path, spec, retained, primary)


def test_rehashed_wrong_workload_is_rejected(tmp_path: Path, results) -> None:
    spec, primary, retained, summary, _, directory = results
    name = "workloads/vjepa21_real_clip/linear/fold-0/receipt.json"
    path = directory / name
    value = json.loads(path.read_text())
    value["request"]["held_sample_ids"].reverse()
    write_json(path, value)
    summary["artifacts"][name] = locks.p0.digest(path.read_bytes())
    repin_summary(tmp_path, results)
    with pytest.raises(RuntimeError, match="different folds"):
        locks.validate_p1_results(tmp_path, spec, retained, primary)


def test_missing_artifact_cannot_be_replaced_with_protected_path(tmp_path: Path, results) -> None:
    spec, primary, retained, summary, _, _ = results
    summary["artifacts"]["../confirmation/labels.csv"] = summary["artifacts"].pop("metrics.csv")
    repin_summary(tmp_path, results)
    with pytest.raises(RuntimeError, match="inventory"):
        locks.validate_p1_results(tmp_path, spec, retained, primary)


@pytest.mark.parametrize(
    "change", ["order", "fold", "label", "nonfinite", "dtype", "object", "member"]
)
def test_oof_semantics_fail_even_after_rehash(tmp_path: Path, results, change: str) -> None:
    spec, primary, _, summary, arrays, directory = results
    if change == "order":
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
    elif change == "fold":
        arrays["folds"][0] = 4
    elif change == "label":
        arrays["labels"][0] = 1
    elif change == "nonfinite":
        arrays["vjepa21_real_clip__linear"][0, 0] = np.nan
    elif change == "dtype":
        arrays["vjepa21_real_clip__linear"] = arrays["vjepa21_real_clip__linear"].astype(np.float32)
    elif change == "object":
        arrays["sample_ids"] = arrays["sample_ids"].astype(object)
    else:
        arrays["unexpected"] = np.ones(2)
    np.savez(directory / "oof_probabilities.npz", **arrays)
    spec["p1_artifacts"]["oof"] = pin(tmp_path, directory / "oof_probabilities.npz")
    with pytest.raises((RuntimeError, ValueError)):
        locks.validate_oof(tmp_path, spec["p1_artifacts"]["oof"], primary, summary)


def test_json_sha_and_size_checked_before_decode(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(b"not json")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        locks.checked_json(tmp_path, {"path": path.name, "sha256": "a" * 64, "size_bytes": 8})
    item = pin(tmp_path, path)
    item["size_bytes"] += 1
    with pytest.raises(RuntimeError, match="byte count"):
        locks.checked_json(tmp_path, item)


def test_primary_invalid_row_is_rejected_before_fold_build(tmp_path: Path, monkeypatch) -> None:
    primary = primary_rows()
    clips = [{**row, "all_frames_valid": "true", "center_frame_valid": "true"} for row in primary]
    clips[0]["all_frames_valid"] = "false"
    monkeypatch.setattr(locks.p0, "read_primary_cohort", lambda *_: (primary, {}))
    monkeypatch.setattr(locks.p0, "load_protocol", lambda *_: {})
    monkeypatch.setattr(locks.p0, "checked_bytes", lambda *_: b"")
    monkeypatch.setattr(locks.p0, "csv_rows", lambda *_: clips)
    monkeypatch.setattr(locks, "verified_receipt", lambda *_: {})
    spec = protocol()
    spec["primary_rows"] = len(primary)
    retained = {
        "inputs": {"clip_index": {"path": "clip.csv", "sha256": "a" * 64}, "frame_manifest": {}}
    }
    with pytest.raises(RuntimeError, match="every original row valid"):
        locks.validate_primary(tmp_path, retained, spec)


def test_non_ancestor_p1_rejected_before_source_or_cache_reads(tmp_path: Path, monkeypatch) -> None:
    retained = {
        "status": locks.p1.STATUS,
        "authorization": locks.AUTHORIZATION,
        "repository_commit": "a" * 40,
    }
    monkeypatch.setattr(locks, "checked_json", lambda *_: retained)
    monkeypatch.setattr(
        locks.p0.common,
        "_git",
        lambda *_: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git")),
    )
    with pytest.raises(RuntimeError, match="not an ancestor"):
        locks.validate_p1_lineage(tmp_path, protocol(), "b" * 40, {})


def test_environment_drift_rejected_before_any_extraction_read(tmp_path: Path, monkeypatch) -> None:
    spec = protocol()
    retained = {
        "status": locks.p1.STATUS,
        "authorization": locks.AUTHORIZATION,
        "repository_commit": "a" * 40,
        "repository_tree": "tree",
        "protocol": {},
        "sources": {"protocol": {"path": "protocol", "sha256": "c" * 64}},
        "source_sha256": {"protocol": "c" * 64},
        "protocol_sha256": "c" * 64,
        "environment": {"torch": "old"},
    }
    calls = []
    monkeypatch.setattr(locks, "checked_json", lambda *_: calls.append(True) or retained)
    monkeypatch.setattr(locks.p0.common, "_git", lambda *_: "tree")
    monkeypatch.setattr(locks.p1, "load_protocol", lambda *_: {})
    monkeypatch.setattr(
        locks.p0.common, "git_source_receipt", lambda *_: retained["sources"]["protocol"]
    )
    with pytest.raises(RuntimeError, match="environment"):
        locks.validate_p1_lineage(tmp_path, spec, "b" * 40, {"torch": "new"})
    assert len(calls) == 1


def test_clean_committed_repository_required_before_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(locks.p0.common, "_git", lambda *_: "?? uncommitted.py")
    monkeypatch.setattr(locks, "load_protocol", lambda *_: pytest.fail("Read before clean check"))
    with pytest.raises(RuntimeError, match="clean committed"):
        locks.build_p2a_payload(tmp_path)


def test_randomization_reproducible_and_fold_safe() -> None:
    spec = protocol()
    first = locks.randomization_receipt(spec)
    assert first == locks.randomization_receipt(spec)
    assert first["bootstrap_indices_shape"] == [10000, 11]
    assert first["swap_signs_shape"] == [2048, 11]
    folds = locks.p1.make_fold_map(primary_rows(), spec)
    for outer in range(5):
        for row in folds:
            assert (row[f"inner_fold_o{outer}"] == -1) == (row["outer_fold"] == outer)


def test_lock_nonoverwriting_and_tamper_comparison(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "lock.json"
    retained = {
        "status": locks.STATUS,
        "authorization": locks.AUTHORIZATION,
        "locked_at_utc": "2026-09-07T12:00:00+00:00",
        "repository_commit": "a" * 40,
    }
    locks.p0.write_lock(tmp_path, path, retained)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="overwrite"):
        locks.p0.write_lock(tmp_path, path, retained)
    assert path.read_bytes() == before
    monkeypatch.setattr(
        locks, "build_p2a_payload", lambda *_: {**retained, "repository_commit": "b" * 40}
    )
    with pytest.raises(RuntimeError, match="repository_commit"):
        locks.validate_p2a_lock(tmp_path, path)


def test_npy_object_payloads_are_not_unpickled() -> None:
    stream = io.BytesIO()
    np.save(stream, np.array([object()], dtype=object))
    stream.seek(0)
    with pytest.raises(ValueError, match="Object arrays"):
        np.load(stream, allow_pickle=False)
