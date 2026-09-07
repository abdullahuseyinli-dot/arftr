from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import lock_okutama_video_p1 as locks

ROOT = Path(__file__).resolve().parents[1]


def protocol() -> dict:
    return locks.load_protocol(ROOT)


def save_npy(path: Path, array: np.ndarray) -> dict:
    np.save(path, array, allow_pickle=True)
    return {"path": path.name, "sha256": locks.p0.digest(path.read_bytes())}


@pytest.fixture
def cache(tmp_path: Path):
    directory = tmp_path / ".runs" / "video_cache"
    directory.mkdir(parents=True)
    spec = protocol()
    ids = ["row0", "row1", "row2"]
    validity = np.array([[True, True], [False, True], [False, False]], dtype=bool)
    arms = ["vjepa21_real_clip", "vjepa21_repeated_center"]
    summary = {
        "status": "OKUTAMA_VIDEO_FROZEN_FEATURE_CACHE_COMPLETE",
        "mode": "full",
        "all_primary_rows": True,
        "rows": 3,
        "sample_ids": ids,
        "extraction_lock_sha256": "a" * 64,
        "candidate_fits": 0,
        "labels_used_for_fitting_or_selection": 0,
        "protected_rows_read": 0,
        "arms": {},
    }
    for column, name in enumerate(arms):
        shape = [3, 8, 9, 768]
        values = np.zeros(shape, dtype=np.float16)
        values[validity[:, column]] = 1.5
        spec["cache_contracts"]["video"]["arms"][name] = shape
        summary["arms"][name] = {
            **save_npy(directory / (name + ".npy"), values),
            "shape": shape,
            "dtype": "float16",
            "valid_rows": int(validity[:, column].sum()),
        }
    summary["validity"] = save_npy(directory / "validity.npy", validity)
    summary["completed"] = save_npy(directory / "completed.npy", np.ones(3, dtype=bool))
    request = {
        "mode": "full",
        "sample_ids_sha256": locks.p0.canonical_digest(ids),
        "extraction_lock_sha256": "a" * 64,
        "extractor_sha256": "b" * 64,
        "arms": arms,
    }
    request["request_sha256"] = locks.p0.canonical_digest(request)
    summary["request_sha256"] = request["request_sha256"]
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory, spec, ids, validity, summary


def validate_cache(root: Path, fixture):
    directory, spec, ids, validity, _ = fixture
    return locks.validate_feature_cache(
        root, directory, "video", spec, ids, validity, "a" * 64, "b" * 64
    )


def update_summary(cache) -> None:
    directory, _, _, _, summary = cache
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_protocol_fits_only_probes_and_separates_nested_baseline() -> None:
    spec = protocol()
    assert spec["authorization"] == {
        "probe_fitting": True,
        "raw_image_extraction": False,
        "feature_extraction": False,
        "backbone_fitting": False,
        "protected_data_access": False,
    }
    assert spec["baseline"]["probabilities_member"] == "t2_fixed_distinct_probabilities"
    assert "cannot enter tuning" in spec["baseline"]["selection_restriction"]
    assert spec["probes"]["linear"]["C_values"] == [0.01, 0.1, 1.0]
    assert spec["probes"]["attentive"]["epochs"] == 15
    assert spec["statistics"]["bootstrap_resamples"] == 10000


def test_full_cache_validity_and_receipts_pass(tmp_path: Path, cache) -> None:
    output = validate_cache(tmp_path, cache)
    feature = output["features"]["vjepa21_real_clip"]
    assert feature["shape"] == [3, 8, 9, 768]
    assert feature["valid_rows"] == 1
    assert feature["validity"]["sha256"] == output["validity"]["sha256"]


@pytest.mark.parametrize(
    "field,value",
    [("mode", "pilot"), ("all_primary_rows", False), ("rows", 2), ("status", "RUNNING")],
)
def test_nonfull_caches_rejected_before_arrays(
    tmp_path: Path, cache, monkeypatch, field: str, value
) -> None:
    cache[-1][field] = value
    update_summary(cache)
    monkeypatch.setattr(
        locks.np, "load", lambda *_args, **_kwargs: pytest.fail("Decoded nonfull cache")
    )
    with pytest.raises(RuntimeError, match="FULL cache"):
        validate_cache(tmp_path, cache)


def test_changed_order_rejected_before_arrays(tmp_path: Path, cache, monkeypatch) -> None:
    cache[-1]["sample_ids"] = list(reversed(cache[2]))
    update_summary(cache)
    monkeypatch.setattr(
        locks.np, "load", lambda *_args, **_kwargs: pytest.fail("Decoded wrong identities")
    )
    with pytest.raises(RuntimeError, match="sample-ID order"):
        validate_cache(tmp_path, cache)


def test_array_hash_checked_before_numpy_decode(tmp_path: Path, cache, monkeypatch) -> None:
    (cache[0] / "validity.npy").write_bytes(b"different")
    monkeypatch.setattr(
        locks.np, "load", lambda *_args, **_kwargs: pytest.fail("Decoded wrong bytes")
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch before decoding"):
        validate_cache(tmp_path, cache)


@pytest.mark.parametrize(
    "case,match",
    [
        ("validity", "validity differs"),
        ("completed", "incomplete original"),
        ("dtype", "shape/dtype"),
        ("nonfinite", "nonfinite"),
        ("invalid_nonzero", "invalid-row"),
    ],
)
def test_rehashed_bad_array_semantics_rejected(
    tmp_path: Path, cache, case: str, match: str
) -> None:
    directory, _, _, validity, summary = cache
    if case == "validity":
        changed = validity.copy()
        changed[1, 0] = True
        summary["validity"] = save_npy(directory / "validity.npy", changed)
    elif case == "completed":
        summary["completed"] = save_npy(directory / "completed.npy", np.array([True, True, False]))
    else:
        name = "vjepa21_real_clip"
        path = directory / (name + ".npy")
        values = np.load(path)
        if case == "dtype":
            values = values.astype(np.float32)
        elif case == "nonfinite":
            values[0, 0, 0, 0] = np.nan
        else:
            values[2, 0, 0, 0] = 5
        summary["arms"][name].update(save_npy(path, values))
    update_summary(cache)
    with pytest.raises(RuntimeError, match=match):
        validate_cache(tmp_path, cache)


def test_request_cannot_swap_extraction_source(tmp_path: Path, cache) -> None:
    directory = cache[0]
    request = json.loads((directory / "request.json").read_text())
    request["extractor_sha256"] = "c" * 64
    request["request_sha256"] = locks.p0.canonical_digest(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    cache[-1]["request_sha256"] = request["request_sha256"]
    update_summary(cache)
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source/mode/identity"):
        validate_cache(tmp_path, cache)


def test_cache_path_cannot_escape_local_directory(tmp_path: Path, cache) -> None:
    cache[-1]["validity"]["path"] = "../protected.npy"
    update_summary(cache)
    with pytest.raises(RuntimeError, match="basenames"):
        validate_cache(tmp_path, cache)


def baseline_fixture(tmp_path: Path, object_ids: bool = False):
    spec = protocol()
    primary = [
        {"sample_id": f"row{index}", "recording_id": "1.2", "label_index": str(index)}
        for index in range(3)
    ]
    arrays = {
        "sample_ids": np.array(
            [row["sample_id"] for row in primary], dtype=object if object_ids else str
        ),
        "recording_ids": np.array(["1.2"] * 3),
        "labels": np.arange(3),
        "t2_fixed_distinct_probabilities": np.eye(3),
    }
    path = tmp_path / "baseline.npz"
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    raw = stream.getvalue()
    path.write_bytes(raw)
    spec["baseline"].update(
        path="baseline.npz", sha256=locks.p0.digest(raw), size_bytes=len(raw), macro_f1=1.0
    )
    return spec, primary, arrays


def test_baseline_probabilities_and_labels_are_bound(tmp_path: Path) -> None:
    spec, primary, _ = baseline_fixture(tmp_path)
    result = locks.validate_baseline(tmp_path, spec, primary)
    assert result["macro_f1"] == 1.0
    primary[0]["label_index"] = "1"
    with pytest.raises(RuntimeError, match="identity/order/labels"):
        locks.validate_baseline(tmp_path, spec, primary)


def test_baseline_object_dtype_is_never_unpickled(tmp_path: Path) -> None:
    spec, primary, _ = baseline_fixture(tmp_path, object_ids=True)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        locks.validate_baseline(tmp_path, spec, primary)


def test_nested_folds_never_expose_outer_held_scenarios() -> None:
    primary = []
    for fold, scenarios in locks.p0.FOLDS.items():
        for scenario in scenarios:
            primary.extend(
                {
                    "sample_id": f"{scenario}-{index}",
                    "recording_id": scenario,
                    "fold": fold,
                    "label_index": str(index % 3),
                }
                for index in range(6)
            )
    rows = locks.make_fold_map(primary, protocol())
    for outer in range(5):
        current_held = {row["recording_id"] for row in rows if row["outer_fold"] == outer}
        for row in rows:
            assert (row[f"inner_fold_o{outer}"] == -1) == (row["outer_fold"] == outer)
        for inner in range(3):
            held = {row["recording_id"] for row in rows if row[f"inner_fold_o{outer}"] == inner}
            fit = {
                row["recording_id"]
                for row in rows
                if row["outer_fold"] != outer and row[f"inner_fold_o{outer}"] != inner
            }
            assert held and fit
            assert not held & fit
            assert not (held | fit) & current_held


def test_p0_fitting_authority_rejected_before_other_reads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        locks.p0,
        "read_lock",
        lambda *_args: (
            {"status": locks.p0.EXTRACTION_STATUS, "authorization": {"model_fitting": True}},
            b"{}",
        ),
    )
    with pytest.raises(RuntimeError, match="non-fitting P0"):
        locks.validate_p0_lineage(tmp_path, tmp_path / "lock.json", "a" * 40)


def test_unrelated_p0_commit_rejected(tmp_path: Path, monkeypatch) -> None:
    extraction = {
        "status": locks.p0.EXTRACTION_STATUS,
        "authorization": {"model_fitting": False},
        "source_lock": {"path": "source.json", "sha256": "b" * 64},
    }
    source = {"status": locks.p0.MATERIALIZATION_STATUS, "repository_commit": "c" * 40}
    responses = iter([(extraction, b"{}"), (source, b"{}")])
    monkeypatch.setattr(locks.p0, "read_lock", lambda *_args: next(responses))
    monkeypatch.setattr(locks.p0, "checked_bytes", lambda *_args: b"{}")
    monkeypatch.setattr(
        locks.p0.common,
        "_git",
        lambda *_args: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git merge-base")),
    )
    with pytest.raises(RuntimeError, match="not an ancestor"):
        locks.validate_p0_lineage(tmp_path, tmp_path / "lock.json", "a" * 40)


def test_p1_requires_clean_commit_before_probe_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(locks.p0.common, "_git", lambda *_args: "?? new_source.py")
    monkeypatch.setattr(
        locks, "load_protocol", lambda *_args: pytest.fail("Read before clean check")
    )
    with pytest.raises(RuntimeError, match="clean committed"):
        locks.build_p1_payload(tmp_path, tmp_path / "lock", tmp_path / "video", tmp_path / "dino")


def test_p1_lock_creation_is_nonoverwriting(tmp_path: Path) -> None:
    path = tmp_path / "p1.json"
    locks.p0.write_lock(tmp_path, path, {"status": locks.STATUS})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="overwrite"):
        locks.p0.write_lock(tmp_path, path, {"status": "changed"})
    assert path.read_bytes() == before


def test_dino_cache_pins_preprocessing_and_snapshot_bytes(tmp_path: Path) -> None:
    files = {"model.safetensors": {"sha256": "c" * 64, "size_bytes": 50}}
    snapshot = {"path": "C:/pinned/dino", "revision": "d" * 40, "files": files}
    video_request = {"checkpoint_sha256": "a" * 64}
    dino_request = {
        "shared_video_extractor_sha256": "b" * 64,
        "image_encoder_module_sha256": "e" * 64,
        "revision": "d" * 40,
        "snapshot": snapshot,
    }
    video_path, dino_path = tmp_path / "video.json", tmp_path / "dino.json"

    def request_receipt(path: Path, payload: dict) -> dict:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"request": {"path": path.name, "sha256": locks.p0.digest(path.read_bytes())}}

    video = request_receipt(video_path, video_request)
    dino = request_receipt(dino_path, dino_request)
    lock = {
        "inputs": {
            "checkpoint": {"sha256": "a" * 64},
            "dinov2_snapshot": json.loads(json.dumps(snapshot)),
        }
    }
    sources = {"video_cache": {"sha256": "b" * 64}, "image_encoder_module": {"sha256": "e" * 64}}
    locks.validate_cache_provenance(tmp_path, video, dino, lock, sources)
    dino_request["image_encoder_module_sha256"] = "f" * 64
    dino = request_receipt(dino_path, dino_request)
    with pytest.raises(RuntimeError, match="preprocessing/encoder source"):
        locks.validate_cache_provenance(tmp_path, video, dino, lock, sources)
    dino_request["image_encoder_module_sha256"] = "e" * 64
    dino_request["snapshot"]["files"]["model.safetensors"]["sha256"] = "f" * 64
    dino = request_receipt(dino_path, dino_request)
    with pytest.raises(RuntimeError, match="snapshot bytes"):
        locks.validate_cache_provenance(tmp_path, video, dino, lock, sources)
