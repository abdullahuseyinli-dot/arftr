from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from experiments import cache_okutama_sear_dense_dino_full as full


def fixture_cache(tmp_path, count=5, chunk_rows=4):
    source = tmp_path / "source.py"
    source.write_text("# immutable test source\n", encoding="utf-8")
    request = {
        "rows": count,
        "chunk_rows": chunk_rows,
        "loaded_source_hashes": {str(source): full.sha256_file(source)},
    }
    ids = np.asarray([f"center-{i}" for i in range(count)])
    fallback = np.asarray([i % 2 == 0 for i in range(count)])
    frames = np.arange(count, dtype=np.int64) * 30
    return tmp_path / "cache", request, ids, fallback, frames


def fill_and_commit(output, request, arrays, completed, ids, start, stop):
    arrays["patch_tokens"][start:stop] = np.arange(start, stop)[:, None, None] + 1
    arrays["center_cls"][start:stop] = np.arange(start, stop)[:, None] + 10
    arrays["local_valid"][start:stop] = True
    full.commit_chunk(output, request, arrays, completed, ids, start, stop, {})


def test_full_protocol_preserves_pilot_contract_and_excludes_fits():
    protocol = json.loads(full.PROTOCOL.read_text())
    full.validate_protocol(protocol)
    assert protocol["authorization"]["full_frozen_center_extraction"]
    assert not protocol["authorization"]["classifier_fitting"]
    assert protocol["shape_per_row"] == [729, 768]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("microbatch", 1),
        ("image_size", 378),
        ("cls_max_absolute_difference", 1e-3),
        ("cohort_rows", 4976),
        ("commit_chunk_rows", 5000),
    ],
)
def test_full_protocol_rejects_silent_changes(key, value):
    protocol = json.loads(full.PROTOCOL.read_text())
    protocol[key] = value
    with pytest.raises(RuntimeError, match="contract"):
        full.validate_protocol(protocol)


def test_full_protocol_rejects_classifier_authorization():
    protocol = json.loads(full.PROTOCOL.read_text())
    protocol["authorization"]["classifier_fitting"] = True
    with pytest.raises(RuntimeError, match="only frozen"):
        full.validate_protocol(protocol)


def test_new_cache_shapes_and_partial_chunk_resume(tmp_path):
    arguments = fixture_cache(tmp_path)
    output, request, ids, fallback, frames = arguments
    arrays, completed = full.prepare_cache(*arguments)
    assert arrays["patch_tokens"].shape == (5, 729, 768)
    assert arrays["center_cls"].shape == (5, 768)
    assert arrays["patch_tokens"].dtype == np.float16
    assert not completed.any()
    fill_and_commit(output, request, arrays, completed, ids, 0, 4)
    # An interrupted uncommitted final row is deliberately not trusted on resume.
    arrays["patch_tokens"][4] = 999
    arrays["patch_tokens"].flush()
    reopened, mask = full.prepare_cache(*arguments)
    assert mask.tolist() == [True, True, True, True, False]
    assert (reopened["patch_tokens"][2] == 3).all()
    fill_and_commit(output, request, reopened, mask, ids, 4, 5)
    _, final_mask = full.prepare_cache(*arguments)
    assert final_mask.all()


def test_receipt_before_bitmap_crash_recovers_verified_completion(tmp_path):
    args = fixture_cache(tmp_path)
    output, request, ids, _, _ = args
    arrays, completed = full.prepare_cache(*args)
    fill_and_commit(output, request, arrays, completed, ids, 0, 4)
    full.array_atomic(output / "completed.npy", np.zeros(5, bool))
    _, recovered = full.prepare_cache(*args)
    assert recovered.tolist() == [True, True, True, True, False]


def test_completion_without_receipt_fails_closed(tmp_path):
    args = fixture_cache(tmp_path)
    output, _, _, _, _ = args
    full.prepare_cache(*args)
    full.array_atomic(output / "completed.npy", np.ones(5, bool))
    with pytest.raises(RuntimeError, match="without a committed"):
        full.prepare_cache(*args)


@pytest.mark.parametrize("name", ["patch_tokens", "center_cls", "local_valid"])
def test_committed_payload_tampering_is_detected(tmp_path, name):
    args = fixture_cache(tmp_path)
    output, request, ids, _, _ = args
    arrays, completed = full.prepare_cache(*args)
    fill_and_commit(output, request, arrays, completed, ids, 0, 4)
    arrays[name][0] = False if name == "local_valid" else 77
    arrays[name].flush()
    with pytest.raises(RuntimeError, match="bytes changed"):
        full.prepare_cache(*args)


def test_metadata_and_request_changes_cannot_reuse_existing_cache(tmp_path):
    args = fixture_cache(tmp_path)
    output, request, ids, fallback, frames = args
    full.prepare_cache(*args)
    changed = copy.deepcopy(request)
    changed["source_condition"] = "other"
    with pytest.raises(RuntimeError, match="different immutable request"):
        full.prepare_cache(output, changed, ids, fallback, frames)
    swapped = ids[::-1].copy()
    with pytest.raises(RuntimeError, match="identity metadata"):
        full.prepare_cache(output, request, swapped, fallback, frames)


def test_foreign_or_partial_initialization_is_preserved(tmp_path):
    args = fixture_cache(tmp_path)
    output = args[0]
    output.mkdir()
    foreign = output / "user-file.txt"
    foreign.write_text("retain", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        full.prepare_cache(*args)
    assert foreign.read_text() == "retain"


def test_foreign_file_or_bad_snapshot_in_initialized_cache_is_rejected(tmp_path):
    args = fixture_cache(tmp_path)
    output = args[0]
    full.prepare_cache(*args)
    extra = output / "source_snapshots" / "foreign.py"
    extra.write_text("# preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source snapshots"):
        full.prepare_cache(*args)
    assert extra.is_file()


def test_commit_refuses_nonfinite_values_and_overwrite(tmp_path):
    args = fixture_cache(tmp_path)
    output, request, ids, _, _ = args
    arrays, completed = full.prepare_cache(*args)
    arrays["patch_tokens"][0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        full.commit_chunk(output, request, arrays, completed, ids, 0, 4, {})
    fill_and_commit(output, request, arrays, completed, ids, 0, 4)
    with pytest.raises(RuntimeError, match="overwrite"):
        full.commit_chunk(output, request, arrays, completed, ids, 0, 4, {})


def test_complete_summary_hash_gate(tmp_path):
    args = fixture_cache(tmp_path, count=4)
    output, request, ids, _, _ = args
    arrays, completed = full.prepare_cache(*args)
    fill_and_commit(output, request, arrays, completed, ids, 0, 4)
    path = output / "center_cls.npy"
    summary = {
        "request_sha256": full.canonical_hash(request),
        "artifacts": {
            path.name: {"sha256": full.sha256_file(path), "size_bytes": path.stat().st_size}
        },
    }
    full.pilot._json_atomic(output / "summary.json", summary)
    _, mask = full.prepare_cache(*args)
    assert mask.all()
    summary["artifacts"][path.name]["sha256"] = "0" * 64
    full.pilot._json_atomic(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        full.prepare_cache(*args)


def test_final_center_batch_keeps_historical_four_image_shape():
    pixels = [torch.full((3, 384, 384), float(i)) for i in range(3)]
    batch = full.padded_center_batch(pixels)
    assert batch.shape == (4, 3, 384, 384)
    assert torch.equal(batch[3], pixels[-1])
    for i in range(3):
        assert torch.equal(batch[i], pixels[i])
    with pytest.raises(ValueError):
        full.padded_center_batch([])
    with pytest.raises(ValueError):
        full.padded_center_batch(pixels, microbatch=3)
