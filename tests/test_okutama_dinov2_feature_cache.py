from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from experiments import cache_okutama_dinov2_features as cache


def test_per_frame_encoder_preserves_exact_native_pixel_order():
    pixels = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1).expand(3, 16, 384, 384)
    calls = []

    def encoder(batch):
        calls.append(batch.clone())
        return batch[:, 0, 0, 0, None].expand(-1, 768)

    values = cache.encode_native_clip(encoder, pixels, device="cpu")
    assert values.shape == (16, 1, 768) and values.dtype == np.float16
    np.testing.assert_array_equal(values[:, 0, 0], np.arange(16))
    assert len(calls) == 4
    torch.testing.assert_close(torch.cat(calls), pixels.permute(1, 0, 2, 3))


def test_encoder_rejects_half_precision_overflow():
    pixels = torch.zeros(3, 16, 384, 384)

    def encoder(batch):
        return torch.full((len(batch), 768), 1e10)

    with pytest.raises(RuntimeError, match="overflowed"):
        cache.encode_native_clip(encoder, pixels, device="cpu")


def test_cache_resume_preserves_completed_rows_and_rejects_changed_request(tmp_path):
    request = {"sample_ids": ["a", "b"], "precision": "bfloat16"}
    features, completed, validity, _ = cache.prepare_cache(tmp_path, request, 2)
    features[0] = 11
    completed[0] = True
    validity[0, 0] = True
    cache._save_state(tmp_path, features, completed, validity)
    del features
    features, completed, validity, _ = cache.prepare_cache(tmp_path, request, 2)
    assert completed.tolist() == [True, False]
    assert validity.tolist() == [[True], [False]]
    assert (features[0] == 11).all()
    with pytest.raises(RuntimeError, match="different extraction request"):
        cache.prepare_cache(tmp_path, {**request, "precision": "float32"}, 2)


def test_cache_rejects_dtype_tamper_and_orphan_output(tmp_path):
    output = tmp_path / "cache"
    features, _, _, _ = cache.prepare_cache(output, {"one": 1}, 2)
    del features
    np.save(output / "completed.npy", np.zeros(2, dtype=np.int64))
    with pytest.raises(RuntimeError, match="dtype"):
        cache.prepare_cache(output, {"one": 1}, 2)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "retain.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        cache.prepare_cache(orphan, {"one": 1}, 2)


def test_snapshot_lock_checked_after_shared_validation_before_dataset(tmp_path, monkeypatch):
    path = tmp_path / "extraction.json"
    declared = {
        "inputs": {"checkpoint": {"path": str(tmp_path / "video.pt")}},
        "upstream": {"path": str(tmp_path / "upstream")},
    }
    path.write_text(json.dumps(declared), encoding="utf-8")
    order = []

    def shared(args):
        order.append("shared")
        assert args.checkpoint.name == "video.pt"
        return declared, [], {}, set()

    def snapshot(root, receipt):
        order.append("snapshot")
        assert receipt is None
        raise RuntimeError("missing locked DINO receipt")

    monkeypatch.setattr(cache, "_validate_and_select", shared)
    monkeypatch.setattr(cache, "validate_dinov2_lock_receipt", snapshot)
    args = argparse.Namespace(extraction_lock=path, model_root=tmp_path)
    with pytest.raises(RuntimeError, match="missing locked DINO"):
        cache.validate_inputs(args)
    assert order == ["shared", "snapshot"]


def test_run_is_resumable_without_loading_model_again(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        archive=tmp_path / "unused.zip",
        manifest_dir=tmp_path / "manifest",
        extraction_lock=lock_path,
        model_root=tmp_path / "model",
        output_dir=tmp_path / "cache",
        mode="full",
        workers=0,
        checkpoint_interval=1,
    )
    clips = [{"sample_id": "a", "valid": True}, {"sample_id": "b", "valid": False}]
    lock = {"inputs": {"dinov2_snapshot": {}}}
    monkeypatch.setattr(cache, "validate_inputs", lambda args: (lock, clips, {}, set(), {}))
    monkeypatch.setattr(cache, "validate_dinov2_lock_receipt", lambda *args: {})
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    for name, value in (
        ("is_available", True),
        ("is_bf16_supported", True),
        ("max_memory_allocated", 100),
        ("max_memory_reserved", 200),
        ("get_device_name", "mock CUDA"),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args: None)
    for backend, name in (
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_tf32"),
    ):
        monkeypatch.setattr(backend, name, getattr(backend, name))
    real_loader = cache.DataLoader

    def cpu_loader(*args, **kwargs):
        return real_loader(*args, **{**kwargs, "pin_memory": False})

    monkeypatch.setattr(cache, "DataLoader", cpu_loader)
    model_loads = []

    def load(*args):
        model_loads.append(True)
        return object(), {"model": "mock_dinov2"}

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, archive, selected, frames, allowed):
            self.selected = selected

        def __len__(self):
            return len(self.selected)

        def __getitem__(self, index):
            return {
                "row_index": index,
                "actual_valid": self.selected[index]["valid"],
                "actual_pixels": torch.tensor([7.0]),
                "encoded_bytes": 10,
            }

    monkeypatch.setattr(cache, "load_dinov2_encoder", load)
    monkeypatch.setattr(cache, "LockedClipDataset", MockDataset)
    monkeypatch.setattr(
        cache, "encode_native_clip", lambda *args: np.full(cache.CACHE_SHAPE, 7, np.float16)
    )
    summary = cache.run(args)
    assert summary["status"] == "OKUTAMA_DINOV2_FROZEN_FEATURE_CACHE_COMPLETE"
    assert summary["arms"][cache.ARM]["shape"] == [2, 16, 1, 768]
    assert summary["arms"][cache.ARM]["valid_rows"] == 1
    assert summary["candidate_fits"] == 0
    assert len(model_loads) == 1
    again = cache.run(args)
    assert again["processed_clips_this_invocation"] == 0
    assert len(model_loads) == 1
    features = np.load(args.output_dir / f"{cache.ARM}.npy")
    assert (features[0] == 7).all() and (features[1] == 0).all()


def test_completed_cache_rejects_finite_feature_tampering(tmp_path):
    request = {"sample_ids": ["a"]}
    features, completed, validity, request_hash = cache.prepare_cache(tmp_path, request, 1)
    features[0] = 7
    completed[0] = True
    validity[0] = True
    cache._save_state(tmp_path, features, completed, validity)
    summary = {
        "request_sha256": request_hash,
        "arms": {cache.ARM: {"sha256": cache.sha256_file(tmp_path / f"{cache.ARM}.npy")}},
        "completed": {"sha256": cache.sha256_file(tmp_path / "completed.npy")},
        "validity": {"sha256": cache.sha256_file(tmp_path / "validity.npy")},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    features[0, 0, 0, 0] = 8
    features.flush()
    del features
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        cache.prepare_cache(tmp_path, request, 1)
