from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from hac import image_encoders as images


def _snapshot(tmp_path, monkeypatch):
    specification = {}
    for name, value in (
        ("config.json", b"{}"),
        ("preprocessor_config.json", b"{ }"),
        ("model.safetensors", b"small mocked model"),
    ):
        (tmp_path / name).write_bytes(value)
        specification[name] = {
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    monkeypatch.setattr(images, "DINO_FILES", specification)
    return tmp_path


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(model_type="dinov2", hidden_size=768, patch_size=14)

    def forward(self, pixel_values, return_dict=True):
        assert return_dict
        tokens = pixel_values.new_zeros((len(pixel_values), 730, 768))
        tokens[:, 0] = 7
        tokens[:, 1:] = 19
        return SimpleNamespace(last_hidden_state=tokens)


def test_snapshot_receipt_binds_all_files_and_root(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch)
    receipt = images.validate_dinov2_snapshot(root)
    assert receipt["revision"] == images.DINO_REVISION
    assert images.validate_dinov2_lock_receipt(root, receipt) == receipt
    bad = {**receipt, "path": str(root / "other")}
    with pytest.raises(RuntimeError, match="root"):
        images.validate_dinov2_lock_receipt(root, bad)
    with pytest.raises(RuntimeError, match="receipt"):
        images.validate_dinov2_lock_receipt(root, None)


def test_snapshot_rejects_same_size_tampering(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch)
    (root / "config.json").write_bytes(b"[]")
    with pytest.raises(RuntimeError, match="hash"):
        images.validate_dinov2_snapshot(root)


def test_snapshot_rejects_unbound_adapter_configuration(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch)
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unbound executable"):
        images.validate_dinov2_snapshot(root)


def test_loader_is_local_frozen_and_returns_final_cls(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch)
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return FakeBackbone()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)),
    )
    encoder, receipt = images.load_dinov2_encoder(root, device="cpu")
    assert calls == [
        (
            str(root.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "attn_implementation": "sdpa",
            },
        )
    ]
    assert not encoder.training and not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    tokens = encoder(torch.zeros(2, 3, 384, 384))
    assert tokens.shape == (2, 768) and (tokens == 7).all()
    assert receipt["source_processor_defaults_applied"] is False
    assert receipt["unpatchified_trailing_pixels_per_axis"] == 6


def test_loader_rejects_snapshot_changed_while_loading(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch)

    def load(*args, **kwargs):
        (root / "config.json").write_bytes(b"[]")
        return FakeBackbone()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)),
    )
    with pytest.raises(RuntimeError, match="hash"):
        images.load_dinov2_encoder(root, device="cpu")


def test_cls_wrapper_rejects_wrong_size_nonfinite_and_train_mode():
    encoder = images.FrozenDinoCLS(FakeBackbone())
    with pytest.raises(ValueError, match="384"):
        encoder(torch.zeros(1, 3, 224, 224))
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.full((1, 3, 384, 384), float("nan")))
    encoder.train()
    with pytest.raises(RuntimeError, match="evaluation"):
        encoder(torch.zeros(1, 3, 384, 384))
