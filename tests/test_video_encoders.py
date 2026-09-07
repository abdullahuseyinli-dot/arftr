from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from hac import video_encoders as video


def test_letterbox_preserves_body_and_normalizes_rgb_in_cthw_order():
    frame = np.full((4, 2, 3), [255, 0, 127], dtype=np.uint8)
    result = video.preprocess_rgb_clip([frame, frame], size=8)
    assert result.shape == (3, 2, 8, 8)
    assert result.dtype == torch.float32 and result.is_contiguous()
    mean, std = torch.tensor(video.IMAGENET_MEAN), torch.tensor(video.IMAGENET_STD)
    body = (torch.tensor([255, 0, 127]) / 255 - mean) / std
    padding = (torch.tensor(video.PADDING_RGB) / 255 - mean) / std
    torch.testing.assert_close(result[:, 0, 0, 2], body)
    torch.testing.assert_close(result[:, 1, -1, 5], body)
    torch.testing.assert_close(result[:, 0, 0, 0], padding)
    assert torch.equal(result[:, 0], result[:, 1])


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        [np.zeros((2, 2, 3))] * 2,
        [Image.new("L", (2, 2))] * 2,
    ],
)
def test_preprocessing_rejects_ambiguous_frame_contract(frames):
    with pytest.raises(ValueError):
        video.preprocess_rgb_clip(frames)


def test_spatial_pooling_preserves_tubelet_and_region_order():
    # 2 temporal cells, each with a 4x4 grid, pooled into disjoint 2x2 regions.
    values = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1).expand(1, 32, 768)
    result = video.pool_spatiotemporal_tokens(values, frames=4, image_size=64, spatial_grid=2)
    assert result.shape == (1, 2, 4, 768)
    torch.testing.assert_close(result[0, 0, :, 0], torch.tensor([2.5, 4.5, 10.5, 12.5]))
    torch.testing.assert_close(result[0, 1, :, 0], torch.tensor([18.5, 20.5, 26.5, 28.5]))


def test_pooling_rejects_wrong_dense_layout_and_nonfinite_features():
    with pytest.raises(ValueError, match="Expected dense"):
        video.pool_spatiotemporal_tokens(torch.zeros(1, 4609, 768))
    with pytest.raises(ValueError, match="finite"):
        video.pool_spatiotemporal_tokens(
            torch.full((1, 4, 768), float("nan")), frames=2, image_size=32, spatial_grid=2
        )


def test_clean_state_uses_only_ema_and_rejects_colliding_names():
    weight = torch.ones(2, 2)
    cleaned = video.clean_ema_encoder_state(
        {"ema_encoder": {"module.backbone.weight": weight}, "predictor": {"unused": weight}}
    )
    assert list(cleaned) == ["weight"] and cleaned["weight"] is weight
    with pytest.raises(RuntimeError, match="collision"):
        video.clean_ema_encoder_state({"ema_encoder": {"module.weight": weight, "weight": weight}})
    with pytest.raises(RuntimeError, match="ema_encoder"):
        video.clean_ema_encoder_state({"target_encoder": {"weight": weight}})


def test_module_source_accepts_only_line_ending_changes(tmp_path, monkeypatch):
    path = tmp_path / "module.py"
    path.write_bytes(b"x = 1\r\n")
    monkeypatch.setattr(video, "_git", lambda *args: b"x = 1\n")
    receipt = video.verify_python_source(tmp_path, path)
    assert receipt["line_endings_normalized"]
    path.write_bytes(b"x = 2\r\n")
    with pytest.raises(RuntimeError, match="differs"):
        video.verify_python_source(tmp_path, path)
    with pytest.raises(RuntimeError, match="outside"):
        video.verify_python_source(tmp_path / "nested", path)


def test_foreign_preimported_namespace_is_rejected(tmp_path, monkeypatch):
    module = types.ModuleType("src")
    module.__path__ = [str(tmp_path.parent)]
    monkeypatch.setattr(video, "_upstream_modules", lambda: [("src", module)])
    with pytest.raises(RuntimeError, match="foreign origin"):
        video._verify_module_origins(tmp_path)


def test_checkpoint_digest_cannot_be_overridden(tmp_path):
    with pytest.raises(RuntimeError, match="pinned acquisition"):
        video.validate_checkpoint(tmp_path / "unused.pt", "0" * 64)


def test_loader_does_not_accept_incomplete_encoder_state(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    module = types.SimpleNamespace(vit_base=lambda **kwargs: model)
    monkeypatch.setattr(video, "validate_checkpoint", lambda *args: {})
    monkeypatch.setattr(video, "import_pinned_encoder", lambda *args: (module, {}))
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"ema_encoder": {"weight": model.weight}}
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        video.load_vjepa21_encoder(tmp_path, Path("unused.pt"), device="cpu")


def test_synthetic_smoke_clip_has_motion_without_external_inputs():
    from smoke_okutama_video_encoder import synthetic_clip

    first, second = synthetic_clip(), synthetic_clip()
    assert len(first) == 16
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not np.array_equal(first[0], first[-1])
