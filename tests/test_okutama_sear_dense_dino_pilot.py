from __future__ import annotations

import copy
import io
import json
import types
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from experiments import pilot_okutama_sear_dense_dino as pilot
from experiments.cache_okutama_video_features import LockedClipDataset


def test_default_contract_is_pilot_only_and_exact():
    protocol = json.loads(pilot.PROTOCOL.read_text())
    pilot.validate_protocol(protocol)
    assert protocol["pilot_rows"] == 32
    assert not protocol["authorization"]["full_cache_extraction"]
    assert protocol["cls_replay_max_absolute_difference"] == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("image_size", 378),
        ("patch_grid", [16, 16]),
        ("center_slot", 7),
        ("cls_replay_max_absolute_difference", 1e-3),
        ("pilot_rows", 4977),
    ],
)
def test_contract_rejects_silent_extraction_changes(key, value):
    protocol = json.loads(pilot.PROTOCOL.read_text())
    protocol[key] = value
    with pytest.raises(RuntimeError, match="contract"):
        pilot.validate_protocol(protocol)


def test_contract_forbids_optimizer_steps_or_broader_authorization():
    protocol = json.loads(pilot.PROTOCOL.read_text())
    changed = copy.deepcopy(protocol)
    changed["authorization"]["classifier_fitting"] = True
    with pytest.raises(RuntimeError, match="authorization"):
        pilot.validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["ordinary_head_benchmarks"]["optimizer_steps"] = 1
    with pytest.raises(RuntimeError, match="optimizer"):
        pilot.validate_protocol(changed)


def test_label_blind_selection_is_identity_deterministic_and_covers_strata():
    ids = [f"sample-{i}" for i in range(20)]
    scenarios = [str(i // 10) for i in range(20)]
    fallback = np.asarray([i % 2 == 0 for i in range(20)])
    selected = pilot.select_label_blind(ids, scenarios, fallback, 8, "test")
    assert len(selected) == 8 and len(set(selected)) == 8
    assert {(scenarios[i], bool(fallback[i])) for i in selected} == {
        (str(s), f) for s in range(2) for f in [False, True]
    }
    order = np.arange(20)[::-1]
    again = pilot.select_label_blind(
        [ids[i] for i in order], [scenarios[i] for i in order], fallback[order], 8, "test"
    )
    assert {ids[i] for i in selected} == {ids[order[i]] for i in again}
    with pytest.raises(ValueError, match="stratum"):
        pilot.select_label_blind(ids, scenarios, fallback, 3, "test")


def test_selection_rejects_duplicate_ids_and_nonboolean_fallback():
    with pytest.raises(ValueError, match="invalid"):
        pilot.select_label_blind(["a", "a"], ["s", "s"], np.zeros(2, bool), 1, "t")
    with pytest.raises(ValueError, match="invalid"):
        pilot.select_label_blind(["a", "b"], ["s", "s"], np.zeros(2), 1, "t")


def make_zip(tmp_path):
    image = Image.new("RGB", (1280, 720), (33, 65, 99))
    raw = io.BytesIO()
    image.save(raw, format="JPEG")
    path = tmp_path / "frames.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("permitted/frame.jpg", raw.getvalue())
    row = {
        "valid_frame": "1",
        "image_member": "permitted/frame.jpg",
        "bbox_xmin": "-10",
        "bbox_ymin": "40",
        "bbox_xmax": "70",
        "bbox_ymax": "140",
    }
    return path, row


def test_center_preprocess_exactly_replays_original_clip(tmp_path):
    path, row = make_zip(tmp_path)
    frames = {"a": [{**row, "time_index": str(i)} for i in range(16)]}
    allowed = {row["image_member"]}
    with zipfile.ZipFile(path) as archive:
        pixels, valid, byte_count = pilot.center_pixels(archive, row, allowed)
    original = LockedClipDataset(path, [{"sample_id": "a"}], frames, allowed)
    result = original[0]
    assert valid and byte_count > 0
    assert torch.equal(pixels, result["actual_pixels"][:, 8])
    assert torch.equal(pixels, result["repeated_pixels"][:, 8])
    original._archive.close()


def test_allowlist_is_checked_before_image_payload_read(tmp_path):
    path, row = make_zip(tmp_path)
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(RuntimeError, match="allowlist"):
            pilot.center_pixels(archive, row, set())
        pixels, valid, byte_count = pilot.center_pixels(archive, {**row, "valid_frame": "0"}, set())
    assert not valid and byte_count == 0 and torch.count_nonzero(pixels) == 0


class FakeBackbone(nn.Module):
    def __init__(self, tokens=730, fill=2.0):
        super().__init__()
        self.tokens, self.fill = tokens, fill

    def forward(self, pixel_values, return_dict):
        assert return_dict
        dense = torch.full((len(pixel_values), self.tokens, 768), self.fill)
        dense[:, 0] = 7
        return types.SimpleNamespace(last_hidden_state=dense)


def test_dense_forward_keeps_cls_and_row_major_patch_order():
    encoder = nn.Module()
    encoder.backbone = FakeBackbone()
    encoder.eval()
    cls, patches = pilot.dense_forward(encoder, torch.zeros(2, 3, 384, 384), device="cpu")
    assert cls.shape == (2, 768) and patches.shape == (2, 27, 27, 768)
    assert cls.dtype == patches.dtype == np.float16
    assert (cls == 7).all() and (patches == 2).all()


@pytest.mark.parametrize(("tokens", "fill"), [(729, 2.0), (730, float("nan")), (730, 1e10)])
def test_dense_forward_rejects_bad_shape_nonfinite_or_fp16_overflow(tokens, fill):
    encoder = nn.Module()
    encoder.backbone = FakeBackbone(tokens, fill)
    encoder.eval()
    with pytest.raises(RuntimeError):
        pilot.dense_forward(encoder, torch.zeros(1, 3, 384, 384), device="cpu")


def test_dense_forward_rejects_training_and_wrong_pixel_contract():
    encoder = nn.Module()
    encoder.backbone = FakeBackbone()
    with pytest.raises(RuntimeError, match="evaluation"):
        pilot.dense_forward(encoder, torch.zeros(1, 3, 384, 384), device="cpu")
    encoder.eval()
    with pytest.raises(ValueError, match="384"):
        pilot.dense_forward(encoder, torch.zeros(1, 3, 378, 378), device="cpu")
    with pytest.raises(ValueError, match="finite"):
        pilot.dense_forward(encoder, torch.full((1, 3, 384, 384), float("nan")), device="cpu")


def test_exact_replay_has_no_tolerance_escape():
    values = np.ones(768, np.float16)
    assert pilot.assert_exact_replay(values, values.copy(), description="test") == 0
    changed = values.copy()
    changed[0] = np.nextafter(changed[0], np.float16(2))
    with pytest.raises(RuntimeError, match="exact replay failed"):
        pilot.assert_exact_replay(changed, values, description="test")
    with pytest.raises(RuntimeError, match="shape/dtype"):
        pilot.assert_exact_replay(values.astype(np.float32), values, description="test")


@pytest.mark.parametrize("kind", ["cnn", "transformer"])
def test_ordinary_surrogates_are_small_and_receive_finite_gradients(kind):
    torch.manual_seed(0)
    model = pilot.OrdinaryDenseResourceHead(kind)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000
    logits = model(torch.randn(1, 729, 768), torch.randn(1, 1536))
    assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
