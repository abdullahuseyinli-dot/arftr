from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hac.absolute_body_evidence import (
    DINO_INPUT_SIZE,
    MULTISCALE_WHOLE,
    FrozenAbsoluteDinoCLS,
    compose_source_images,
    letterbox_rgb,
    make_crop_geometry,
    prepare_crops,
    prepare_source_crops,
    source_actor_boxes,
)


def test_crop_before_resize_regions_and_source_parity():
    box720 = (100.25, 200.5, 140.75, 300.25)
    boxes = source_actor_boxes(box720, (3840, 2160))
    assert np.allclose(np.asarray(boxes["N"]), 3 * np.asarray(boxes["R"]))
    upper720 = make_crop_geometry(boxes["R"], source_size=(1280, 720), source_id="R", region_id="upper")
    upper4k = make_crop_geometry(boxes["N"], source_size=(3840, 2160), source_id="N", region_id="upper")
    assert np.allclose(np.asarray(upper4k.continuous_box), 3 * np.asarray(upper720.continuous_box))
    assert upper720.receipt()["crop_before_model_resize"] is True


def test_source_crop_inventory_is_complete_and_reuses_middle_scale():
    native = np.zeros((216, 384, 3), dtype=np.uint8)
    supplied = np.zeros((720, 1280, 3), dtype=np.uint8)
    sources = compose_source_images(supplied, native)
    boxes = source_actor_boxes((100, 100, 130, 180), (384, 216))
    crops, receipts = prepare_crops(sources, boxes)
    assert {f"{source}_{region}" for source in "JRN" for region in ("whole", "upper", "lower")} <= set(crops)
    assert {f"R_whole_{str(scale).replace('.', 'p')}" for scale in MULTISCALE_WHOLE} <= set(crops)
    assert receipts["R_whole_1p25"]["reuses"] == "R_whole"
    assert all(value.dtype == np.uint8 and value.ndim == 3 for value in crops.values())


def test_supplied_source_can_be_prepared_when_native_source_is_missing():
    supplied = np.zeros((720, 1280, 3), dtype=np.uint8)
    crops, receipts = prepare_source_crops(
        supplied,
        (100, 100, 130, 180),
        source_id="J",
    )
    assert set(crops) == {"J_whole", "J_upper", "J_lower"}
    assert set(receipts) == set(crops)
    with pytest.raises(ValueError, match="Only R"):
        prepare_source_crops(
            supplied,
            (100, 100, 130, 180),
            source_id="J",
            include_multiscale=True,
        )


def test_letterbox_uses_complete_patch_grid_and_is_deterministic():
    raw = np.arange(17 * 9 * 3, dtype=np.uint8).reshape(9, 17, 3)
    first, second = letterbox_rgb(raw), letterbox_rgb(raw)
    assert first.shape == (DINO_INPUT_SIZE, DINO_INPUT_SIZE, 3)
    assert DINO_INPUT_SIZE % 14 == 0
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="complete27x27"):
        letterbox_rgb(raw, output_size=384)


def test_frozen_adapter_has_complete_grid_and_l2_normalizes_cls():
    class Backbone(torch.nn.Module):
        def forward(self, *, pixel_values, return_dict):
            assert return_dict
            hidden = torch.ones((len(pixel_values), 730, 768), device=pixel_values.device)
            hidden[:, 0] *= 3
            return SimpleNamespace(last_hidden_state=hidden)

    model = FrozenAbsoluteDinoCLS(Backbone()).eval()
    output = model(torch.zeros((2, 3, 378, 378)))
    assert output.shape == (2, 768)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(2))
