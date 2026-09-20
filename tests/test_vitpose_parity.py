from __future__ import annotations

import torch
from torch import nn

from hac.vitpose_parity import (
    OFFICIAL_LAYER_NORM_EPS,
    compare_state_dicts_exact,
    convert_original_state_dict,
    corrected_config_payload,
    validate_layer_norm_lock,
)


def test_corrected_config_materializes_official_layer_norm_epsilon():
    corrected = corrected_config_payload(
        {
            "model_type": "vitpose",
            "use_simple_decoder": False,
            "backbone_config": {
                "model_type": "vitpose_backbone",
                "part_features": 0,
                "out_features": ["stage12"],
                "out_indices": [12],
            },
        }
    )
    backbone = corrected["backbone_config"]
    assert backbone["layer_norm_eps"] == OFFICIAL_LAYER_NORM_EPS
    assert backbone["image_size"] == [256, 192]
    assert backbone["patch_size"] == [16, 16]
    assert backbone["num_hidden_layers"] == 12
    assert corrected["use_simple_decoder"] is False


def test_original_qkv_conversion_is_lossless_and_exact():
    qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    original = {
        "backbone.blocks.0.attn.qkv.weight": qkv,
        "backbone.blocks.0.norm1.weight": torch.ones(3),
    }
    # The production converter is pinned to hidden size 768, so exercise a
    # normal non-QKV mapping here and use a real-sized narrow QKV value below.
    original["backbone.blocks.0.attn.qkv.weight"] = torch.arange(
        3 * 768 * 2, dtype=torch.float32
    ).reshape(3 * 768, 2)
    converted, ignored = convert_original_state_dict(original)
    assert ignored == []
    query = converted["backbone.encoder.layer.0.attention.attention.query.weight"]
    key = converted["backbone.encoder.layer.0.attention.attention.key.weight"]
    value = converted["backbone.encoder.layer.0.attention.attention.value.weight"]
    assert torch.equal(torch.cat((query, key, value)), original["backbone.blocks.0.attn.qkv.weight"])
    assert "backbone.encoder.layer.0.layernorm_before.weight" in converted


def test_exact_state_comparison_detects_any_value_change():
    expected = {"x": torch.tensor([1.0, 2.0])}
    assert compare_state_dicts_exact(expected, {"x": expected["x"].clone()})["passed"]
    result = compare_state_dicts_exact(expected, {"x": torch.tensor([1.0, 3.0])})
    assert result["passed"] is False
    assert result["value_mismatches"] == ["x"]


def test_layer_norm_lock_rejects_transformers_default():
    corrected = nn.Sequential(nn.LayerNorm(4, eps=OFFICIAL_LAYER_NORM_EPS))
    default = nn.Sequential(nn.LayerNorm(4, eps=1e-12))
    assert validate_layer_norm_lock(corrected)["passed"] is True
    assert validate_layer_norm_lock(default)["passed"] is False
