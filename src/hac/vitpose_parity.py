"""Strict utilities for the pinned classic ViTPose-B transport.

The public converted checkpoint was emitted with an incomplete backbone
configuration: ``layer_norm_eps`` is absent, so Transformers supplies 1e-12
instead of the official implementation's 1e-6.  These helpers correct that
configuration explicitly and prove that the converted tensors are an exact,
lossless renaming/splitting of the pinned original-checkpoint mirror.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

OFFICIAL_LAYER_NORM_EPS = 1e-6
ORIGINAL_CHECKPOINT_SHA256 = "2e849e1f1dbb5b87191eda7171f1b16468d5d082a7380e93e94b7ce76a061679"
CONVERTED_CHECKPOINT_SHA256 = "cd9a4e6cefc33c51ddcc32dd509fc4114b5845256463a69a10ea2dfde402b2f8"

_ORIGINAL_TO_CONVERTED_KEY_MAPPING = (
    (r"patch_embed\.proj", "embeddings.patch_embeddings.projection"),
    (r"pos_embed", "embeddings.position_embeddings"),
    (r"blocks", "encoder.layer"),
    (r"attn\.proj", "attention.output.dense"),
    (r"attn", "attention.self"),
    (r"norm1", "layernorm_before"),
    (r"norm2", "layernorm_after"),
    (r"last_norm", "layernorm"),
    (r"keypoint_head", "head"),
    (r"final_layer", "conv"),
)

_EXPECTED_BACKBONE_CONFIG: dict[str, Any] = {
    "model_type": "vitpose_backbone",
    "image_size": [256, 192],
    "patch_size": [16, 16],
    "num_channels": 3,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "mlp_ratio": 4,
    "num_experts": 1,
    "part_features": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "qkv_bias": True,
    "out_features": ["stage12"],
    "out_indices": [12],
    "layer_norm_eps": OFFICIAL_LAYER_NORM_EPS,
}


def corrected_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fully explicit classic ViTPose-B config with official epsilon.

    Any supplied field that contradicts the pinned architecture fails closed;
    absent fields are materialized so a later library default cannot silently
    alter the model definition.
    """

    corrected = copy.deepcopy(dict(payload))
    if corrected.get("model_type") != "vitpose":
        raise ValueError("Expected a ViTPose model configuration")
    if corrected.get("use_simple_decoder") is not False:
        raise ValueError("The body-witness lock requires the classic decoder")

    backbone = copy.deepcopy(dict(corrected.get("backbone_config", {})))
    for field, expected in _EXPECTED_BACKBONE_CONFIG.items():
        if field in backbone and backbone[field] != expected:
            raise ValueError(
                f"Backbone configuration mismatch for {field}: "
                f"{backbone[field]!r} != {expected!r}"
            )
        backbone[field] = expected
    corrected["backbone_config"] = backbone
    corrected["use_pretrained_backbone"] = False
    corrected["use_timm_backbone"] = False
    return corrected


def _rename_original_key(key: str) -> str:
    renamed = key
    for pattern, replacement in _ORIGINAL_TO_CONVERTED_KEY_MAPPING:
        renamed = re.sub(pattern, replacement, renamed)
    return renamed


def convert_original_state_dict(
    original_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert official-format keys without changing any tensor values.

    The only shape change is splitting each combined QKV tensor into the three
    learned linear tensors used by Transformers.  Inference-unused auxiliary
    heads and ``cls_token`` are reported rather than silently loaded.
    """

    converted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    hidden_size = 768
    for key, value in original_state_dict.items():
        new_key = _rename_original_key(key)
        if "associate_heads" in new_key or "backbone.cls_token" in new_key:
            ignored.append(key)
            continue
        if "qkv" in new_key:
            if value.shape[0] != 3 * hidden_size:
                raise ValueError(f"Unexpected QKV first dimension for {key}: {value.shape}")
            for index, component in enumerate(("query", "key", "value")):
                target = new_key.replace("self.qkv", f"attention.{component}")
                converted[target] = value[index * hidden_size : (index + 1) * hidden_size]
            continue
        if "head" in new_key:
            deconv_match = re.search(r"deconv_layers\.(0|3)\.weight", new_key)
            if deconv_match:
                number = int(deconv_match.group(1)) // 3 + 1
                new_key = re.sub(
                    r"deconv_layers\.(0|3)\.weight", f"deconv{number}.weight", new_key
                )
            else:
                bn_match = re.search(
                    r"deconv_layers\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)",
                    new_key,
                )
                if bn_match:
                    number = int(bn_match.group(1)) // 3 + 1
                    new_key = re.sub(
                        r"deconv_layers\.\d+\.", f"batchnorm{number}.", new_key
                    )
        if new_key in converted:
            raise ValueError(f"Duplicate converted key: {new_key}")
        converted[new_key] = value
    return converted, ignored


def compare_state_dicts_exact(
    expected: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Compare all keys, shapes, dtypes, and values bit-for-bit."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    value_mismatches: list[str] = []
    compared_values = 0
    compared_bytes = 0
    for key in sorted(expected_keys & actual_keys):
        left = expected[key]
        right = actual[key]
        if left.shape != right.shape:
            shape_mismatches.append(key)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(key)
            continue
        compared_values += left.numel()
        compared_bytes += left.numel() * left.element_size()
        if not torch.equal(left, right):
            value_mismatches.append(key)
    passed = not (missing or unexpected or shape_mismatches or dtype_mismatches or value_mismatches)
    return {
        "passed": passed,
        "expected_tensors": len(expected),
        "actual_tensors": len(actual),
        "compared_values": compared_values,
        "compared_bytes": compared_bytes,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
    }


def validate_layer_norm_lock(model: nn.Module) -> dict[str, Any]:
    """Require every instantiated LayerNorm to use the official epsilon."""

    layer_norms = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    mismatches = [name for name, module in layer_norms if module.eps != OFFICIAL_LAYER_NORM_EPS]
    return {
        "passed": bool(layer_norms) and not mismatches,
        "layer_norm_count": len(layer_norms),
        "required_epsilon": OFFICIAL_LAYER_NORM_EPS,
        "mismatches": mismatches,
    }


def load_locked_vitpose(model_directory: Path, *, local_files_only: bool = True) -> nn.Module:
    """Load the corrected local checkpoint and fail closed on semantic drift."""

    from transformers import VitPoseForPoseEstimation

    model = VitPoseForPoseEstimation.from_pretrained(
        Path(model_directory), local_files_only=local_files_only, use_safetensors=True
    )
    validation = validate_layer_norm_lock(model)
    if not validation["passed"]:
        raise RuntimeError(f"ViTPose LayerNorm lock failed: {validation}")
    if model.config.use_simple_decoder:
        raise RuntimeError("ViTPose classic-decoder lock failed")
    return model
