"""Correct and verify the pinned classic ViTPose-B checkpoint transport.

This is a bounded, label-free checkpoint audit.  It performs no Okutama pose
inference and no action-model update.  The forward reference is a direct
functional transcription of the pinned official backbone and classic head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hac.vitpose_parity import (  # noqa: E402
    CONVERTED_CHECKPOINT_SHA256,
    OFFICIAL_LAYER_NORM_EPS,
    ORIGINAL_CHECKPOINT_SHA256,
    compare_state_dicts_exact,
    convert_original_state_dict,
    corrected_config_payload,
    load_locked_vitpose,
    validate_layer_norm_lock,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runs/research_20260913/vitpose_parity_v1"
REFERENCE_REVISION = "c050ed29112da7704797cc1a65af0234b525010d"
CONVERTER_REVISION = "1bea6c1644c51b1644cdce275f150413fe746202"
SOURCE_MIRROR_REVISION = "e8141930a5f66b269e22713bbd9ee3f06bbe89b0"
CONVERTED_REVISION = "95be2991424e646950d656bb7fc15ec9be119700"

# Fixed before the first reference-forward comparison.  This accommodates
# round-off from the official combined-QKV GEMM versus three equivalent GEMMs
# in Transformers while remaining far below heatmap decision-scale changes.
MAX_ABS_HEATMAP_TOLERANCE = 1e-4
MEAN_ABS_HEATMAP_TOLERANCE = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _official_reference_heatmaps(
    pixels: torch.Tensor, state: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Pinned official ViT + two-deconvolution head, expressed functionally."""

    x = F.conv2d(
        pixels,
        state["backbone.patch_embed.proj.weight"],
        state["backbone.patch_embed.proj.bias"],
        stride=(16, 16),
        padding=(2, 2),
    )
    batch, _, height, width = x.shape
    x = x.flatten(2).transpose(1, 2)
    position = state["backbone.pos_embed"]
    x = x + position[:, 1:] + position[:, :1]
    heads = 12
    head_size = 64
    for index in range(12):
        prefix = f"backbone.blocks.{index}"
        normalized = F.layer_norm(
            x,
            (768,),
            state[f"{prefix}.norm1.weight"],
            state[f"{prefix}.norm1.bias"],
            OFFICIAL_LAYER_NORM_EPS,
        )
        qkv = F.linear(
            normalized,
            state[f"{prefix}.attn.qkv.weight"],
            state[f"{prefix}.attn.qkv.bias"],
        )
        qkv = qkv.reshape(batch, -1, 3, heads, head_size).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * (head_size**-0.5)) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch, -1, 768)
        attended = F.linear(
            attended,
            state[f"{prefix}.attn.proj.weight"],
            state[f"{prefix}.attn.proj.bias"],
        )
        x = x + attended
        normalized = F.layer_norm(
            x,
            (768,),
            state[f"{prefix}.norm2.weight"],
            state[f"{prefix}.norm2.bias"],
            OFFICIAL_LAYER_NORM_EPS,
        )
        hidden = F.linear(
            normalized,
            state[f"{prefix}.mlp.fc1.weight"],
            state[f"{prefix}.mlp.fc1.bias"],
        )
        hidden = F.gelu(hidden, approximate="none")
        hidden = F.linear(
            hidden,
            state[f"{prefix}.mlp.fc2.weight"],
            state[f"{prefix}.mlp.fc2.bias"],
        )
        x = x + hidden
    x = F.layer_norm(
        x,
        (768,),
        state["backbone.last_norm.weight"],
        state["backbone.last_norm.bias"],
        OFFICIAL_LAYER_NORM_EPS,
    )
    x = x.permute(0, 2, 1).reshape(batch, 768, height, width).contiguous()
    x = F.conv_transpose2d(
        x,
        state["keypoint_head.deconv_layers.0.weight"],
        stride=2,
        padding=1,
    )
    x = F.batch_norm(
        x,
        state["keypoint_head.deconv_layers.1.running_mean"],
        state["keypoint_head.deconv_layers.1.running_var"],
        state["keypoint_head.deconv_layers.1.weight"],
        state["keypoint_head.deconv_layers.1.bias"],
        training=False,
        eps=1e-5,
    )
    x = F.relu(x)
    x = F.conv_transpose2d(
        x,
        state["keypoint_head.deconv_layers.3.weight"],
        stride=2,
        padding=1,
    )
    x = F.batch_norm(
        x,
        state["keypoint_head.deconv_layers.4.running_mean"],
        state["keypoint_head.deconv_layers.4.running_var"],
        state["keypoint_head.deconv_layers.4.weight"],
        state["keypoint_head.deconv_layers.4.bias"],
        training=False,
        eps=1e-5,
    )
    x = F.relu(x)
    return F.conv2d(
        x,
        state["keypoint_head.final_layer.weight"],
        state["keypoint_head.final_layer.bias"],
    )


def _materialize_corrected_model(
    converted_directory: Path, output_directory: Path, corrected_config: dict[str, Any]
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"Corrected model directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    link_methods: dict[str, str] = {}
    for name in ("model.safetensors", "preprocessor_config.json"):
        source = converted_directory / name
        target = output_directory / name
        try:
            os.link(source, target)
            link_methods[name] = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            link_methods[name] = "copy"
    with (output_directory / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(corrected_config, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return link_methods


def verify(run: Path) -> dict[str, Any]:
    started = time.perf_counter()
    downloads = run / "downloads"
    converted_directory = downloads / "converted"
    original_path = downloads / "original/vitpose-b.pth"
    converted_path = converted_directory / "model.safetensors"
    config_path = converted_directory / "config.json"
    for required in (original_path, converted_path, config_path, converted_directory / "preprocessor_config.json"):
        if not required.is_file():
            raise FileNotFoundError(required)

    # Hash before deserializing the pickle-format original checkpoint.
    original_hash = sha256_file(original_path)
    converted_hash = sha256_file(converted_path)
    if original_hash != ORIGINAL_CHECKPOINT_SHA256:
        raise RuntimeError("Original-mirror checkpoint SHA256 mismatch")
    if converted_hash != CONVERTED_CHECKPOINT_SHA256:
        raise RuntimeError("Converted checkpoint SHA256 mismatch")

    original_payload = torch.load(original_path, map_location="cpu", weights_only=False)
    original_state = original_payload["state_dict"]
    expected_converted, ignored = convert_original_state_dict(original_state)
    converted_state = load_file(converted_path, device="cpu")
    tensor_parity = compare_state_dicts_exact(expected_converted, converted_state)
    tensor_parity["ignored_original_keys"] = ignored
    if not tensor_parity["passed"] or ignored:
        raise RuntimeError(f"Converted tensor parity failed: {tensor_parity}")
    del converted_state, expected_converted, original_payload

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    original_effective_epsilon = raw_config.get("backbone_config", {}).get(
        "layer_norm_eps", "OMITTED_DEFAULTS_TO_1e-12"
    )
    corrected_config = corrected_config_payload(raw_config)
    corrected_directory = run / "corrected_model"
    link_methods = _materialize_corrected_model(
        converted_directory, corrected_directory, corrected_config
    )
    model = load_locked_vitpose(corrected_directory).eval().cpu()
    layer_norm_lock = validate_layer_norm_lock(model)

    # Deterministic, analytic, non-dataset fixture: no Okutama pixels or labels.
    torch.set_num_threads(1)
    fixture = torch.linspace(-2.0, 2.0, 3 * 256 * 192, dtype=torch.float32).reshape(
        1, 3, 256, 192
    )
    with torch.inference_mode():
        reference = _official_reference_heatmaps(fixture, original_state)
        corrected = model(fixture).heatmaps
    difference = (reference - corrected).abs()
    forward_parity = {
        "fixture": "linspace(-2,2,3*256*192), shape [1,3,256,192]",
        "output_shape": list(corrected.shape),
        "finite": bool(torch.isfinite(corrected).all()),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean((reference - corrected) ** 2))),
        "max_abs_tolerance_fixed_before_run": MAX_ABS_HEATMAP_TOLERANCE,
        "mean_abs_tolerance_fixed_before_run": MEAN_ABS_HEATMAP_TOLERANCE,
    }
    forward_parity["passed"] = bool(
        forward_parity["finite"]
        and forward_parity["output_shape"] == [1, 17, 64, 48]
        and forward_parity["max_abs_difference"] <= MAX_ABS_HEATMAP_TOLERANCE
        and forward_parity["mean_abs_difference"] <= MEAN_ABS_HEATMAP_TOLERANCE
    )

    # Quantify the defect we corrected by reproducing the omitted-field default
    # on the same loaded weights.  Restore the lock before returning.
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = 1e-12
    with torch.inference_mode():
        old_default = model(fixture).heatmaps
    old_difference = (corrected - old_default).abs()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = OFFICIAL_LAYER_NORM_EPS
    default_mismatch_effect = {
        "old_default_epsilon": 1e-12,
        "corrected_epsilon": OFFICIAL_LAYER_NORM_EPS,
        "max_abs_heatmap_change": float(old_difference.max()),
        "mean_abs_heatmap_change": float(old_difference.mean()),
        "outputs_bit_exact": bool(torch.equal(corrected, old_default)),
    }

    computational_pass = bool(
        tensor_parity["passed"] and layer_norm_lock["passed"] and forward_parity["passed"]
    )
    result = {
        "status": (
            "PASS_COMPUTATIONAL_MISMATCH_CORRECTED_OFFICIAL_BYTE_IDENTITY_UNRESOLVED"
            if computational_pass
            else "FAIL_COMPUTATIONAL_PARITY"
        ),
        "scope": "checkpoint/config audit only; no Okutama pose output and no task training",
        "checkpoint_hashes": {
            "source_mirror_vitpose_b_pth": original_hash,
            "converted_model_safetensors": converted_hash,
            "corrected_model_safetensors": sha256_file(
                corrected_directory / "model.safetensors"
            ),
            "corrected_config_json": sha256_file(corrected_directory / "config.json"),
        },
        "pinned_revisions": {
            "official_repository": REFERENCE_REVISION,
            "converter": CONVERTER_REVISION,
            "source_mirror": SOURCE_MIRROR_REVISION,
            "converted_model": CONVERTED_REVISION,
        },
        "configuration": {
            "raw_converted_effective_layer_norm_epsilon": original_effective_epsilon,
            "corrected_layer_norm_epsilon": OFFICIAL_LAYER_NORM_EPS,
            "all_runtime_layer_norms_locked": layer_norm_lock,
        },
        "tensor_parity": tensor_parity,
        "full_heatmap_reference_parity": forward_parity,
        "old_default_mismatch_effect": default_mismatch_effect,
        "corrected_model": {
            "path": str(corrected_directory),
            "materialization": link_methods,
            "local_files_only_required": True,
        },
        "remaining_provenance_limit": (
            "The publisher's OneDrive bytes/checksum remain inaccessible, so source-mirror "
            "identity to publisher bytes is not claimed."
        ),
        "pilot_pose_images_processed": 0,
        "task_models_fitted": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not computational_pass:
        raise RuntimeError(json.dumps(result, indent=2))
    write_new_json(run / "vitpose_computational_parity_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
