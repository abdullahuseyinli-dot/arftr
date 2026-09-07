"""Pinned local DINOv2 CLS extraction for the matched native-video control."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hac.video_encoders import sha256_file

DINO_MODEL_ID = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINO_FILES = {
    "config.json": {
        "size_bytes": 548,
        "sha256": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    },
    "model.safetensors": {
        "size_bytes": 346345912,
        "sha256": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
    },
    "preprocessor_config.json": {
        "size_bytes": 436,
        "sha256": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    },
}


def validate_dinov2_snapshot(model_root: Path) -> dict[str, Any]:
    root = model_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("DINO snapshot root must be a local directory")
    # AutoModel can discover adapters or alternative weight indexes by filename.
    # Do not permit those unbound executable inputs alongside the three pinned files.
    executable_suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth", ".py"}
    if any(
        path.is_file()
        and path.name not in DINO_FILES
        and path.suffix.lower() in executable_suffixes
        for path in root.iterdir()
    ):
        raise RuntimeError("DINO snapshot contains an unbound executable input")
    receipts = {}
    for name, expected in DINO_FILES.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"Pinned DINO snapshot file size changed: {name}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise RuntimeError(f"Pinned DINO snapshot file hash changed: {name}")
        receipts[name] = {"path": str(path.resolve()), **expected}
    return {
        "model_id": DINO_MODEL_ID,
        "revision": DINO_REVISION,
        "path": str(root),
        "files": receipts,
        "local_files_only": True,
        "remote_code": False,
    }


def validate_dinov2_lock_receipt(
    model_root: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise RuntimeError("Extraction lock lacks its DINO snapshot receipt")
    declared_root = receipt.get("path", receipt.get("root"))
    if not declared_root or Path(declared_root).resolve() != model_root.resolve():
        raise RuntimeError("DINO model root differs from the extraction lock")
    if receipt.get("revision") != DINO_REVISION:
        raise RuntimeError("DINO revision differs from the pinned extraction contract")
    files = receipt.get("files", {})
    if set(files) != set(DINO_FILES):
        raise RuntimeError("DINO lock must include all three pinned snapshot files")
    for name, expected in DINO_FILES.items():
        item = files[name]
        if any(item.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"DINO lock snapshot bytes differ from the pinned contract: {name}")
        if "path" in item and Path(item["path"]).resolve() != (model_root / name).resolve():
            raise RuntimeError("DINO snapshot file path differs from its locked root")
    return validate_dinov2_snapshot(model_root)


class FrozenDinoCLS(nn.Module):
    """Return final normalized CLS tokens without fitting or spatial pooling."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.requires_grad_(False).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (3, 384, 384):
            raise ValueError("Matched DINO inputs must be B,3,384,384")
        if not pixel_values.is_floating_point() or not torch.isfinite(pixel_values).all():
            raise ValueError("DINO input pixels must be finite normalized floating-point tensors")
        if self.training or self.backbone.training:
            raise RuntimeError("DINO extraction requires frozen evaluation mode")
        output = self.backbone(pixel_values=pixel_values, return_dict=True)
        dense = output.last_hidden_state
        if dense.ndim != 3 or dense.shape[0] != pixel_values.shape[0] or dense.shape[2] != 768:
            raise RuntimeError("DINO final hidden-state shape differs from the pinned model")
        if dense.shape[1] != 1 + 27 * 27:
            raise RuntimeError(
                "DINO 384px patch-token count differs from the pinned patch-14 model"
            )
        cls = dense[:, 0, :]
        if not torch.isfinite(cls).all():
            raise RuntimeError("DINO CLS tokens contain nonfinite values")
        return cls


def load_dinov2_encoder(
    model_root: Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[FrozenDinoCLS, dict[str, Any]]:
    snapshot = validate_dinov2_snapshot(model_root)
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(model_root.resolve()),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    if (backbone.config.model_type, backbone.config.hidden_size, backbone.config.patch_size) != (
        "dinov2",
        768,
        14,
    ):
        raise RuntimeError("Local DINO model configuration differs from the pinned backbone")
    # Close the initial hash-to-load window without modifying the local snapshot.
    if validate_dinov2_snapshot(model_root) != snapshot:
        raise RuntimeError("DINO snapshot changed during loading")
    encoder = FrozenDinoCLS(backbone).to(device)
    return encoder, {
        "model": "dinov2_base_native16_cls",
        "snapshot": snapshot,
        "representation": "last_hidden_state[:,0,:]",
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "input_size": [384, 384],
        "input_transform": "exact_LockedClipDataset_actual_pixels;no_additional_resize_or_crop",
        "normalization": "ImageNet_mean_std_shared_with_video_encoder",
        "source_processor_defaults_applied": False,
        "patch_size": 14,
        "patch_grid": [27, 27],
        "unpatchified_trailing_pixels_per_axis": 6,
        "position_encoding": "upstream_Dinov2_spatial_interpolation",
        "frozen": True,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
    }
