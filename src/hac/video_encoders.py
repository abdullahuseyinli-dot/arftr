"""Pinned, encoder-only V-JEPA 2.1 inference without upstream auto-downloads."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SOURCE_REPOSITORY = "https://github.com/facebookresearch/vjepa2.git"
SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
CHECKPOINT_BYTES = 1_664_223_428
# Observed acquisition digest, not a publisher-signed digest.
CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
ENCODER_MODULE = "app.vjepa_2_1.models.vision_transformer"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PADDING_RGB = (124, 116, 104)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Upstream Git verification failed: {' '.join(arguments)}")
    return result.stdout


def validate_source_checkout(source_root: Path) -> dict[str, Any]:
    root = source_root.resolve(strict=True)
    actual_root = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if actual_root != root:
        raise RuntimeError("Source root is not the exact upstream checkout root")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError("The upstream source commit is not the pinned commit")
    # Upstream has case-colliding config names on Windows. Verify every imported
    # Python source against Git below, and disclose the unrelated checkout changes.
    status = _git(root, "status", "--porcelain", "--untracked-files=no").decode().splitlines()
    return {
        "repository": SOURCE_REPOSITORY,
        "root": str(root),
        "commit": commit,
        "tracked_checkout_status": status,
        "integrity_policy": "all_imported_app_src_modules_match_pinned_git_blobs",
        "line_ending_policy": "CRLF_to_LF_only",
    }


def verify_python_source(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError("An upstream Python module originated outside its source root")
    relative = resolved.relative_to(root.resolve()).as_posix()
    if resolved.suffix != ".py":
        raise RuntimeError("An upstream module is not a Python source file")
    expected = _git(root, "cat-file", "blob", f"{SOURCE_COMMIT}:{relative}")
    actual = resolved.read_bytes()
    if actual.replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
        raise RuntimeError(f"Imported upstream source differs from pinned Git blob: {relative}")
    return {
        "path": relative,
        "sha256": hashlib.sha256(actual).hexdigest(),
        "git_blob_sha256": hashlib.sha256(expected).hexdigest(),
        "line_endings_normalized": actual != expected,
    }


def _upstream_modules() -> list[tuple[str, ModuleType]]:
    return [
        (name, module)
        for name, module in tuple(sys.modules.items())
        if module is not None and (name in {"app", "src"} or name.startswith(("app.", "src.")))
    ]


def _verify_module_origins(root: Path) -> list[dict[str, Any]]:
    receipts = []
    for name, module in _upstream_modules():
        origin = getattr(module, "__file__", None)
        if origin:
            receipt = verify_python_source(root, Path(origin))
            receipt["module"] = name
            receipts.append(receipt)
        else:
            namespace_paths = list(getattr(module, "__path__", ()))
            # ``src`` is a PEP 420 namespace in upstream.  After restoring the
            # caller's sys.path its dynamic top-level search path can point at the
            # caller repository, even though every imported executable child came
            # from the pinned checkout and is verified above.  A namespace has no
            # executable bytes of its own; foreign child modules still fail.
            if name == "src" and namespace_paths:
                continue
            if not namespace_paths or any(
                not Path(path).resolve().is_relative_to(root) for path in namespace_paths
            ):
                raise RuntimeError(f"Upstream namespace has a foreign origin: {name}")
    return sorted(receipts, key=lambda item: item["module"])


def import_pinned_encoder(source_root: Path) -> tuple[ModuleType, dict[str, Any]]:
    root = source_root.resolve(strict=True)
    receipt = validate_source_checkout(root)
    _verify_module_origins(root)  # Reject already-imported conflicting app/src packages.
    module_path = root / "app/vjepa_2_1/models/vision_transformer.py"
    verify_python_source(root, module_path)
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        module = importlib.import_module(ENCODER_MODULE)
    finally:
        sys.path[:] = previous_path
    if Path(module.__file__).resolve() != module_path.resolve():
        raise RuntimeError("The encoder module does not have the exact expected origin")
    receipt["imported_python_sources"] = _verify_module_origins(root)
    return module, receipt


def clean_ema_encoder_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("ema_encoder")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint is missing a nonempty ema_encoder state")
    cleaned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("Encoder state contains a non-tensor or invalid key")
        key = name.replace("module.", "").replace("backbone.", "")
        if not key or key in cleaned:
            raise RuntimeError("Checkpoint prefix removal produced a key collision")
        cleaned[key] = value
    return cleaned


def validate_checkpoint(path: Path, expected_sha256: str = CHECKPOINT_SHA256) -> dict[str, Any]:
    if expected_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint digest is not the pinned acquisition digest")
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError("Checkpoint byte count does not match the published file")
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise RuntimeError("Checkpoint SHA-256 does not match its acquisition receipt")
    return {
        "path": str(resolved),
        "url": CHECKPOINT_URL,
        "bytes": CHECKPOINT_BYTES,
        "sha256": digest,
        "hash_authority": "observed_acquisition_not_publisher_signed",
        "state_key": "ema_encoder",
    }


def load_vjepa21_encoder(
    source_root: Path,
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
    device: str | torch.device = "cuda",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_receipt = validate_checkpoint(checkpoint_path, checkpoint_sha256)
    module, source_receipt = import_pinned_encoder(source_root)
    configuration = {
        "patch_size": 16,
        "img_size": (384, 384),
        "num_frames": 64,
        "tubelet_size": 2,
        "use_sdpa": True,
        "use_silu": False,
        "wide_silu": True,
        "uniform_power": False,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
    }
    encoder = module.vit_base(**configuration)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    encoder.load_state_dict(clean_ema_encoder_state(state), strict=True)
    del state
    # Detect checkpoint replacement while the mmap-backed tensors were copied.
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed while loading the encoder")
    source_receipt["imported_python_sources"] = _verify_module_origins(source_root.resolve())
    encoder.requires_grad_(False).eval().to(device)
    return encoder, {
        "model": "vjepa2_1_vit_base_384",
        "source": source_receipt,
        "checkpoint": checkpoint_receipt,
        "encoder_configuration": configuration,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "predictor_constructed": False,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow", "timm", "einops")
        },
    }


def preprocess_rgb_clip(
    frames: Sequence[Image.Image | np.ndarray], *, size: int = 384
) -> torch.Tensor:
    """Letterbox RGB frames without body clipping; return normalized C,T,H,W.

    This is a declared domain transform, not upstream resize-and-center-crop.
    Padding is RGB (124,116,104); resize uses PIL bilinear with rounded dimensions.
    """
    if not frames or len(frames) % 2 or size < 1:
        raise ValueError("A video clip requires a positive even frame count and positive size")
    tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Array frames must be uint8 RGB with shape H,W,3")
            image = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                raise ValueError("PIL frames must already be explicitly converted to RGB")
            image = frame
        else:
            raise TypeError("Frames must be RGB PIL images or uint8 arrays")
        if min(image.size) < 1:
            raise ValueError("An RGB frame has empty spatial dimensions")
        ratio = size / max(image.size)
        width = min(size, max(1, round(image.width * ratio)))
        height = min(size, max(1, round(image.height * ratio)))
        canvas = Image.new("RGB", (size, size), PADDING_RGB)
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
        tensors.append(torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1))
    pixels = torch.stack(tensors, dim=1).to(torch.float32).div_(255)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return pixels.sub_(mean).div_(std).contiguous()


def pool_spatiotemporal_tokens(
    tokens: torch.Tensor,
    *,
    frames: int = 16,
    image_size: int = 384,
    spatial_grid: int = 3,
) -> torch.Tensor:
    """Pool each tubelet's spatial grid independently; output B,T/2,G*G,768."""
    if frames < 2 or frames % 2 or image_size < 16 or image_size % 16:
        raise ValueError("Invalid frame count or patch-aligned image size")
    patch_grid = image_size // 16
    if spatial_grid < 1 or patch_grid % spatial_grid:
        raise ValueError("Spatial pooling grid must divide the patch grid exactly")
    temporal = frames // 2
    expected = temporal * patch_grid * patch_grid
    if tokens.ndim != 3 or tokens.shape[1:] != (expected, 768):
        raise ValueError(f"Expected dense tokens B,{expected},768; got {tuple(tokens.shape)}")
    if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError("Dense encoder tokens must be finite floating-point values")
    batch = tokens.shape[0]
    maps = tokens.reshape(batch * temporal, patch_grid, patch_grid, 768).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(maps.float(), kernel_size=patch_grid // spatial_grid)
    return pooled.permute(0, 2, 3, 1).reshape(batch, temporal, spatial_grid**2, 768)
