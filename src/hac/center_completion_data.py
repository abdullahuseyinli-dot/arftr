"""Fail-closed RGB preparation for center-evidence completion.

This module deliberately knows nothing about action labels, ARFTR outputs, pose,
support categories, or historical error membership.  It consumes only pinned JPEG
members and supplied actor boxes.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hac.okutama_native_video import CropGeometry, crop_geometry, valid_box
from hac.video_encoders import IMAGENET_MEAN, IMAGENET_STD, PADDING_RGB

SOURCE_SIZE = (1280, 720)
OUTPUT_SIZE = 384
PATCH_SIZE = 14
PATCH_GRID = 27
CONTEXT_FRACTION = 0.125
MASK_BANDS = {
    "upper": (0.0, 0.475),
    "lower": (0.525, 1.0),
}
FILL_BANDS = {
    "upper": (0.0, 0.525),
    "lower": (0.475, 1.0),
}


@dataclass(frozen=True)
class AllowlistedMember:
    image_member: str
    crc32: int
    size_bytes: int


@dataclass(frozen=True)
class RawActorCrop:
    rgb: np.ndarray
    source_valid: np.ndarray
    actor_box_local: tuple[float, float, float, float]
    geometry: CropGeometry


@dataclass(frozen=True)
class PreparedCenter:
    unmasked_pixels: torch.Tensor
    masked_pixels: torch.Tensor
    target_patch_masks: np.ndarray
    visible_patch_masks: np.ndarray
    valid_patch_mask: np.ndarray
    raw_crop: RawActorCrop


@dataclass(frozen=True)
class PreparedNeighbor:
    pixels: torch.Tensor
    valid_patch_mask: np.ndarray
    raw_crop: RawActorCrop


def _as_box(row: Mapping[str, str]) -> tuple[float, float, float, float]:
    box = tuple(
        float(row[key])
        for key in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
    )
    if len(box) != 4 or not valid_box(box):
        raise RuntimeError("Supplied center-completion box is invalid")
    return box


def decode_allowlisted_jpeg(
    archive: zipfile.ZipFile,
    row: Mapping[str, str],
    allowlist: Mapping[str, AllowlistedMember],
) -> tuple[Image.Image, dict[str, object]]:
    """Decode one exact pinned member after metadata and CRC validation."""

    if row.get("valid_frame", "").lower() not in {"1", "true"}:
        raise RuntimeError("Center-completion preflight refuses an invalid source row")
    if (int(row["image_width"]), int(row["image_height"])) != SOURCE_SIZE:
        raise RuntimeError("Frame manifest source dimensions changed")
    member = row["image_member"]
    expected = allowlist.get(member)
    if expected is None:
        raise RuntimeError("Source JPEG is outside the pinned image allowlist")
    info = archive.getinfo(member)
    if info.is_dir() or info.file_size != expected.size_bytes or info.CRC != expected.crc32:
        raise RuntimeError("Source JPEG ZIP metadata differs from the pinned allowlist")
    raw = archive.read(info)
    if len(raw) != expected.size_bytes or zlib.crc32(raw) != expected.crc32:
        raise RuntimeError("Source JPEG payload byte count or CRC32 changed")
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        if source.format != "JPEG" or source.size != SOURCE_SIZE:
            raise RuntimeError("Source member is not the pinned 1280x720 JPEG condition")
        rgb = source.convert("RGB").copy()
    return rgb, {
        "image_member": member,
        "crc32": expected.crc32,
        "size_bytes": expected.size_bytes,
        "compressed_payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def raw_actor_crop(image: Image.Image, box: tuple[float, float, float, float]) -> RawActorCrop:
    """Expand the supplied box to 1.25x, preserving padding and pixel-edge geometry."""

    if image.mode != "RGB" or image.size != SOURCE_SIZE or not valid_box(box):
        raise ValueError("Actor crop requires a valid RGB source and supplied 720p box")
    geometry = crop_geometry(
        box,
        output_size=OUTPUT_SIZE,
        context_fraction=CONTEXT_FRACTION,
    )
    left, top, right, bottom = geometry.crop_box
    height, width = bottom - top, right - left
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = np.asarray(PADDING_RGB, dtype=np.uint8)
    source_valid = np.zeros((height, width), dtype=np.uint8)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        raise RuntimeError("Expanded actor crop does not intersect its source image")
    tile = np.asarray(image.crop(source_box), dtype=np.uint8)
    x0, y0 = source_box[0] - left, source_box[1] - top
    rgb[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    source_valid[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = 255
    actor_local = (box[0] - left, box[1] - top, box[2] - left, box[3] - top)
    return RawActorCrop(rgb, source_valid, actor_local, geometry)


def _raw_actor_band_mask(crop: RawActorCrop, bounds: tuple[float, float]) -> np.ndarray:
    """Rasterize one actor-relative rectangle using raw-crop pixel centers."""

    low, high = bounds
    x1, y1, x2, y2 = crop.actor_box_local
    height, width = crop.rgb.shape[:2]
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    y_low = y1 + low * (y2 - y1)
    y_high = y1 + high * (y2 - y1)
    mask = (
        (ys[:, None] >= y_low)
        & (ys[:, None] < y_high)
        & (xs[None, :] >= x1)
        & (xs[None, :] < x2)
    )
    mask &= crop.source_valid.astype(bool)
    if not mask.any():
        raise RuntimeError("Actor target band contains no valid raw source pixels")
    return mask


def raw_target_mask(crop: RawActorCrop, mask_id: str) -> np.ndarray:
    """Rasterize the scored subset of a fixed actor-relative target band."""

    try:
        bounds = MASK_BANDS[mask_id]
    except KeyError as exc:
        raise ValueError(f"Unknown center target mask: {mask_id}") from exc
    return _raw_actor_band_mask(crop, bounds)


def raw_fill_mask(crop: RawActorCrop, mask_id: str) -> np.ndarray:
    """Rasterize hidden pixels, including the unscored 5% middle safety band."""

    try:
        bounds = FILL_BANDS[mask_id]
    except KeyError as exc:
        raise ValueError(f"Unknown center fill mask: {mask_id}") from exc
    return _raw_actor_band_mask(crop, bounds)


def apply_raw_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Masking requires uint8 H,W,3 RGB")
    if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
        raise ValueError("Raw target mask shape or dtype changed")
    result = rgb.copy()
    result[mask] = np.asarray(PADDING_RGB, dtype=np.uint8)
    return result


def perturb_hidden_raw_pixels(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deterministically alter every hidden source pixel for dependency auditing."""

    result = rgb.copy()
    result[mask] = np.bitwise_xor(result[mask], np.uint8(255))
    if np.array_equal(result[mask], rgb[mask]):
        raise RuntimeError("Hidden-pixel perturbation unexpectedly changed no bytes")
    return result


def _letterbox_sizes(width: int, height: int) -> tuple[int, int, int, int]:
    ratio = OUTPUT_SIZE / max(width, height)
    resized_width = min(OUTPUT_SIZE, max(1, round(width * ratio)))
    resized_height = min(OUTPUT_SIZE, max(1, round(height * ratio)))
    pad_x = (OUTPUT_SIZE - resized_width) // 2
    pad_y = (OUTPUT_SIZE - resized_height) // 2
    return resized_width, resized_height, pad_x, pad_y


def letterbox_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return the exact locked uint8 PIL-bilinear 384x384 letterbox."""

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Preprocessing requires uint8 H,W,3 RGB")
    height, width = rgb.shape[:2]
    resized_width, resized_height, pad_x, pad_y = _letterbox_sizes(width, height)
    canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), PADDING_RGB)
    resized = Image.fromarray(rgb, mode="RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas.paste(resized, (pad_x, pad_y))
    return np.array(canvas, copy=True)


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Replay the locked letterbox and ImageNet normalization."""

    pixels = torch.from_numpy(letterbox_rgb(rgb)).permute(2, 0, 1).float().div_(255)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return pixels.sub_(mean).div_(std).contiguous()


def _preprocess_binary_full_support(mask: np.ndarray) -> np.ndarray:
    """Mark outputs whose full PIL-bilinear source footprint is inside ``mask``."""

    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Support transform requires a boolean H,W mask")
    height, width = mask.shape
    resized_width, resized_height, pad_x, pad_y = _letterbox_sizes(width, height)
    source = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    values = np.asarray(resized, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = values
    return canvas == 255


def full_support_patch_mask(raw_mask: np.ndarray) -> np.ndarray:
    pixels = _preprocess_binary_full_support(raw_mask)
    covered = pixels[: PATCH_GRID * PATCH_SIZE, : PATCH_GRID * PATCH_SIZE]
    return covered.reshape(PATCH_GRID, PATCH_SIZE, PATCH_GRID, PATCH_SIZE).all(axis=(1, 3))


def prepare_center(image: Image.Image, row: Mapping[str, str]) -> PreparedCenter:
    crop = raw_actor_crop(image, _as_box(row))
    unmasked = preprocess_rgb(crop.rgb)
    masked_pixels = []
    target_masks = []
    visible_masks = []
    for mask_id in MASK_BANDS:
        fill_mask = raw_fill_mask(crop, mask_id)
        score_mask = raw_target_mask(crop, mask_id)
        masked_pixels.append(preprocess_rgb(apply_raw_mask(crop.rgb, fill_mask)))
        target_masks.append(full_support_patch_mask(score_mask))
        visible_masks.append(
            full_support_patch_mask(crop.source_valid.astype(bool) & ~fill_mask)
        )
    targets = np.stack(target_masks)
    visible = np.stack(visible_masks)
    if not targets.reshape(2, -1).any(axis=1).all() or np.logical_and(*targets).any():
        raise RuntimeError("Center target patch masks are empty or overlapping")
    valid = full_support_patch_mask(crop.source_valid.astype(bool))
    if not (targets <= valid[None]).all():
        raise RuntimeError("A target patch depends on padded rather than source pixels")
    if np.logical_and(targets, visible).any():
        raise RuntimeError("A target patch leaked into the visible-anchor set")
    return PreparedCenter(
        unmasked_pixels=unmasked,
        masked_pixels=torch.stack(masked_pixels),
        target_patch_masks=targets,
        visible_patch_masks=visible,
        valid_patch_mask=valid,
        raw_crop=crop,
    )


def prepare_neighbor(image: Image.Image, row: Mapping[str, str]) -> PreparedNeighbor:
    crop = raw_actor_crop(image, _as_box(row))
    return PreparedNeighbor(
        pixels=preprocess_rgb(crop.rgb),
        valid_patch_mask=full_support_patch_mask(crop.source_valid.astype(bool)),
        raw_crop=crop,
    )
