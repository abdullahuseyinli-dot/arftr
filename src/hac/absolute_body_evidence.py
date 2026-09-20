"""Label-blind absolute RGB observations for the source-resolved posture witness.

The central contract is crop-before-encoding.  A ``parts`` observation is three
independent frozen-encoder calls (whole, upper and lower), never a pooling of
tokens from one already-resized whole-person image.  Source, geometry and
missingness are explicit so an unavailable observation can retain ARFTR.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from hac.video_encoders import IMAGENET_MEAN, IMAGENET_STD, PADDING_RGB

SOURCE_SIZE_720 = (1280, 720)
DINO_INPUT_SIZE = 378
DINO_PATCH_SIZE = 14
DINO_PATCH_GRID = 27
FEATURE_DIM = 768
SOURCE_IDS = ("J", "R", "N")
REGION_IDS = ("whole", "upper", "lower")
PART_VERTICAL_INTERVALS = {"upper": (-0.05, 0.60), "lower": (0.40, 1.05)}
WHOLE_CONTEXT_SCALE = 1.25
MULTISCALE_WHOLE = (1.0, 1.25, 1.5)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays are not auditable source observations")
    digest = hashlib.sha256(canonical_digest([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _valid_box(box: Sequence[float], source_size: tuple[int, int]) -> bool:
    if len(box) != 4:
        return False
    value = np.asarray(box, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value[2:] <= value[:2]):
        return False
    width, height = source_size
    return bool(min(value[2], width) > max(value[0], 0) and min(value[3], height) > max(value[1], 0))


@dataclass(frozen=True)
class AbsoluteCropGeometry:
    source_id: str
    region_id: str
    actor_box: tuple[float, float, float, float]
    continuous_box: tuple[float, float, float, float]
    integer_box: tuple[int, int, int, int]
    source_size: tuple[int, int]
    whole_scale: float | None

    @property
    def raw_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.integer_box
        return right - left, bottom - top

    def receipt(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            raw_size=list(self.raw_size),
            rounding="floor_left_top__ceil_right_bottom",
            coordinates="continuous_pixel_edges",
            padding_rgb=list(PADDING_RGB),
            crop_before_model_resize=True,
        )
        return result


def make_crop_geometry(
    actor_box: Sequence[float],
    *,
    source_size: tuple[int, int],
    source_id: str,
    region_id: str,
    whole_scale: float = WHOLE_CONTEXT_SCALE,
) -> AbsoluteCropGeometry:
    """Create one source-coordinate crop with fixed outward rounding.

    Whole crops scale both axes about the actor center.  Upper/lower crops use a
    1.25-wide horizontal field and fixed actor-relative vertical intervals.
    """
    box = tuple(float(x) for x in actor_box)
    if source_id not in SOURCE_IDS or region_id not in REGION_IDS:
        raise ValueError("Unknown source or region identifier")
    if not _valid_box(box, source_size) or not math.isfinite(whole_scale) or whole_scale <= 0:
        raise ValueError("Malformed source actor geometry")
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    if region_id == "whole":
        crop_width, crop_height = whole_scale * width, whole_scale * height
        continuous = (
            center_x - crop_width / 2,
            center_y - crop_height / 2,
            center_x + crop_width / 2,
            center_y + crop_height / 2,
        )
        scale: float | None = float(whole_scale)
    else:
        low, high = PART_VERTICAL_INTERVALS[region_id]
        horizontal = WHOLE_CONTEXT_SCALE * width
        continuous = (
            center_x - horizontal / 2,
            y1 + low * height,
            center_x + horizontal / 2,
            y1 + high * height,
        )
        scale = None
    integer = (
        math.floor(continuous[0]),
        math.floor(continuous[1]),
        math.ceil(continuous[2]),
        math.ceil(continuous[3]),
    )
    if integer[2] <= integer[0] or integer[3] <= integer[1]:
        raise ValueError("Rounded source crop is empty")
    return AbsoluteCropGeometry(source_id, region_id, box, continuous, integer, source_size, scale)


def crop_with_padding(rgb: np.ndarray, geometry: AbsoluteCropGeometry) -> np.ndarray:
    image = np.asarray(rgb)
    source_width, source_height = geometry.source_size
    if image.dtype != np.uint8 or image.shape != (source_height, source_width, 3):
        raise ValueError("RGB image and declared source dimensions differ")
    left, top, right, bottom = geometry.integer_box
    raw_width, raw_height = geometry.raw_size
    output = np.empty((raw_height, raw_width, 3), dtype=np.uint8)
    output[:] = np.asarray(PADDING_RGB, dtype=np.uint8)
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(source_width, right), min(source_height, bottom)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Crop does not intersect the source image")
    output[y1 - top : y2 - top, x1 - left : x2 - left] = image[y1:y2, x1:x2]
    return output


def exact_native_downsample720(native_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(native_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Native source must be uint8 RGB")
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(SOURCE_SIZE_720, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ).copy()


def letterbox_rgb(rgb: np.ndarray, *, output_size: int = DINO_INPUT_SIZE) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Letterbox requires nonempty uint8 RGB")
    if output_size != DINO_INPUT_SIZE or output_size % DINO_PATCH_SIZE:
        raise ValueError("The posture witness is locked to a complete27x27 patch grid")
    height, width = image.shape[:2]
    ratio = output_size / max(width, height)
    resized_width = min(output_size, max(1, round(width * ratio)))
    resized_height = min(output_size, max(1, round(height * ratio)))
    left = (output_size - resized_width) // 2
    top = (output_size - resized_height) // 2
    canvas = Image.new("RGB", (output_size, output_size), tuple(PADDING_RGB))
    resized = Image.fromarray(image, mode="RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.uint8).copy()


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(letterbox_rgb(rgb)).permute(2, 0, 1).float().div_(255)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return pixels.sub_(mean).div_(std).contiguous()


class FrozenAbsoluteDinoCLS(nn.Module):
    """DINO CLS adapter that requires complete patch coverage at378x378."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.requires_grad_(False).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected = (3, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(f"Absolute DINO input must be B,{expected}")
        if not pixel_values.is_floating_point() or not torch.isfinite(pixel_values).all():
            raise ValueError("DINO inputs must be finite floating point")
        if self.training or self.backbone.training:
            raise RuntimeError("Absolute DINO extraction requires eval mode")
        hidden = self.backbone(pixel_values=pixel_values, return_dict=True).last_hidden_state
        expected_shape = (len(pixel_values), 1 + DINO_PATCH_GRID**2, FEATURE_DIM)
        if tuple(hidden.shape) != expected_shape or not torch.isfinite(hidden).all():
            raise RuntimeError("Pinned DINO does not provide a complete27x27 hidden state")
        return F.normalize(hidden[:, 0].float(), dim=1)


def compose_source_images(supplied720: np.ndarray, native: np.ndarray) -> dict[str, np.ndarray]:
    supplied = np.asarray(supplied720)
    original = np.asarray(native)
    if supplied.dtype != np.uint8 or supplied.shape != (720, 1280, 3):
        raise ValueError("J source must be exact1280x720 uint8 RGB")
    if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("N source must be nonempty uint8 RGB")
    return {"J": supplied, "R": exact_native_downsample720(original), "N": original}


def source_actor_boxes(box720: Sequence[float], native_size: tuple[int, int]) -> dict[str, tuple[float, ...]]:
    box = tuple(float(value) for value in box720)
    if not _valid_box(box, SOURCE_SIZE_720):
        raise ValueError("Supplied actor box is invalid")
    sx, sy = native_size[0] / SOURCE_SIZE_720[0], native_size[1] / SOURCE_SIZE_720[1]
    native = tuple(value * scale for value, scale in zip(box, (sx, sy, sx, sy), strict=True))
    if not _valid_box(native, native_size):
        raise ValueError("Scaled native actor box is invalid")
    return {"J": box, "R": box, "N": native}


def prepare_source_crops(
    source_image: np.ndarray,
    actor_box: Sequence[float],
    *,
    source_id: str,
    include_multiscale: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Prepare one source independently so source missingness stays explicit."""
    if source_id not in SOURCE_IDS or (include_multiscale and source_id != "R"):
        raise ValueError("Only R may carry the locked multiscale whole-body arm")
    image = np.asarray(source_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Source image must be nonempty uint8 RGB")
    size = (image.shape[1], image.shape[0])
    crops: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for region_id in REGION_IDS:
        geometry = make_crop_geometry(
            actor_box,
            source_size=size,
            source_id=source_id,
            region_id=region_id,
        )
        key = f"{source_id}_{region_id}"
        crops[key] = crop_with_padding(image, geometry)
        receipts[key] = {
            **geometry.receipt(),
            "raw_rgb_sha256": array_digest(crops[key]),
            "letterbox_sha256": array_digest(letterbox_rgb(crops[key])),
        }
    if include_multiscale:
        for scale in MULTISCALE_WHOLE:
            key = f"R_whole_{str(scale).replace('.', 'p')}"
            if scale == WHOLE_CONTEXT_SCALE:
                crops[key] = crops["R_whole"]
                receipts[key] = {**receipts["R_whole"], "reuses": "R_whole"}
                continue
            geometry = make_crop_geometry(
                actor_box,
                source_size=size,
                source_id="R",
                region_id="whole",
                whole_scale=scale,
            )
            crops[key] = crop_with_padding(image, geometry)
            receipts[key] = {
                **geometry.receipt(),
                "raw_rgb_sha256": array_digest(crops[key]),
                "letterbox_sha256": array_digest(letterbox_rgb(crops[key])),
            }
    return crops, receipts


def prepare_crops(
    source_images: Mapping[str, np.ndarray],
    boxes: Mapping[str, Sequence[float]],
    *,
    include_multiscale_r: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Return named raw crops and receipts for all source×region arms."""
    if set(source_images) != set(SOURCE_IDS) or set(boxes) != set(SOURCE_IDS):
        raise ValueError("All three pinned sources are required")
    crops: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for source_id in SOURCE_IDS:
        image = np.asarray(source_images[source_id])
        size = (image.shape[1], image.shape[0])
        for region_id in REGION_IDS:
            geometry = make_crop_geometry(
                boxes[source_id], source_size=size, source_id=source_id, region_id=region_id
            )
            key = f"{source_id}_{region_id}"
            crops[key] = crop_with_padding(image, geometry)
            receipts[key] = {
                **geometry.receipt(),
                "raw_rgb_sha256": array_digest(crops[key]),
                "letterbox_sha256": array_digest(letterbox_rgb(crops[key])),
            }
    if include_multiscale_r:
        image = np.asarray(source_images["R"])
        for scale in MULTISCALE_WHOLE:
            key = f"R_whole_{str(scale).replace('.', 'p')}"
            if scale == WHOLE_CONTEXT_SCALE:
                crops[key] = crops["R_whole"]
                receipts[key] = {**receipts["R_whole"], "reuses": "R_whole"}
                continue
            geometry = make_crop_geometry(
                boxes["R"], source_size=(image.shape[1], image.shape[0]), source_id="R",
                region_id="whole", whole_scale=scale
            )
            crops[key] = crop_with_padding(image, geometry)
            receipts[key] = {
                **geometry.receipt(),
                "raw_rgb_sha256": array_digest(crops[key]),
                "letterbox_sha256": array_digest(letterbox_rgb(crops[key])),
            }
    return crops, receipts
