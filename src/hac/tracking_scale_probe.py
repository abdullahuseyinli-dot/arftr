"""Matched-context, different-detail inputs for a frozen tracking scale probe.

The ROI is fixed from the center-frame box alone. Native integer crops and a
crop sampled from the already-downscaled full-frame tensor have the same ROI
coordinates and model-resolution support grid. The second arm cannot restore
information lost by the full-frame resize. No tracker inference, future boxes,
new point identities, padding pixels, or activity labels are used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from hac.learned_persistent_tracking import PreparedVideo


@dataclass(frozen=True)
class FixedCrop:
    """Integer half-open native-image bounds; coordinates refer to pixel centers."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self):
        for name in ("left", "top", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Crop bounds must be integers")
            object.__setattr__(self, name, int(value))
        if self.left < 0 or self.top < 0 or self.width < 2 or self.height < 2:
            raise ValueError("Crop origins must be nonnegative and both dimensions at least two")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _shape(shape, name: str) -> tuple[int, int]:
    values = tuple(shape)
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 2 for value in values
    ):
        raise ValueError(f"{name} must be integer (height, width), both at least two")
    return int(values[0]), int(values[1])


def fixed_actor_crop(box, image_shape, model_shape, context: float = 4.0) -> FixedCrop:
    """Fit a center bbox, enlarge linearly, then shift/clamp inside the image.

    Shapes are (height, width); box is (xmin, ymin, xmax, ymax). A continuous
    aspect-matched rectangle enclosing the box is multiplied by ``context``.
    Dimensions are rounded outward on the reduced integer model-aspect lattice
    (4:3 for a 512x384 model), then capped to the largest fitting rectangle of
    that aspect. Context may shrink at image limits, but the actor box may not
    be clipped. An impossible box/aspect combination is rejected, not padded.
    This function accepts one center box, never a sequence of future boxes.
    """
    height, width = _shape(image_shape, "Image shape")
    model_height, model_width = _shape(model_shape, "Model shape")
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("One finite center box (xmin,ymin,xmax,ymax) is required")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Center box must have positive area and lie inside the native image")
    if isinstance(context, bool) or not np.isfinite(context) or context < 1:
        raise ValueError("Linear context must be finite and at least one")
    divisor = math.gcd(model_width, model_height)
    unit_width, unit_height = model_width // divisor, model_height // divisor
    max_units = min(width // unit_width, height // unit_height)
    fit_units = max((x1 - x0) / unit_width, (y1 - y0) / unit_height)
    if max_units < math.ceil(fit_units):
        raise ValueError("Actor box cannot fit the model aspect inside this image without padding")
    wanted_units = math.ceil(float(context) * fit_units)
    units = min(wanted_units, max_units)
    crop_width, crop_height = unit_width * units, unit_height * units
    left = min(max(math.floor((x0 + x1 - crop_width) / 2), 0), width - crop_width)
    top = min(max(math.floor((y0 + y1 - crop_height) / 2), 0), height - crop_height)
    crop = FixedCrop(left, top, crop_width, crop_height)
    # Fractional boxes narrower than two pixels can otherwise be cut by an
    # integer-origin rounding. Never silently change a supplied center box.
    if not (crop.left <= x0 and crop.top <= y0 and crop.right >= x1 and crop.bottom >= y1):
        raise ValueError("Integer crop cannot enclose the fractional center box at this context")
    return crop


def _translated_points(points, crop: FixedCrop, direction: float, valid=None) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("Point arrays must retain their identity axes and end in xy")
    if valid is None:
        if np.isinf(values).any():
            raise ValueError("Point coordinates must be finite or NaN for missing identities")
        result = values.copy()
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != values.shape[:-1]:
            raise ValueError("Visibility must match every point/time identity axis")
        if not np.isfinite(values[mask]).all():
            raise ValueError("Visible identities must have finite point coordinates")
        result = values.copy()
        result[~mask] = np.nan
    result += direction * np.asarray([crop.left, crop.top], dtype=np.float64)
    return result


def points_to_crop(points, crop: FixedCrop, valid=None) -> np.ndarray:
    """Subtract the integer ROI origin, preserving shape, order and missing IDs.

    Coordinates are not clipped or filtered: outside-ROI points remain outside.
    A supplied visibility mask marks missing values NaN without repacking IDs.
    Integer slicing has no half-pixel offset; this is not an AREA resize.
    """
    return _translated_points(points, crop, -1.0, valid)


def points_from_crop(points, crop: FixedCrop, valid=None) -> np.ndarray:
    """Add the integer ROI origin; inverse of points_to_crop on visible IDs."""
    return _translated_points(points, crop, 1.0, valid)


def prepare_matched_crop_pair(backend, full_prepared: PreparedVideo, native_frames, roi: FixedCrop):
    """Return (native_detail, low_detail) PreparedVideo objects, without inference.

    ``full_prepared`` must have been prepared from the same RGB uint8 frames;
    the caller must bind their source hashes. Dimensions/time count are checked
    here, but tensor provenance cannot be inferred from shape alone.

    High uses the backend's unmodified native-ROI preprocessing. Low samples
    ``full_prepared.tensor[0]`` directly at the corresponding native pixel
    centers with align_corners=True. Grid and output are model-sized, never a
    native-resolution upsample. Both returned objects carry the SAME ROI source
    dimensions, so identical local queries get identical predictor scaling and
    support-grid context. No query/model call is made by this function.
    """
    if not isinstance(full_prepared, PreparedVideo) or not isinstance(roi, FixedCrop):
        raise ValueError("Expected a PreparedVideo and an integer FixedCrop")
    frames = list(native_frames)
    if len(frames) < 2 or not isinstance(frames[0], np.ndarray) or frames[0].ndim != 3:
        raise ValueError("At least two native RGB frames are required")
    height, width, channels = frames[0].shape
    if channels != 3 or min(height, width) < 2:
        raise ValueError("Native frames must be RGB with both dimensions at least two")
    if any(
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.shape != (height, width, 3)
        for frame in frames
    ):
        raise ValueError("Native frames must be aligned RGB uint8 with no missing frames")
    if roi.right > width or roi.bottom > height:
        raise ValueError("Fixed crop leaves the source image; no padding is allowed")
    if (full_prepared.source_height, full_prepared.source_width) != (height, width):
        raise ValueError("Full-frame prepared source dimensions differ from native frames")
    model_height, model_width = _shape(backend.model_shape, "Model shape")
    torch = backend.torch
    tensor = full_prepared.tensor
    expected_shape = (1, len(frames), 3, model_height, model_width)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
        raise ValueError("Full-frame prepared tensor has a different time/model-resolution shape")
    if tensor.dtype not in (torch.float32, torch.float64) or not torch.isfinite(tensor).all():
        raise ValueError("Full-frame prepared tensor must contain finite float32/float64 pixels")
    crops = [frame[roi.top : roi.bottom, roi.left : roi.right] for frame in frames]
    high = backend.prepare_video(crops)
    if (
        (high.source_height, high.source_width) != (roi.height, roi.width)
        or tuple(high.tensor.shape) != expected_shape
        or high.tensor.device != tensor.device
        or high.tensor.dtype != tensor.dtype
    ):
        raise ValueError(
            "Backend crop preprocessing changed geometry, dtype, device or frame count"
        )
    with torch.inference_mode():
        # Same pixel-center lattice as direct F.interpolate(crop, align_corners=True).
        xs = torch.linspace(
            roi.left, roi.right - 1, model_width, dtype=tensor.dtype, device=tensor.device
        )
        ys = torch.linspace(
            roi.top, roi.bottom - 1, model_height, dtype=tensor.dtype, device=tensor.device
        )
        normalized_x = 2.0 * xs / (width - 1) - 1.0
        normalized_y = 2.0 * ys / (height - 1) - 1.0
        yy, xx = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None].expand(len(frames), -1, -1, -1)
        # All sample locations are inside the real image. zeros padding is never used.
        if torch.any(grid < -1.0) or torch.any(grid > 1.0):
            raise ValueError("ROI sampling would extrapolate outside the full-frame tensor")
        sampled = torch.nn.functional.grid_sample(
            tensor[0], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        low = PreparedVideo(sampled[None], roi.height, roi.width)
    return high, low
