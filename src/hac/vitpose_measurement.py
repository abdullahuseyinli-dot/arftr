"""Frozen ViTPose measurements for the body-witness pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hac.body_witness_data import MASK_IDS, BodyCropGeometry

COCO_LANDMARK_INDICES = np.asarray([5, 6, 11, 12, 13, 14, 15, 16], dtype=np.int64)
DESCRIPTOR_HEIGHT = 16
DESCRIPTOR_WIDTH = 12
DESCRIPTOR_BINS = DESCRIPTOR_HEIGHT * DESCRIPTOR_WIDTH + 1
HEATMAP_HEIGHT = 64
HEATMAP_WIDTH = 48
MODEL_HEIGHT = 256
MODEL_WIDTH = 192
NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True)
class PoseMeasurement:
    descriptor: np.ndarray
    peak_magnitude: np.ndarray
    peak_raw_xy: np.ndarray


def _validate_inputs(
    heatmaps: np.ndarray, source_to_model: np.ndarray, geometry: BodyCropGeometry
) -> tuple[np.ndarray, np.ndarray]:
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    affine = np.asarray(source_to_model, dtype=np.float64)
    if heatmaps.shape != (8, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise ValueError("Expected eight selected 64x48 ViTPose heatmaps")
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise ValueError("Pose source-to-model affine must be finite 3x3")
    if not np.isfinite(heatmaps).all() or geometry.raw_size[0] < 1 or geometry.raw_size[1] < 1:
        raise ValueError("Pose measurement inputs must be finite and nonempty")
    return heatmaps, affine


def _raw_cell_coordinates(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Each 64x48 heatmap cell represents a 4x4 model-input cell.  The center
    # convention is frozen before pilot extraction.
    model_x = (np.arange(HEATMAP_WIDTH, dtype=np.float64) + 0.5) * (
        MODEL_WIDTH / HEATMAP_WIDTH
    )
    model_y = (np.arange(HEATMAP_HEIGHT, dtype=np.float64) + 0.5) * (
        MODEL_HEIGHT / HEATMAP_HEIGHT
    )
    grid_x, grid_y = np.meshgrid(model_x, model_y)
    homogeneous = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=-1)
    inverse = np.linalg.inv(affine)
    raw = homogeneous @ inverse.T
    return raw[..., 0] / raw[..., 2], raw[..., 1] / raw[..., 2]


def _target_region(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    geometry: BodyCropGeometry,
    visible_mask_id: str | None,
) -> np.ndarray:
    width, height = geometry.raw_size
    inside = (raw_x >= 0) & (raw_x < width) & (raw_y >= 0) & (raw_y < height)
    if visible_mask_id is None:
        return inside
    if visible_mask_id not in MASK_IDS:
        raise ValueError("Unknown visible body-witness mask")
    crop_top = geometry.crop_box[1]
    actor_top, actor_bottom = geometry.native_box[1], geometry.native_box[3]
    actor_center_raw = (actor_top + actor_bottom) / 2 - crop_top
    half_gap = 0.05 * (actor_bottom - actor_top) / 2
    if visible_mask_id == "upper_visible":
        target = raw_y >= actor_center_raw + half_gap
    else:
        target = raw_y < actor_center_raw - half_gap
    return inside & target


def measure_pose_heatmaps(
    selected_heatmaps: np.ndarray,
    source_to_model: np.ndarray,
    geometry: BodyCropGeometry,
    *,
    visible_mask_id: str | None = None,
) -> PoseMeasurement:
    """Convert raw selected-joint heatmaps into the fixed 193-bin descriptor.

    The first 192 bins cover a 16x12 grid over raw crop coordinates.  All
    non-target mass, including letterbox padding and the excluded middle band,
    is retained in the final outside bin.
    """

    heatmaps, affine = _validate_inputs(selected_heatmaps, source_to_model, geometry)
    raw_x, raw_y = _raw_cell_coordinates(affine)
    target = _target_region(raw_x, raw_y, geometry, visible_mask_id)
    width, height = geometry.raw_size
    grid_x = np.floor(raw_x / width * DESCRIPTOR_WIDTH).astype(np.int64)
    grid_y = np.floor(raw_y / height * DESCRIPTOR_HEIGHT).astype(np.int64)
    grid_x = np.clip(grid_x, 0, DESCRIPTOR_WIDTH - 1)
    grid_y = np.clip(grid_y, 0, DESCRIPTOR_HEIGHT - 1)
    flat_bins = grid_y * DESCRIPTOR_WIDTH + grid_x

    positive = np.maximum(heatmaps, 0.0).astype(np.float64)
    descriptors = np.zeros((8, DESCRIPTOR_BINS), dtype=np.float64)
    for landmark in range(8):
        np.add.at(
            descriptors[landmark, : DESCRIPTOR_BINS - 1],
            flat_bins[target],
            positive[landmark][target],
        )
        descriptors[landmark, -1] = positive[landmark][~target].sum()
    descriptors[:, -1] += NORMALIZATION_EPSILON
    descriptors /= descriptors.sum(axis=1, keepdims=True)

    flat_peak = heatmaps.reshape(8, -1).argmax(axis=1)
    peak_y, peak_x = np.divmod(flat_peak, HEATMAP_WIDTH)
    peak_model = np.stack(
        (
            (peak_x.astype(np.float64) + 0.5) * (MODEL_WIDTH / HEATMAP_WIDTH),
            (peak_y.astype(np.float64) + 0.5) * (MODEL_HEIGHT / HEATMAP_HEIGHT),
            np.ones(8, dtype=np.float64),
        ),
        axis=1,
    )
    peak_raw = peak_model @ np.linalg.inv(affine).T
    peak_raw = peak_raw[:, :2] / peak_raw[:, 2:]
    peaks = heatmaps.reshape(8, -1).max(axis=1)

    descriptor = descriptors.astype(np.float32)
    if not np.allclose(descriptor.sum(axis=1), 1.0, atol=5e-6, rtol=0):
        raise RuntimeError("Pose descriptor normalization failed")
    return PoseMeasurement(
        descriptor=descriptor,
        peak_magnitude=peaks.astype(np.float32),
        peak_raw_xy=peak_raw.astype(np.float32),
    )
