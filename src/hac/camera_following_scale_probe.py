"""Matched-detail inputs in one independently estimated camera-following frame.

The camera uses only fixed even background identities from the 720 carrier.
The crop comes from one center box. No actor trajectory, future box, activity
label, tracker inference, adaptive crop size, or transform interpolation enters
these helpers. Both detail arms share every camera/ROI sampling coordinate.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from hac.learned_persistent_tracking import PreparedVideo
from hac.persistent_articulation_v2 import apply_similarity, fit_similarity
from hac.tracking_scale_probe import FixedCrop

BACKGROUND_FIT_IDS = tuple(range(0, 64, 2))
BACKGROUND_HELD_IDS = tuple(range(1, 64, 2))
# OpenCV INTER_AREA's native -> exact one-third source pixel-center map.
NATIVE_TO_720 = np.asarray([[1 / 3, 0.0, -1 / 3], [0.0, 1 / 3, -1 / 3], [0.0, 0.0, 1.0]])
NATIVE_TO_720.setflags(write=False)


def _image_shape(shape, name):
    values = tuple(shape)
    if len(values) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 2 for value in values
    ):
        raise ValueError(f"{name} must be integer (height, width), both at least two")
    return int(values[0]), int(values[1])


def _camera_array(transforms, count):
    values = np.asarray(transforms, dtype=np.float64)
    if values.shape != (count, 3, 3) or not np.isfinite(values).all():
        raise ValueError(
            "Every time needs a finite 3x3 camera; missing frames cannot be interpolated"
        )
    if not np.allclose(values[:, 2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0):
        raise ValueError("Camera transforms must be affine similarities, not projective")
    if not (
        np.allclose(values[:, 0, 0], values[:, 1, 1], atol=1e-10, rtol=0)
        and np.allclose(values[:, 0, 1], -values[:, 1, 0], atol=1e-10, rtol=0)
    ):
        raise ValueError("Camera transforms must be nonreflecting similarities, not shear")
    if np.any(values[:, 0, 0] ** 2 + values[:, 1, 0] ** 2 < 1e-12):
        raise ValueError("Camera similarities must be invertible")
    return values


def camera_from_background(positions720, valid, center_index, actor_diag_native):
    """Fit each center-to-time camera on fixed even IDs; audit odd IDs only.

    Input axes are [time, the original 64 background identities, xy]. A frame
    with fewer than three fitting identities or insufficient rank fails the
    whole call. Held identities never repair a fit. The center transform is
    made exactly identity after the same support/rank check as every frame.

    Returned transforms operate on native pixel centers, including the AREA
    half-pixel origin convention: A_native = inv(S) @ A_720 @ S. The receipt's
    endpoint errors are held-background consistency, not physical ground truth.
    """
    positions = np.asarray(positions720, dtype=np.float64)
    visibility = np.asarray(valid)
    if positions.ndim != 3 or positions.shape[1:] != (64, 2) or len(positions) < 2:
        raise ValueError("Background positions must be [time>=2,64,2] with fixed identity order")
    if visibility.dtype != np.bool_ or visibility.shape != positions.shape[:2]:
        raise ValueError("Background visibility must be boolean [time,64]")
    if not np.isfinite(positions[visibility]).all():
        raise ValueError("Visible background identities must have finite coordinates")
    if (
        isinstance(center_index, bool)
        or not isinstance(center_index, Integral)
        or not 0 <= center_index < len(positions)
    ):
        raise ValueError("Center index must select one original time")
    if not np.isfinite(actor_diag_native) or actor_diag_native <= 0:
        raise ValueError("Native actor diagonal must be positive and finite")

    fitting, held = np.asarray(BACKGROUND_FIT_IDS), np.asarray(BACKGROUND_HELD_IDS)
    inverse_scale = np.linalg.inv(NATIVE_TO_720)
    matrices, fit_counts, held_counts, errors_by_time = [], [], [], []
    for time_index in range(len(positions)):
        common = visibility[center_index] & visibility[time_index]
        usable = fitting[common[fitting]]
        transform = fit_similarity(positions[center_index, usable], positions[time_index, usable])
        if transform is None:
            raise ValueError(
                f"Camera fit failed at time {time_index}: fixed fit support={len(usable)} or rank insufficient; "
                "no held substitution or interpolation"
            )
        if time_index == center_index:
            transform = np.eye(3, dtype=np.float64)
        native = inverse_scale @ transform @ NATIVE_TO_720
        if time_index == center_index:
            native = np.eye(3, dtype=np.float64)
        matrices.append(native)
        fit_counts.append(len(usable))
        usable_held = held[common[held]] if time_index != center_index else np.empty(0, dtype=int)
        held_counts.append(len(usable_held))
        predicted = apply_similarity(transform, positions[center_index, usable_held])
        error = 3.0 * np.linalg.norm(predicted - positions[time_index, usable_held], axis=1)
        errors_by_time.append((error / actor_diag_native).tolist())

    expected = (len(positions) - 1) * len(held)
    flat_error = np.asarray([value for row in errors_by_time for value in row])
    receipt = {
        "status": "CAMERA_FOLLOWING_TRANSFORMS_COMPLETE",
        "camera_fit_ids": list(BACKGROUND_FIT_IDS),
        "camera_held_ids": list(BACKGROUND_HELD_IDS),
        "fit_count_by_time": fit_counts,
        "held_count_by_time": held_counts,
        "held_expected_noncenter_pairs": expected,
        "held_observed_noncenter_pairs": int(len(flat_error)),
        "held_noncenter_support_fraction": len(flat_error) / expected,
        "held_error_normalized_median": float(np.median(flat_error)) if len(flat_error) else None,
        "held_errors_normalized_by_time": errors_by_time,
        "center_index": int(center_index),
        "center_exact_identity": True,
        "center_in_error_aggregate": False,
        "interpolated_frames": 0,
        "coordinate_map": "A_native = inv(S) @ A_720 @ S; S(xy)=(xy+0.5)/3-0.5",
        "native_actor_diagonal": float(actor_diag_native),
        "endpoint": "held_background_consistency_not_physical_ground_truth",
    }
    return _camera_array(np.stack(matrices), len(positions)), receipt


def prepare_camera_crop_pair(backend, full_prepared, native_frames, roi, transforms):
    """Return sharp/low-detail PreparedVideo objects on the same moving lattice.

    Output-local coordinates retain native ROI units: p_native(t) equals
    A_t @ (p_local + [roi.left, roi.top, 1]). Both videos have roi's source
    dimensions and backend.model_shape tensor dimensions, so query scaling and
    the predictor's automatic support-grid geometry are identical.

    High samples native float RGB frame-by-frame. Low samples the exact already
    downscaled full-frame tensor; no native upsample is constructed. The caller
    must hash-bind that tensor's source frames. Outside native pixel-center
    bounds [0,W-1] x [0,H-1], BOTH samples are explicitly zeroed. This avoids
    differing partial border extrapolation at the two source resolutions.
    Use map_camera_tracks_to_native to reject tracks on padding, not clip them.
    """
    if not isinstance(full_prepared, PreparedVideo) or not isinstance(roi, FixedCrop):
        raise ValueError("Expected a PreparedVideo and a FixedCrop")
    frames = list(native_frames)
    if len(frames) < 2 or not isinstance(frames[0], np.ndarray) or frames[0].ndim != 3:
        raise ValueError("At least two aligned native RGB uint8 frames are required")
    height, width, channels = frames[0].shape
    if (
        channels != 3
        or min(height, width) < 2
        or any(
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != (height, width, 3)
            for frame in frames
        )
    ):
        raise ValueError("Native frames must be aligned RGB uint8 with no missing times")
    if roi.right > width or roi.bottom > height:
        raise ValueError("The center ROI must lie inside the native source")
    if (full_prepared.source_height, full_prepared.source_width) != (height, width):
        raise ValueError("Prepared full-frame source dimensions differ from native frames")
    matrices = _camera_array(transforms, len(frames))
    model_height, model_width = _image_shape(backend.model_shape, "Model shape")
    torch, tensor = backend.torch, full_prepared.tensor
    expected = (1, len(frames), 3, model_height, model_width)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
        raise ValueError("Prepared full-frame tensor changed time/model dimensions")
    if tensor.dtype not in (torch.float32, torch.float64) or not torch.isfinite(tensor).all():
        raise ValueError("Prepared full-frame tensor must contain finite float32/float64 pixels")

    high_frames, low_frames = [], []
    with torch.inference_mode():
        xs = torch.linspace(
            roi.left, roi.right - 1, model_width, dtype=tensor.dtype, device=tensor.device
        )
        ys = torch.linspace(
            roi.top, roi.bottom - 1, model_height, dtype=tensor.dtype, device=tensor.device
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        reference = torch.stack((xx, yy), dim=-1)
        for time_index, frame in enumerate(frames):
            transform = torch.as_tensor(
                matrices[time_index], dtype=tensor.dtype, device=tensor.device
            )
            native_xy = reference @ transform[:2, :2].T + transform[:2, 2]
            inside = (
                (native_xy[..., 0] >= 0)
                & (native_xy[..., 0] <= width - 1)
                & (native_xy[..., 1] >= 0)
                & (native_xy[..., 1] <= height - 1)
            )
            grid = native_xy * native_xy.new_tensor([2.0 / (width - 1), 2.0 / (height - 1)]) - 1.0
            grid = grid[None]
            source = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]
            source = source.to(device=tensor.device, dtype=tensor.dtype)
            sharp = torch.nn.functional.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            low = torch.nn.functional.grid_sample(
                tensor[:, time_index],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sharp = torch.where(inside[None, None], sharp, 0.0)
            low = torch.where(inside[None, None], low, 0.0)
            high_frames.append(sharp[0])
            low_frames.append(low[0])
        high = PreparedVideo(torch.stack(high_frames)[None], roi.height, roi.width)
        low = PreparedVideo(torch.stack(low_frames)[None], roi.height, roi.width)
    return high, low


def map_camera_tracks_to_native(localpositions, valid, transforms, roi, native_shape):
    """Map [time, identity, xy] without repacking; reject crop/source padding.

    Visibility is intersected with finite coordinates and closed pixel-center
    bounds [0,size-1] in BOTH local ROI and native image. Missing predictions
    remain NaN, never boundary-clamped. For reverse-time tracker outputs, pass
    transforms[::-1]; the function intentionally does not guess time order.

    At the original center A=I, this map is a translation, hence independently
    measured return-to-center Euclidean cycle errors equal native errors.
    """
    positions = np.asarray(localpositions, dtype=np.float64)
    visibility = np.asarray(valid)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("Local tracks must retain [time, identity, xy] axes")
    if visibility.dtype != np.bool_ or visibility.shape != positions.shape[:2]:
        raise ValueError("Track visibility must be boolean [time,identity]")
    if not isinstance(roi, FixedCrop):
        raise ValueError("Expected the same FixedCrop used to prepare the video")
    height, width = _image_shape(native_shape, "Native shape")
    if roi.right > width or roi.bottom > height:
        raise ValueError("The center ROI must lie inside the native source")
    matrices = _camera_array(transforms, len(positions))
    ok = visibility.copy() & np.isfinite(positions).all(axis=-1)
    ok &= (
        (positions[..., 0] >= 0)
        & (positions[..., 0] <= roi.width - 1)
        & (positions[..., 1] >= 0)
        & (positions[..., 1] <= roi.height - 1)
    )
    reference = positions + np.asarray([roi.left, roi.top])
    mapped = np.einsum("tij,tnj->tni", matrices[:, :2, :2], reference) + matrices[:, None, :2, 2]
    ok &= np.isfinite(mapped).all(axis=-1)
    ok &= (
        (mapped[..., 0] >= 0)
        & (mapped[..., 0] <= width - 1)
        & (mapped[..., 1] >= 0)
        & (mapped[..., 1] <= height - 1)
    )
    mapped[~ok] = np.nan
    return mapped, ok
