"""Local-only CoTracker3 carrier and independent endpoint round-trip checks.

Official API: https://github.com/facebookresearch/co-tracker/blob/main/cotracker/predictor.py
No Hub calls, downloads, checkpoint selection, activity labels or task fitting.
The upstream predictor resizes every source to its model resolution. We perform
that same align_corners=True resize one frame at a time to bound native4K GPU
memory, and invert only the coordinate *scale*, never a tracking cycle.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hac.persistent_articulation_v2 import TrackSequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_model_identity(repository: Path, checkpoint: Path) -> dict:
    repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
    if not (repository / "cotracker/predictor.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Pinned local CoTracker repository and checkpoint are required")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Tracked CoTracker source changes are not permitted")
    sources = {
        str(p.relative_to(repository)).replace("\\", "/"): file_sha256(p)
        for p in sorted((repository / "cotracker").rglob("*.py"))
    }
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(sources.items())).encode()
    ).hexdigest()
    return {
        "repository": str(repository),
        "revision": revision,
        "python_tree_sha256": digest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "predictor_sha256": sources["cotracker/predictor.py"],
        "model": "CoTracker3_scaled_offline",
    }


@dataclass(frozen=True)
class PreparedVideo:
    tensor: object
    source_height: int
    source_width: int


class LocalCoTracker3:
    def __init__(self, identity: dict, *, device: str = "cuda"):
        actual = local_model_identity(Path(identity["repository"]), Path(identity["checkpoint"]))
        if actual != identity:
            raise ValueError("Local tracker code/checkpoint identity changed after preparation")
        import torch

        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "This bounded CoTracker smoke requires the isolated CUDA environment"
            )
        repository = str(Path(identity["repository"]).resolve())
        for name, module in tuple(sys.modules.items()):
            if name == "cotracker" or name.startswith("cotracker."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(Path(repository)):
                    raise RuntimeError("A different CoTracker implementation is already imported")
        sys.path.insert(0, repository)
        module = importlib.import_module("cotracker.predictor")
        self.predictor = (
            module.CoTrackerPredictor(
                checkpoint=identity["checkpoint"], offline=True, v2=False, window_len=60
            )
            .eval()
            .to(device)
        )
        self.torch, self.device = torch, device
        self.model_shape = tuple(int(x) for x in self.predictor.interp_shape)
        self.identity = actual
        self.calls = 0

    def prepare_video(self, frames: list[np.ndarray]) -> PreparedVideo:
        """Accept RGB uint8; resize as upstream does, without a 6GB 4K tensor."""
        if len(frames) < 2:
            raise ValueError("At least two RGB frames are required")
        h, w, c = frames[0].shape
        if c != 3 or min(h, w) < 2:
            raise ValueError("Expected RGB frames")
        torch = self.torch
        resized = []
        with torch.inference_mode():
            for frame in frames:
                if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
                    raise ValueError("All frames must be aligned RGB uint8")
                source = (
                    torch.from_numpy(np.ascontiguousarray(frame))
                    .permute(2, 0, 1)[None]
                    .to(self.device)
                    .float()
                )
                resized.append(
                    torch.nn.functional.interpolate(
                        source, self.model_shape, mode="bilinear", align_corners=True
                    )[0]
                )
            tensor = torch.stack(resized)[None]
        return PreparedVideo(tensor, h, w)

    def predict(self, video: PreparedVideo, queries: np.ndarray, *, reverse: bool = False):
        """Queries are (physical frame index, original-source x, original-source y)."""
        torch = self.torch
        values = np.asarray(queries, dtype=np.float64)
        count = int(video.tensor.shape[1])
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Queries must be finite [identity, (frame,x,y)]")
        if (
            not len(values)
            or np.any(values[:, 0] != np.floor(values[:, 0]))
            or np.any((values[:, 0] < 0) | (values[:, 0] >= count))
        ):
            raise ValueError("Query frames must be in-range integers")
        h, w = self.model_shape
        scale = np.asarray(
            [(w - 1) / (video.source_width - 1), (h - 1) / (video.source_height - 1)]
        )
        scaled = values.copy()
        scaled[:, 1:] *= scale
        with torch.inference_mode():
            query = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)[None]
            tensor = video.tensor.flip(1) if reverse else video.tensor
            tracks, visibility = self.predictor(tensor, queries=query, backward_tracking=True)
            positions = tracks[0].detach().cpu().numpy().astype(np.float64) / scale
            valid = visibility[0].detach().cpu().numpy().astype(bool)
        if valid.ndim == 3 and valid.shape[-1] == 1:
            valid = valid[..., 0]
        if positions.shape != (count, len(values), 2) or valid.shape != (count, len(values)):
            raise RuntimeError("Unexpected upstream point/visibility shape")
        valid &= np.isfinite(positions).all(axis=-1)
        valid &= (positions[..., 0] >= 0) & (positions[..., 0] < video.source_width)
        valid &= (positions[..., 1] >= 0) & (positions[..., 1] < video.source_height)
        positions[~valid] = np.nan
        self.calls += 1
        return positions, valid


@dataclass(frozen=True)
class CycledTracks:
    sequence: TrackSequence
    cycle_error: np.ndarray  # [sampled_time, identity], center always NaN
    cycle_valid: np.ndarray
    sampled_indices: tuple[int, ...]
    predictor_calls: int
    center_index: int = 30


def independently_cycled_tracks(
    predict: Callable,
    seeds: np.ndarray,
    *,
    times_seconds: np.ndarray,
    center_index: int,
    sampled_indices: tuple[int, ...],
    point_ids: tuple[str, ...],
) -> CycledTracks:
    """Re-query each noncenter endpoint on a reversed video, then observe center.

    No predicted coordinate is transformed back algebraically. Each round trip
    is a new tracker inference with endpoint locations as the only query.
    Missing endpoint/return visibility remains missing, not a zero cycle error.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    times = np.asarray(times_seconds, dtype=np.float64)
    count, points = len(times), len(seeds)
    if seeds.shape != (points, 2) or points != len(point_ids) or not np.isfinite(seeds).all():
        raise ValueError("Seed identities and coordinates must agree")
    if not 0 <= center_index < count or tuple(sorted(set(sampled_indices))) != sampled_indices:
        raise ValueError("Center/sample indices are invalid")
    if (
        not sampled_indices
        or sampled_indices[0] < 0
        or sampled_indices[-1] >= count
        or center_index not in sampled_indices
    ):
        raise ValueError("Samples must contain center and stay within video")
    queries = np.column_stack((np.full(points, center_index), seeds))
    positions, valid = predict(queries, False)
    sequence = TrackSequence(positions, valid, times, point_ids)
    if sequence.positions.shape != (count, points, 2):
        raise ValueError("Predictor changed the query identity/time population")
    center_error = np.linalg.norm(sequence.positions[center_index] - seeds, axis=1)
    if not sequence.valid[center_index].all() or np.max(center_error) > 0.01:
        raise ValueError("Tracker failed center query identity")
    cycles = np.full((len(sampled_indices), points), np.nan)
    cycle_valid = np.zeros(cycles.shape, bool)
    calls = 1
    for row, index in enumerate(sampled_indices):
        if index == center_index:
            continue
        ids = np.flatnonzero(sequence.valid[index])
        if not len(ids):
            continue
        endpoint_queries = np.column_stack(
            (np.full(len(ids), count - 1 - index), sequence.positions[index, ids])
        )
        returned, returned_valid = predict(endpoint_queries, True)
        calls += 1
        returned, returned_valid = np.asarray(returned), np.asarray(returned_valid, bool)
        if returned.shape != (count, len(ids), 2) or returned_valid.shape != (count, len(ids)):
            raise ValueError("Reverse tracker changed query identities")
        reverse_center = count - 1 - center_index
        ok = returned_valid[reverse_center] & np.isfinite(returned[reverse_center]).all(axis=1)
        cycles[row, ids[ok]] = np.linalg.norm(returned[reverse_center, ok] - seeds[ids[ok]], axis=1)
        cycle_valid[row, ids[ok]] = True
    return CycledTracks(sequence, cycles, cycle_valid, sampled_indices, calls, center_index)


def point_survival_and_cycle(
    tracks: CycledTracks, diagonal: float, *, actor_count: int = 64
) -> dict:
    if (
        not np.isfinite(diagonal)
        or diagonal <= 0
        or not 0 < actor_count <= len(tracks.sequence.point_ids)
    ):
        raise ValueError("Positive diagonal and nonempty actor identities required")
    valid = tracks.sequence.valid[np.asarray(tracks.sampled_indices), :actor_count]
    cycle = tracks.cycle_error[:, :actor_count]
    finite = np.isfinite(cycle) & tracks.cycle_valid[:, :actor_count]
    center_row = tracks.sampled_indices.index(tracks.center_index)
    if finite[center_row].any():
        raise ValueError("Center must not contribute to measured tracking cycles")
    noncenter = np.arange(len(valid)) != center_row
    returned_all = np.all(finite[noncenter], axis=0)
    joint = returned_all & np.all(valid, axis=0)
    return {
        "actor_survival_all_nine": float(np.mean(np.all(valid, axis=0))),
        "actor_reverse_survival_all_noncenter": float(np.mean(returned_all)),
        "actor_joint_forward_reverse_survival": float(np.mean(joint)),
        "cycle_median_normalized": float(np.median(cycle[finite]) / diagonal)
        if finite.any()
        else None,
        "cycle_pairs_observed": int(finite.sum()),
        "cycle_pairs_expected": (len(valid) - 1) * actor_count,
        "cycle_missing_pairs": (len(valid) - 1) * actor_count - int(finite.sum()),
        "center_excluded": True,
        "independent_tracker_roundtrip": True,
    }
