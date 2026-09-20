"""Tracker-free persistent token/gauge features for the PTG discovery probe.

The implementation is deliberately small and deterministic.  It uses a fixed
rank slice of the verified DINO token embedding and local five-neighbour
transport on the 12x12 token grid.  The transport is an evidence carrier, not a
task label or a learned tracker.  It exposes raw path, time-even/time-odd, and
common-translation residual statistics to a separate reader.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def grid_coordinates(device: torch.device | str = "cpu") -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(12, device=device, dtype=torch.float32),
        torch.arange(12, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def local_neighbour_indices(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    coords = grid_coordinates(device)
    offsets = torch.tensor(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        dtype=torch.float32,
        device=device,
    )
    targets = coords[:, None, :] + offsets[None, :, :]
    valid = ((targets >= 0) & (targets < 12)).all(-1)
    indices = (targets[..., 0] * 12 + targets[..., 1]).clamp(0, 143).long()
    distance = offsets.square().sum(-1)
    return indices, valid, distance


def _macro_pool(values: torch.Tensor) -> torch.Tensor:
    """Pool [B,T,144,R] into [B,T,9,R] without destroying the grid axes."""
    b, t, n, r = values.shape
    if n != 144:
        raise ValueError("PTG expects a 12x12 token grid")
    return (
        values.reshape(b, t, 3, 4, 3, 4, r)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(b, t, 9, 16, r)
        .mean(-2)
    )


def _pair_transport(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local soft identity transport for one adjacent frame pair."""
    indices, valid, distance = local_neighbour_indices(source.device)
    neighbours = target[:, indices, :]
    query = F.normalize(source, dim=-1, eps=1e-6)
    keys = F.normalize(neighbours, dim=-1, eps=1e-6)
    scores = (query[:, :, None, :] * keys).sum(-1) - 0.02 * distance[None, None, :]
    scores = scores.masked_fill(~valid[None, :, :], -torch.inf)
    assignment = torch.softmax(scores / 0.10, dim=-1)
    transported = (assignment[..., None] * neighbours).sum(-2)
    neighbour_coords = grid_coordinates(source.device)[indices]
    expected_coords = (assignment[..., None] * neighbour_coords[None, :, :, :]).sum(-2)
    entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(-1)
    return transported, expected_coords, entropy


def _validate_inputs(tokens: torch.Tensor, times: torch.Tensor) -> None:
    if tokens.ndim != 4 or tokens.shape[2:] != (144, 768):
        raise ValueError("tokens must be [batch,time,144,768]")
    if tokens.shape[1] < 4 or not torch.isfinite(tokens).all():
        raise ValueError("tokens must be finite and contain at least four frames")
    if times.shape != tokens.shape[:2] or not torch.isfinite(times).all() or torch.any(times[:, 1:] <= times[:, :-1]):
        raise ValueError("times must be finite and increasing")


@torch.inference_mode()
def persistent_token_features(
    tokens: torch.Tensor,
    times: torch.Tensor,
    *,
    rank: int = 32,
    center_index: int = 4,
) -> dict[str, torch.Tensor]:
    """Return deterministic raw and gauge/parity features for a batch.

    No labels or task predictions enter this function.  The rank slice is fixed
    before fitting; it is intentionally not estimated from outer labels.
    """
    _validate_inputs(tokens, times)
    if rank != 32 or not 0 <= center_index < tokens.shape[1]:
        raise ValueError("PTG rank and center index are locked")
    values = tokens[..., :rank].float()
    macro = _macro_pool(values)
    center = macro[:, center_index]
    temporal = macro.mean(1)
    pair_changes, pair_flows, pair_entropy = [], [], []
    for step in range(tokens.shape[1] - 1):
        moved, expected, entropy = _pair_transport(values[:, step], values[:, step + 1])
        pair_changes.append(moved - values[:, step])
        pair_flows.append(expected - grid_coordinates(tokens.device)[None])
        pair_entropy.append(entropy)
    changes = torch.stack(pair_changes, dim=1)
    flows = torch.stack(pair_flows, dim=1)
    entropy = torch.stack(pair_entropy, dim=1)
    change_macro = _macro_pool(changes)
    flow_macro = _macro_pool(flows)
    # Estimate and remove the common crop translation.  The residual is a
    # lightweight gauge witness; no raw-pixel tracker is involved.
    common_flow = flow_macro.mean(2)
    residual_flow = flow_macro - common_flow[:, :, None, :]
    even_terms, odd_terms = [], []
    max_lag = min(center_index, tokens.shape[1] - 1 - center_index)
    for lag in range(1, max_lag + 1):
        plus = macro[:, center_index + lag]
        minus = macro[:, center_index - lag]
        even_terms.append((plus + minus) / 2 - center)
        dt = (times[:, center_index + lag] - times[:, center_index - lag]).clamp_min(1e-3)
        odd_terms.append((plus - minus) / dt[:, None, None])
    even = torch.stack(even_terms, dim=1).mean(1)
    odd = torch.stack(odd_terms, dim=1).mean(1)
    raw = torch.cat(
        (
            center.flatten(1),
            temporal.flatten(1),
            change_macro.mean(1).flatten(1),
            change_macro.std(1, unbiased=False).flatten(1),
        ),
        dim=1,
    )
    gauge = torch.cat(
        (
            raw,
            even.flatten(1),
            odd.flatten(1),
            residual_flow.mean(1).flatten(1),
            residual_flow.std(1, unbiased=False).flatten(1),
            entropy.mean((1, 2)).unsqueeze(1),
            entropy.std((1, 2), unbiased=False).unsqueeze(1),
        ),
        dim=1,
    )
    return {
        "raw": raw,
        "gauge": gauge,
        "center": center.flatten(1),
        "cycle_witness": residual_flow,
        "assignment_entropy": entropy,
    }


@torch.inference_mode()
def coarse_temporal_features(tokens: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Matched low-resolution control from repeated 3x3-compatible tokens."""
    if tokens.ndim != 4 or tokens.shape[2:] != (9, 768):
        raise ValueError("coarse tokens must be [batch,time,9,768]")
    values = tokens[..., :32].float()
    macro = values
    center = macro[:, 4]
    temporal = macro.mean(1)
    change = macro[:, 1:] - macro[:, :-1]
    return torch.cat((center.flatten(1), temporal.flatten(1), change.mean(1).flatten(1), change.std(1, unbiased=False).flatten(1)), dim=1)


class PTGDiscoveryHead(nn.Module):
    """Small matched-capacity three-class discovery reader."""

    def __init__(self, input_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
