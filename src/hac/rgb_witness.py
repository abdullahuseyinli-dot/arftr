"""Matched pose-free conditional RGB-witness heads and bounded action."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("direct", "template", "conditional", "unconditional")
EXPECTED_PARAMETERS = {"direct": 25601, "template": 26241, "conditional": 26145, "unconditional": 25825}
PCA_DIM, GEOMETRY_DIM = 96, 6


def _linear(layer: nn.Linear, *, zero: bool = False) -> None:
    if zero:
        nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    else:
        nn.init.xavier_uniform_(layer.weight, gain=1.0); nn.init.zeros_(layer.bias)


class Reader(nn.Module):
    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.first, self.final = nn.Linear(input_dim, width), nn.Linear(width, 1)
        _linear(self.first); _linear(self.final, zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(F.gelu(self.first(x))).squeeze(-1)


class ConditionalDecoder(nn.Module):
    def __init__(self, conditional: bool) -> None:
        super().__init__()
        self.conditional = conditional
        self.first, self.final = nn.Linear(104, 64), nn.Linear(64, PCA_DIM)
        _linear(self.first); _linear(self.final)
        if conditional:
            self.gamma = nn.Parameter(torch.zeros(2, 64))
            self.beta = nn.Parameter(torch.zeros(2, 64))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context [B,2,104]; output [B,2,C,96], C=2 or1.
        h = F.gelu(self.first(context))
        if self.conditional:
            h = h[:, :, None] * (1 + self.gamma[None, None]) + self.beta[None, None]
        else:
            h = h[:, :, None]
        return self.final(h)


class RGBWitness(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown RGB-witness arm")
        self.arm = arm
        if arm == "direct":
            self.reader = Reader(198, 128)
        elif arm == "template":
            self.templates = nn.Parameter(torch.zeros(2, 2, PCA_DIM))  # mask,class,target
            self.reader = Reader(200, 128)
        elif arm == "conditional":
            self.decoder = ConditionalDecoder(True)
            self.reader = Reader(200, 64)
        else:
            self.decoder = ConditionalDecoder(False)
            self.reader = Reader(199, 64)
        if self.trainable_parameters != EXPECTED_PARAMETERS[arm]:
            raise RuntimeError("RGB-witness active parameter count changed")

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, features: torch.Tensor, geometry: torch.Tensor, available: torch.Tensor,
                anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if (features.ndim != 3 or features.shape[1:] != (2, PCA_DIM)
                or geometry.shape != (len(features), GEOMETRY_DIM)
                or available.shape != (len(features), 2)
                or anchor.shape != (len(features), 3)):
            raise ValueError("Malformed RGB-witness inputs")
        if not all(torch.isfinite(x).all() for x in (features, geometry, anchor)):
            raise ValueError("RGB-witness inputs must be finite")
        direct = torch.cat((features.flatten(1), geometry), dim=1)
        energies = None
        predictions = None
        if self.arm == "direct":
            reader_input = direct
        elif self.arm == "template":
            predictions = self.templates[None].expand(len(features), -1, -1, -1)
            targets = features.flip(1)[:, :, None]
            energies = (predictions - targets).square().mean(-1)  # [B,mask,class]
            reader_input = torch.cat((direct, energies.mean(1)), 1)
        else:
            mask = torch.eye(2, dtype=features.dtype, device=features.device)[None].expand(len(features), -1, -1)
            geom = geometry[:, None].expand(-1, 2, -1)
            context = torch.cat((features, geom, mask), -1)
            predictions = self.decoder(context)
            targets = features.flip(1)[:, :, None]
            energies = (predictions - targets).square().mean(-1)
            reader_input = torch.cat((direct, energies.mean(1)), 1)
        delta = .5 * torch.tanh(self.reader(reader_input))
        eligible = available.all(1) & (anchor[:, 2] < torch.minimum(anchor[:, 0], anchor[:, 1]))
        effective = delta * eligible.to(delta.dtype)
        q = anchor[:, 0] + anchor[:, 1]
        odds = torch.log(anchor[:, 0].clamp_min(1e-12) / anchor[:, 1].clamp_min(1e-12)) + effective
        sit = q * torch.sigmoid(odds)
        candidate = torch.stack((sit, q - sit, anchor[:, 2]), 1)
        candidate = torch.where(eligible[:, None], candidate, anchor)
        return {"candidate": candidate, "delta": delta, "effective_delta": effective,
                "eligible": eligible, "observation_available": available.all(1),
                "energies": energies, "predictions": predictions}


def rgb_witness_loss(output: dict[str, torch.Tensor], features: torch.Tensor,
                     anchor: torch.Tensor, labels: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rows = len(labels)
    candidate = output["candidate"]
    task = -torch.log(candidate[torch.arange(rows, device=labels.device), labels].clamp_min(1e-12)).mean()
    kl = (anchor * (torch.log(anchor.clamp_min(1e-12)) - torch.log(candidate.clamp_min(1e-12)))).sum(1).mean()
    residual = output["effective_delta"].abs().mean()
    witness = candidate.new_zeros(())
    energy_ce = candidate.new_zeros(())
    posture = (labels < 2) & output["observation_available"]
    if arm != "direct" and posture.any():
        predictions = output["predictions"]
        targets = features.flip(1)[:, :, None]
        if arm == "unconditional":
            per = (predictions - targets).square().mean((1, 2, 3))
        else:
            selected = predictions[torch.arange(rows, device=labels.device), :, labels.clamp_max(1)]
            per = (selected - features.flip(1)).square().mean((1, 2))
        witness = (per * posture).sum() / rows
        if arm in ("template", "conditional"):
            logits = -output["energies"].mean(1)
            ce = F.cross_entropy(logits, labels.clamp_max(1), reduction="none")
            energy_ce = (ce * posture).sum() / rows
    total = task + .1 * kl + .01 * residual + witness + .1 * energy_ce
    return total, {"task": task, "kl": kl, "residual": residual, "witness": witness, "energy_ce": energy_ce}


def bounded_pair_candidate(anchor: np.ndarray, delta: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy parity path used by audits; rejected rows copy anchor bytes."""
    p = np.asarray(anchor)
    d = np.asarray(delta, dtype=np.float64)
    av = np.asarray(available, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3 or d.shape != (len(p),) or av.shape != (len(p), 2):
        raise ValueError("Malformed pair candidate inputs")
    eligible = av.all(1) & (p[:, 2] < np.minimum(p[:, 0], p[:, 1]))
    out = p.copy()
    odds = np.log(np.clip(p[:, 0], 1e-12, 1) / np.clip(p[:, 1], 1e-12, 1)) + np.clip(d, -.5, .5)
    sit = (p[:, 0] + p[:, 1]) / (1 + np.exp(-odds))
    out[eligible, 0] = sit[eligible]
    out[eligible, 1] = p[eligible, 0] + p[eligible, 1] - sit[eligible]
    out[eligible, 2] = p[eligible, 2]
    return out, eligible


@dataclass(frozen=True)
class ScopeTransform:
    pca_mean: np.ndarray
    components: np.ndarray
    pca_scale: np.ndarray
    geometry_mean: np.ndarray
    geometry_scale: np.ndarray

    def apply(self, features: np.ndarray, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = np.einsum("nsd,sdk->nsk", np.asarray(features, np.float64) - self.pca_mean[None], self.components)
        projected = np.clip(projected / self.pca_scale[None], -8, 8).astype(np.float32)
        normalized_geometry = ((np.asarray(geometry, np.float64) - self.geometry_mean) / self.geometry_scale).astype(np.float32)
        return projected, normalized_geometry


def fit_scope_transform(features: np.ndarray, geometry: np.ndarray) -> ScopeTransform:
    """Deterministic per-slot covariance PCA; no labels or held rows."""
    values = np.asarray(features, dtype=np.float64)
    geom = np.asarray(geometry, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 768) or len(geom) != len(values):
        raise ValueError("Malformed scope-transform inputs")
    means, components, scales = [], [], []
    for slot in range(2):
        mean = values[:, slot].mean(0)
        centered = values[:, slot] - mean
        # Fixed symmetric eigensolver; descending eigenvalue, deterministic sign.
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered)-1, 1))
        order = np.argsort(eigenvalues)[::-1][:PCA_DIM]
        comp = vectors[:, order]
        pivot = np.argmax(np.abs(comp), axis=0)
        comp *= np.where(comp[pivot, np.arange(PCA_DIM)] < 0, -1, 1)
        means.append(mean); components.append(comp); scales.append(np.sqrt(np.maximum(eigenvalues[order], 1e-8)))
    gmean = geom.mean(0); gscale = geom.std(0); gscale[gscale == 0] = 1
    return ScopeTransform(np.stack(means), np.stack(components), np.maximum(np.stack(scales), 1e-4), gmean, gscale)
