"""Expose CPTR initialization semantics using synthetic tensors, without data or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from hac.cptr import PartTrajectoryResidualNetwork
from hac.vcoco_v3_neural import decode_factorized_logits
from hac.vcoco_v3_temporal import StaticStudentOutput, TemporalOutput, uniform_clip_indices


class SyntheticAnchor(nn.Module):
    """A frozen, constant-logit stand-in for an established classifier."""

    def __init__(self, *, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        self.encoder = nn.Sequential(nn.LayerNorm(4))
        self.register_buffer("logits", torch.tensor([0.0, 2.0] if temporal else [2.0, 0.0]))
        self.last_mask: torch.Tensor | None = None

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None):
        logits = self.logits.expand(len(values), -1)
        probabilities = decode_factorized_logits(logits, logits)
        features = torch.zeros(len(values), 4, dtype=values.dtype, device=values.device)
        if self.temporal:
            self.last_mask = mask.clone() if mask is not None else None
            return TemporalOutput(probabilities, logits, logits, features, values.new_zeros(1))
        return StaticStudentOutput(
            probabilities, logits, logits, values.new_zeros(len(values)), features
        )


def synthetic_model() -> PartTrajectoryResidualNetwork:
    # fork_rng makes this diagnostic independent of, and invisible to, caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260906)
        return PartTrajectoryResidualNetwork(
            SyntheticAnchor(temporal=False),
            SyntheticAnchor(temporal=True),
            frame_input_dim=4,
            quality_dim=8,
            trajectory_sequence_dim=21,
            trajectory_summary_dim=58,
            model_dim=4,
            layers=1,
            heads=2,
            feedforward_dim=8,
            dropout=0.0,
            use_short=True,
        ).eval()


def synthetic_kwargs(valid_frames: int) -> dict:
    if valid_frames not in (0, 4, 8):
        raise ValueError("The fixed diagnostic uses exactly 0, 4, or 8 valid frames")
    mask = torch.arange(8)[None] < valid_frames
    return {
        "static_features": torch.zeros(1, 4),
        "short_features": torch.zeros(1, 8, 4),
        "short_valid_mask": mask,
        "short_centre_index": 4,
        "quality_features": torch.zeros(1, 8),
    }


@torch.inference_mode()
def initialization_report() -> dict:
    model = synthetic_model()
    rows = []
    for valid_frames in (8, 4, 0):
        kwargs = synthetic_kwargs(valid_frames)
        output = model(**kwargs)
        static = model.static_fallback(kwargs["static_features"])
        legacy = model.legacy_temporal(kwargs["short_features"], torch.ones(1, 8, dtype=torch.bool))
        coefficient = torch.sigmoid(torch.tensor(5.0)) * (valid_frames / 8)
        expected_logits = static.posture_logits + coefficient * (
            legacy.posture_logits - static.posture_logits
        )
        rows.append(
            {
                "valid_fraction": valid_frames / 8,
                "effective_legacy_coefficient": float(output.posture_gates[0, 0]),
                "expected_coefficient": float(coefficient),
                "max_logit_equation_error": float(
                    (output.posture_logits - expected_logits).abs().max()
                ),
                "new_residual_max_abs": float(output.learned_temporal_residual.abs().max()),
                "probabilities": output.probabilities[0].tolist(),
                "baseline_probabilities": legacy.probabilities[0].tolist(),
                "max_probability_change_from_baseline": float(
                    (output.probabilities - legacy.probabilities).abs().max()
                ),
                "predicted_class_index": int(output.probabilities.argmax(1)[0]),
                "baseline_predicted_class_index": int(legacy.probabilities.argmax(1)[0]),
                "baseline_exactly_preserved": bool(
                    torch.equal(output.probabilities, legacy.probabilities)
                ),
                "static_exactly_returned": bool(
                    torch.equal(output.probabilities, static.probabilities)
                ),
            }
        )
    indices = uniform_clip_indices(17, center_index=8, samples=8, span_frames=8)
    return {
        "status": "SYNTHETIC_CPTR_INITIALIZATION_DIAGNOSTIC",
        "scope": "Current implementation; constant synthetic anchors; no accuracy estimate",
        "dataset_rows_read": 0,
        "checkpoints_loaded": 0,
        "fit_steps": 0,
        "seed": 20260906,
        "torch_version": torch.__version__,
        "rows": rows,
        "declared_short_clip_sampler": {
            "frame_count": 17,
            "center_index": 8,
            "samples": 8,
            "span_frames": 8,
            "indices": indices.tolist(),
            "unique_frames": len(set(indices.tolist())),
        },
        "interpretation": (
            "Zero-initialized new heads do not preserve the temporal baseline: its "
            "coefficient is sigmoid(5) times valid fraction. This algebraic result "
            "does not establish the cause or magnitude of real-data performance losses."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = initialization_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive write keeps earlier audit artifacts immutable.
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
