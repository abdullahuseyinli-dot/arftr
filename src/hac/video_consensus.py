"""Deterministic cross-view probability consensus for the Okutama P6 replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ARMS = (
    "ocvc_uniform_diverse_triad",
    "ocvc_uniform_spatial_sensitivity",
)

ARM_COMPONENTS = {
    ARMS[0]: (
        ("p3", "long_vjepa_mean"),
        ("p3", "dual_scale_vjepa_dino"),
        ("p5", "orthogonal_moments_factorized"),
    ),
    ARMS[1]: (
        ("p3", "long_vjepa_mean"),
        ("p3", "dual_scale_vjepa_dino"),
        ("p5", "spatial_contrast_factorized"),
    ),
}


def validate_probabilities(values: np.ndarray, *, rows: int | None = None) -> None:
    """Fail closed unless values are a finite multiclass probability matrix."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("OCVC requires an [N,3] probability matrix")
    if rows is not None and len(values) != rows:
        raise ValueError("OCVC component row count changed")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("OCVC component contains invalid probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=2e-6):
        raise ValueError("OCVC component rows do not sum to one")


def uniform_consensus(components: Sequence[np.ndarray]) -> np.ndarray:
    """Average aligned probability views in float64 and explicitly renormalize."""
    if len(components) < 2:
        raise ValueError("OCVC requires at least two representation views")
    rows = len(components[0])
    for values in components:
        validate_probabilities(values, rows=rows)
    output = np.zeros((rows, 3), dtype=np.float64)
    for values in components:
        output += np.asarray(values, dtype=np.float64)
    output /= np.float64(len(components))
    output /= output.sum(axis=1, keepdims=True)
    validate_probabilities(output, rows=rows)
    return output


def derive_consensus(
    sources: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Materialize both locked P6 arms without labels, fitting, or routing."""
    outputs: dict[str, np.ndarray] = {}
    for arm in ARMS:
        try:
            components = [sources[phase][name] for phase, name in ARM_COMPONENTS[arm]]
        except KeyError as error:
            raise ValueError(f"Missing locked OCVC component: {error}") from error
        outputs[arm] = uniform_consensus(components)
    return outputs
