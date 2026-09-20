"""Population-parameterized ARFTR composition with strict scenario exclusion.

This module separates the deterministic ARFTR composition ``F(S)`` from the
expensive M4/P6/A3 producers.  It is usable both to replay the five retained
outer populations and to consume newly fitted ancestor artifacts.  Parameter
selection reads labels only on ``S``; prediction rows must be scenario-disjoint.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from hac.actor_memory_base import canonical_hash, probability_metrics
from hac.arftr import ARFTRParameters, apply_arftr, exact_track_neighbors

SEEDS = (42, 43, 44)


def _probabilities(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.shape != shape
        or not np.issubdtype(result.dtype, np.floating)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or not np.allclose(result.sum(-1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"Malformed {name} probabilities")
    return result


def parameter_grid(protocol: dict[str, Any]) -> tuple[ARFTRParameters, ...]:
    selection = protocol["selection"]
    result = tuple(
        ARFTRParameters(*values)
        for values in itertools.product(
            selection["posture_restoration"],
            selection["motion_restoration"],
            selection["a3_motion_residual"],
            selection["temporal_strength"],
        )
    )
    if len(result) != 300 or selection.get("candidates_per_outer_fold") != 300:
        raise RuntimeError("Historical ARFTR parameter grid changed")
    return result


def select_parameters(
    labels: np.ndarray,
    m4: np.ndarray,
    p6: np.ndarray,
    a3: np.ndarray,
    neighbors: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[ARFTRParameters, dict[str, Any]]:
    candidates = []
    epsilon = protocol["factorization"]["epsilon"]
    for parameters in parameter_grid(protocol):
        probabilities = apply_arftr(
            m4,
            p6,
            a3,
            neighbors,
            parameters,
            epsilon=epsilon,
        )
        candidates.append((parameters, probability_metrics(labels, probabilities)))
    selected, metrics = min(
        candidates,
        key=lambda item: (
            -item[1]["macro_f1"],
            item[1]["nll"],
            float(item[0].as_array().sum()),
            tuple(float(value) for value in item[0].as_array()),
        ),
    )
    return selected, metrics


@dataclass(frozen=True)
class PopulationInputs:
    """Aligned inputs produced without labels from the prediction scenarios."""

    m4_inner: np.ndarray
    p6_inner: np.ndarray
    a3_inner: np.ndarray
    m4_outer: np.ndarray
    p6_outer: np.ndarray
    a3_outer: np.ndarray


@dataclass(frozen=True)
class PopulationOutput:
    population_id: str
    training_scenarios: tuple[str, ...]
    prediction_scenarios: tuple[str, ...]
    parameters: ARFTRParameters
    training_metrics: dict[str, Any]
    seed_probabilities: np.ndarray
    mean_probabilities: np.ndarray
    inner_neighbor_map: np.ndarray
    outer_neighbor_map: np.ndarray


def execute_population(
    data: dict[str, np.ndarray],
    train_rows: np.ndarray,
    prediction_rows: np.ndarray,
    inputs: PopulationInputs,
    protocol: dict[str, Any],
) -> PopulationOutput:
    """Select ARFTR inside ``train_rows`` and predict disjoint scenarios."""
    train = np.asarray(train_rows, dtype=np.int64)
    held = np.asarray(prediction_rows, dtype=np.int64)
    total = len(data["labels"])
    if (
        train.ndim != 1
        or held.ndim != 1
        or not len(train)
        or not len(held)
        or not np.array_equal(train, np.unique(train))
        or not np.array_equal(held, np.unique(held))
        or train.min() < 0
        or held.min() < 0
        or train.max() >= total
        or held.max() >= total
        or np.intersect1d(train, held).size
    ):
        raise ValueError("F(S) requires nonempty sorted disjoint row populations")
    train_scenarios = tuple(sorted(set(data["scenarios"][train].tolist())))
    prediction_scenarios = tuple(sorted(set(data["scenarios"][held].tolist())))
    if set(train_scenarios) & set(prediction_scenarios):
        raise RuntimeError("F(S) prediction scenarios overlap its fitting population")
    labels = np.asarray(data["labels"])
    if not np.array_equal(np.unique(labels[train]), np.arange(3)):
        raise RuntimeError("F(S) training population is missing a class")
    n_train, n_held = len(train), len(held)
    m4_inner = _probabilities(inputs.m4_inner, (n_train, 3), "inner M4")
    p6_inner = _probabilities(inputs.p6_inner, (n_train, 3), "inner P6")
    a3_inner = _probabilities(inputs.a3_inner, (n_train, 3), "inner A3")
    m4_outer = _probabilities(inputs.m4_outer, (len(SEEDS), n_held, 3), "outer M4")
    p6_outer = _probabilities(inputs.p6_outer, (n_held, 3), "outer P6")
    a3_outer = _probabilities(inputs.a3_outer, (len(SEEDS), n_held, 3), "outer A3")
    inner_neighbors = exact_track_neighbors(data, train)
    outer_neighbors = exact_track_neighbors(data, held)
    parameters, training_metrics = select_parameters(
        labels[train],
        m4_inner,
        p6_inner,
        a3_inner,
        inner_neighbors,
        protocol,
    )
    seed_probabilities = np.stack(
        [
            apply_arftr(
                m4_outer[index],
                p6_outer,
                a3_outer[index],
                outer_neighbors,
                parameters,
                epsilon=protocol["factorization"]["epsilon"],
            )
            for index in range(len(SEEDS))
        ]
    )
    return PopulationOutput(
        population_id=canonical_hash(list(train_scenarios)),
        training_scenarios=train_scenarios,
        prediction_scenarios=prediction_scenarios,
        parameters=parameters,
        training_metrics=training_metrics,
        seed_probabilities=seed_probabilities,
        mean_probabilities=seed_probabilities.mean(0),
        inner_neighbor_map=inner_neighbors,
        outer_neighbor_map=outer_neighbors,
    )
