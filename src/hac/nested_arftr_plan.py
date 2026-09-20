"""No-fit dependency planner for fixed-recipe ARFTR posture corrections.

This module proves set exclusion and historical split/class feasibility.  It does
not claim that a generalized ``F(S)`` executor exists, that caches are reusable,
or that ``F(T)`` has replayed the retained ARFTR bytes.  Those remain hard gates.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from hac.actor_memory_base import canonical_hash, group_splits

POLICY_PREFIX = "hac-body-policy-v1|"
EXPECTED_CLASSES = np.arange(3)


def policy_blocks(scenarios: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    values = tuple(sorted(scenarios))
    if len(values) < 3 or len(set(values)) != len(values) or any(not x for x in values):
        raise ValueError("Three-block policy needs at least three unique scenario strings")
    ranked = sorted(
        values,
        key=lambda scenario: (
            hashlib.sha256((POLICY_PREFIX + scenario).encode()).hexdigest(),
            scenario,
        ),
    )
    result = tuple(tuple(ranked[index::3]) for index in range(3))
    if any(not block for block in result) or set().union(*map(set, result)) != set(values):
        raise RuntimeError("Policy blocks do not partition the training scenarios")
    return result


def _validate_metadata(
    labels: np.ndarray, scenarios: np.ndarray, folds: np.ndarray, sample_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, groups, outer, ids = map(np.asarray, (labels, scenarios, folds, sample_ids))
    rows = len(y)
    if (
        y.shape != (rows,)
        or groups.shape != (rows,)
        or outer.shape != (rows,)
        or ids.shape != (rows,)
        or y.dtype.kind not in "iu"
        or groups.dtype.kind != "U"
        or ids.dtype.kind != "U"
        or outer.dtype.kind not in "iu"
        or not np.array_equal(np.unique(y), EXPECTED_CLASSES)
        or not np.array_equal(np.unique(outer), np.arange(5))
        or len(np.unique(ids)) != rows
    ):
        raise ValueError("Canonical label/scenario/fold/sample metadata changed")
    for fold in range(5):
        if set(groups[outer == fold]) & set(groups[outer != fold]):
            raise RuntimeError("Outer scenario leakage")
    return y, groups, outer, ids


def _population_receipt(
    population: tuple[str, ...], labels: np.ndarray, scenarios: np.ndarray, sample_ids: np.ndarray
) -> tuple[dict[str, Any], set[tuple[int, ...]]]:
    rows = np.flatnonzero(np.isin(scenarios, population))
    if not np.array_equal(np.unique(labels[rows]), EXPECTED_CLASSES):
        raise RuntimeError("Explicit F population is missing a class")
    base_populations: set[tuple[int, ...]] = {tuple(rows)}
    for inner_train, _ in group_splits(labels, scenarios, rows):
        base_populations.add(tuple(inner_train))
        for base_train, _ in group_splits(labels, scenarios, inner_train):
            base_populations.add(tuple(base_train))
    for nested in base_populations:
        group_splits(labels, scenarios, np.asarray(nested, dtype=np.int64))
    return (
        {
            "id": canonical_hash(list(population)),
            "scenarios": list(population),
            "scenario_count": len(population),
            "rows": int(len(rows)),
            "class_counts": np.bincount(labels[rows], minlength=3).tolist(),
            "sample_ids_sha256": canonical_hash(sample_ids[rows].tolist()),
            "historical_split_class_check": "passed_three_levels",
        },
        base_populations,
    )


def fixed_recipe_plan(
    labels: np.ndarray,
    scenarios: np.ndarray,
    folds: np.ndarray,
    sample_ids: np.ndarray,
    *,
    enforce_canonical_counts: bool = False,
) -> dict[str, Any]:
    y, groups, outer, ids = _validate_metadata(labels, scenarios, folds, sample_ids)
    all_scenarios = set(groups.tolist())
    populations: set[tuple[str, ...]] = set()
    outer_plans = []
    for fold in range(5):
        held = set(groups[outer == fold].tolist())
        training = all_scenarios - held
        blocks = policy_blocks(training)
        outer_population = tuple(sorted(training))
        populations.add(outer_population)
        block_items = []
        for block_index, block in enumerate(blocks):
            ancestor = tuple(sorted(training - set(block)))
            if held & set(ancestor) or set(block) & set(ancestor):
                raise RuntimeError("Three-block correction prior leaks held scenarios")
            populations.add(ancestor)
            block_rows = np.flatnonzero(np.isin(groups, block))
            block_items.append(
                {
                    "block": block_index,
                    "target_scenarios": list(block),
                    "target_rows": int(len(block_rows)),
                    "target_sample_ids_sha256": canonical_hash(ids[block_rows].tolist()),
                    "prior_F_id": canonical_hash(list(ancestor)),
                    "prior_F_scenarios": list(ancestor),
                }
            )
        outer_plans.append(
            {
                "outer_fold": fold,
                "outer_held_scenarios": sorted(held),
                "training_scenarios": sorted(training),
                "outer_prior_F_id": canonical_hash(list(outer_population)),
                "blocks": block_items,
            }
        )
    population_receipts, all_base = [], set()
    for population in sorted(populations):
        receipt, base = _population_receipt(population, y, groups, ids)
        population_receipts.append(receipt)
        all_base.update(base)
    actual_estimators = sum(
        4 * (4 * min(3, len(set(groups[list(rows)]))) + 1) for rows in all_base
    )
    result = {
        "status": "NO_FIT_SET_AND_HISTORICAL_SPLIT_AUDIT_COMPLETE_EXECUTOR_UNCERTIFIED",
        "policy_prefix": POLICY_PREFIX,
        "topology": "fixed_recipe_three_blocks_no_additional_policy_selection",
        "rows": int(len(y)),
        "scenario_count": len(all_scenarios),
        "outer_plans": outer_plans,
        "F_populations": population_receipts,
        "counts": {
            "outer_context_F_calls": 20,
            "unique_F_populations": len(populations),
            "F_scenario_count_histogram": dict(sorted(Counter(map(len, populations)).items())),
            "minimum_explicit_F_scenarios": min(map(len, populations)),
            "unique_base_training_populations": len(all_base),
            "M4_A3_neural_fits_before_reuse": 30 * len(populations),
            "base_estimators_before_reuse": actual_estimators,
            "correction_heads_scientific_plus_calibration": 35,
            "correction_heads_with_equal_acquisition_control": 40,
            "reusable_task_fits_certified": 0,
        },
        "checks": {
            "outer_scenario_exclusion": "passed",
            "three_block_disjointness": "passed",
            "historical_three_level_split_and_class_coverage": "passed",
            "primary_arm_and_policy_fixed_before_outer_scores": True,
            "post_fit_source_threshold_or_regularizer_selection": False,
        },
        "hard_gates_before_fit": [
            "Implement full historical F(S) without simplifying any nested selection",
            "Bind every transitive feature/protocol/code/checkpoint dependency",
            "Certify reused predictions as out-of-population for the exact F set",
            "Replay each original outer F(T) exactly against retained ARFTR",
            "Time one missing primitive and project every job before launch",
        ],
        "not_certified": [
            "generalized F(S) executor",
            "transitive cache reuse",
            "retained F(T) probability parity",
            "fit runtime or convergence",
            "new source feature utility",
        ],
    }
    expected = {
        "unique_F_populations": 19,
        "unique_base_training_populations": 183,
        "M4_A3_neural_fits_before_reuse": 570,
        "base_estimators_before_reuse": 9084,
    }
    if enforce_canonical_counts and any(
        result["counts"][key] != value for key, value in expected.items()
    ):
        raise RuntimeError(f"Fixed topology differs from the reviewed counts: {result['counts']}")
    return result
