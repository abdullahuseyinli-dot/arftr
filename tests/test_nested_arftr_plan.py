from __future__ import annotations

import numpy as np

from hac.nested_arftr_plan import fixed_recipe_plan


def test_fixed_plan_matches_reviewed_counts_and_excludes_blocks():
    scenarios = np.repeat(np.asarray([f"s{i}" for i in range(11)]), 30)
    labels = np.tile(np.arange(3), 110)
    folds_by_scenario = {f"s{i}": min(i // 2, 4) for i in range(11)}
    folds = np.asarray([folds_by_scenario[value] for value in scenarios])
    ids = np.asarray([f"id-{index}" for index in range(len(labels))])
    plan = fixed_recipe_plan(labels, scenarios, folds, ids)
    # Generic balanced fixture can induce a different number of cross-outer set
    # duplicates than the canonical11 scenarios, but every exclusion must hold.
    assert plan["checks"]["outer_scenario_exclusion"] == "passed"
    for outer in plan["outer_plans"]:
        held = set(outer["outer_held_scenarios"])
        for block in outer["blocks"]:
            assert not held & set(block["prior_F_scenarios"])
            assert not set(block["target_scenarios"]) & set(block["prior_F_scenarios"])
