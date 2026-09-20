from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "r0_exact_m4",
    "r1_p6_restoration_only",
    "r2_a3_motion_only",
    "r3_temporal_only",
    "r4_residual_no_temporal",
    "r5_arftr_full",
    "r6_shuffled_neighbor_control",
)


def test_runner_has_frozen_arms_grid_and_outer_label_embargo():
    assert ARMS == (
        "r0_exact_m4",
        "r1_p6_restoration_only",
        "r2_a3_motion_only",
        "r3_temporal_only",
        "r4_residual_no_temporal",
        "r5_arftr_full",
        "r6_shuffled_neighbor_control",
    )
    source = (ROOT / "experiments/run_okutama_arftr.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fit_fold = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_fold"
    )
    fit_source = ast.get_source_segment(source, fit_fold)
    assert fit_source is not None
    assert "data[\"labels\"][held]" not in fit_source
    assert '"outer_held_labels_read": 0' in fit_source
    assert "ARFTR_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED" in fit_source
    assert "probability_metrics" not in fit_source


def test_parameter_grid_is_the_frozen_300_candidates():
    protocol = json.loads(
        (ROOT / "experiments/okutama_arftr_protocol.json").read_text(encoding="utf-8")
    )
    selection = protocol["selection"]
    grid = tuple(
        itertools.product(
            selection["posture_restoration"],
            selection["motion_restoration"],
            selection["a3_motion_residual"],
            selection["temporal_strength"],
        )
    )
    assert len(grid) == 300
    assert grid[0] == (0.0, 0.1, 0.0, 0.1)
    assert grid[-1] == (0.4, 0.5, 0.3, 0.3)
