from __future__ import annotations

import ast
from pathlib import Path

from experiments.run_okutama_class_diagonal_residual import ARMS, exact_scenario_swap

ROOT = Path(__file__).resolve().parents[1]


def test_runner_has_one_fixed_primary_and_outer_metric_embargo():
    assert ARMS == ("c0_exact_m4", "c1_class_diagonal_protected_residual")
    source = (ROOT / "experiments/run_okutama_class_diagonal_residual.py").read_text()
    tree = ast.parse(source)
    assert "CLASS_DIAGONAL_FOLD_COMPLETE_OUTER_METRICS_EMBARGOED" in source
    fit_fold = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_fold")
    fit_source = ast.get_source_segment(source, fit_fold)
    assert fit_source is not None
    assert "probability_metrics" not in fit_source
    assert "data[\"labels\"][held]" not in fit_source
    assert '"outer_held_labels_read": 0' in fit_source


def test_exact_scenario_swap_enumerates_every_assignment():
    import numpy as np

    labels = np.array([0, 0, 1, 1])
    anchor = np.array([[0.6, 0.4, 0], [0.4, 0.6, 0], [0.6, 0.4, 0], [0.4, 0.6, 0]])
    candidate = np.array([[0.7, 0.3, 0], [0.7, 0.3, 0], [0.3, 0.7, 0], [0.3, 0.7, 0]])
    result = exact_scenario_swap(labels, candidate, anchor, np.array([10, 10, 20, 20]))
    assert result["scenario_groups"] == 2
    assert result["assignments"] == 4
    assert result["enumeration_complete"] is True
    assert result["random_sampling"] is False
