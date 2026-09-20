from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import run_okutama_source_posture as runner

ROOT = Path(__file__).resolve().parents[1]


def test_task_fit_requires_corrected_independently_audited_ancestors() -> None:
    source = (ROOT / "experiments/run_okutama_source_posture.py").read_text(
        encoding="utf-8"
    )
    assert "generalized_arftr_ancestors_v2" in source
    assert "GENERALIZED_ARFTR_ANCESTORS_V2_INDEPENDENT_AUDIT_PASS" in source
    assert "all_prediction_label_counterfactual_invariant" in source
    tree = ast.parse(source)
    fit_fold = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_fold"
    )
    fit_source = ast.get_source_segment(source, fit_fold)
    assert fit_source is not None
    assert 'data["labels"][held]' not in fit_source


def test_cache_auditor_requires_complete_canonical_coverage() -> None:
    source = (
        ROOT / "experiments/audit_okutama_source_posture_cache.py"
    ).read_text(encoding="utf-8")
    assert "sorted(coverage) != list(range(len(canonical_ids)))" in source
    assert "len(replay_matches) != len(reference_by_id)" in source


def _example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray((0, 1, 2, 0, 1, 2), dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)[labels]
    scenarios = np.asarray((0, 0, 1, 1, 2, 2), dtype=np.int64)
    return labels, probabilities, scenarios


def test_scenario_resampling_is_exactly_null_for_identical_predictions() -> None:
    labels, probabilities, scenarios = _example()

    bootstrap = runner._scenario_bootstrap(
        labels,
        probabilities,
        probabilities.copy(),
        scenarios,
        resamples=1000,
        seed=7,
    )
    exact = runner._exact_scenario_swap(
        labels, probabilities, probabilities.copy(), scenarios
    )

    assert bootstrap["delta_mean"] == 0
    assert bootstrap["delta_95_interval"] == [0.0, 0.0]
    assert exact == {
        "assignments": 8,
        "observed_delta": 0.0,
        "one_sided_pvalue": 1.0,
        "two_sided_pvalue": 1.0,
    }


def test_immutable_npz_resumes_with_string_and_nan_arrays(tmp_path: Path) -> None:
    path = tmp_path / "artifact.npz"
    arrays = {
        "names": np.asarray(("alpha", "beta")),
        "values": np.asarray((1.0, np.nan)),
    }

    runner._immutable_npz(path, **arrays)
    runner._immutable_npz(path, **arrays)

    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved["names"], arrays["names"])
        assert np.array_equal(saved["values"], arrays["values"], equal_nan=True)
