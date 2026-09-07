import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.run_vcoco_continuation_v0 import (
    FAMILY_ORDER,
    PROTECTED_COUNTERS,
    build_fold_map,
    enumerate_v0_candidates,
    fit_family_outer_fold,
    load_shared_randomization,
    paired_group_swap_test,
    paired_nll_bootstrap,
    prepare_shared_randomization,
    relative_safe_path,
    require_converged_svm_fit,
    require_zero_protected_access,
    validate_fold_map,
    validate_protocol,
)


def continuation_protocol() -> dict:
    return json.loads(
        Path("experiments/vcoco_continuation_protocol.json").read_text(encoding="utf-8")
    )


def synthetic_development_rows() -> pd.DataFrame:
    unique_groups = 4_123
    people = 6_640
    group_index = np.concatenate(
        [np.arange(unique_groups), np.arange(people - unique_groups)]
    )
    labels = np.asarray(("sitting", "standing", "walking_running"))[group_index % 3]
    return pd.DataFrame(
        {
            "person_id": [f"person-{index:05d}" for index in range(people)],
            "image_id": [f"image-{index:05d}" for index in group_index],
            "split": ["train"] * 3_090 + ["val"] * 3_550,
            "label_3": labels,
            "bbox_area_fraction": np.full(people, 0.1),
            "bbox_aspect_ratio": np.full(people, 0.5),
            "bbox_center_x_fraction": np.full(people, 0.5),
            "bbox_center_y_fraction": np.full(people, 0.5),
            "person_pixel_height": np.full(people, 100.0),
        }
    )


@pytest.fixture(scope="module")
def protocol_and_folds() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    protocol = continuation_protocol()
    rows = synthetic_development_rows()
    return protocol, rows, build_fold_map(rows, protocol)


def test_v0_protocol_has_four_matched_eight_candidate_families():
    protocol = continuation_protocol()
    validate_protocol(protocol)
    assert tuple(protocol["V0"]["families"]) == FAMILY_ORDER
    candidates = {
        family: enumerate_v0_candidates(protocol, family) for family in FAMILY_ORDER
    }
    assert {family: len(values) for family, values in candidates.items()} == {
        family: 8 for family in FAMILY_ORDER
    }
    assert all(len({item.candidate_id for item in values}) == 8 for values in candidates.values())
    assert protocol["V0"]["shared_geometry_features"]["dimensions"] == 6
    changed = json.loads(json.dumps(protocol))
    changed["V0"]["shared_geometry_features"]["dimensions"] = 5
    with pytest.raises(RuntimeError, match="geometry-feature contract"):
        validate_protocol(changed)
    assert protocol["V0"]["linear_svm_grid"]["maximum_iterations"] == 2_000
    assert protocol["V0"]["linear_svm_grid"]["require_iteration_limit_not_reached"] is True


def test_v0_fails_closed_on_iteration_limited_svm_fit():
    require_converged_svm_fit(
        SimpleNamespace(optimization_={"iteration_limit_reached": False}),
        context="synthetic-converged",
    )
    with pytest.raises(RuntimeError, match="iteration limit"):
        require_converged_svm_fit(
            SimpleNamespace(optimization_={"iteration_limit_reached": True}),
            context="synthetic-limited",
        )


def test_fold_map_is_deterministic_grouped_and_marks_outer_held(
    protocol_and_folds: tuple[dict, pd.DataFrame, pd.DataFrame],
):
    protocol, rows, first = protocol_and_folds
    second = build_fold_map(rows, protocol)
    pd.testing.assert_frame_equal(first, second)
    assert first["row_index"].tolist() == list(range(len(rows)))
    assert first.groupby("image_id")["outer_fold"].nunique().max() == 1
    for outer_fold in range(5):
        held = first["outer_fold"] == outer_fold
        assert (first.loc[held, f"inner_fold_o{outer_fold}"] == -1).all()
        assert (first.loc[held, f"stack_fold_o{outer_fold}"] == -1).all()
        train = ~held
        assert set(first.loc[train, f"inner_fold_o{outer_fold}"]) == {0, 1, 2}
        assert set(first.loc[train, f"stack_fold_o{outer_fold}"]) == {0, 1, 2}
        assert (
            first.loc[train].groupby("image_id")[f"inner_fold_o{outer_fold}"].nunique().max()
            == 1
        )
        assert (
            first.loc[train].groupby("image_id")[f"stack_fold_o{outer_fold}"].nunique().max()
            == 1
        )
        for inner_fold in range(3):
            field = f"selection_stack_fold_o{outer_fold}_i{inner_fold}"
            eligible = train & (first[f"inner_fold_o{outer_fold}"] != inner_fold)
            assert (first.loc[~eligible, field] == -1).all()
            assert set(first.loc[eligible, field]) == {0, 1, 2}
            assert first.loc[eligible].groupby("image_id")[field].nunique().max() == 1


def test_fold_map_validation_fails_closed_on_assignment_change(
    protocol_and_folds: tuple[dict, pd.DataFrame, pd.DataFrame],
):
    protocol, rows, fold_map = protocol_and_folds
    changed = fold_map.copy()
    held_index = changed.index[changed["outer_fold"] == 0][0]
    changed.loc[held_index, "inner_fold_o0"] = 0
    with pytest.raises(RuntimeError, match="does not match"):
        validate_fold_map(changed, rows, protocol)


def test_outer_selection_never_passes_outer_held_rows_to_a_fit(
    protocol_and_folds: tuple[dict, pd.DataFrame, pd.DataFrame],
):
    protocol, rows, fold_map = protocol_and_folds
    row_ids = np.arange(len(rows), dtype=np.float32)[:, None]
    features = {name: row_ids.copy() for name in protocol["V0"]["families"]["mixed_flat"]["features"]}
    geometry = np.zeros((len(rows), 0), dtype=np.float32)
    labels = np.asarray((0, 1, 2), dtype=np.int64)[
        rows["image_id"].str.removeprefix("image-").astype(int).to_numpy() % 3
    ]
    groups = rows["image_id"].to_numpy(dtype=str)
    outer_fold = 2
    outer_held = set(fold_map.index[fold_map["outer_fold"] == outer_fold])
    calls = []

    def fake_fit(
        candidate,
        declaration,
        fit_protocol,
        train_features,
        target_features,
        train_geometry,
        target_geometry,
        train_labels,
        train_groups,
        stack_assignment,
        *,
        estimator_seed,
    ):
        del candidate, declaration, fit_protocol, train_geometry, target_geometry
        train_ids = train_features["dino_tight"][:, 0].astype(int)
        target_ids = target_features["dino_tight"][:, 0].astype(int)
        assert not set(train_ids).intersection(outer_held)
        assert np.array_equal(train_labels, labels[train_ids])
        assert len(train_groups) == len(train_ids) == len(stack_assignment)
        calls.append(
            (
                set(train_ids),
                set(target_ids),
                estimator_seed,
                tuple(np.asarray(stack_assignment, dtype=int)),
            )
        )
        return np.full((len(target_ids), 3), 1.0 / 3.0)

    held, probabilities, selection = fit_family_outer_fold(
        "mixed_flat",
        0,
        outer_fold,
        protocol,
        features,
        geometry,
        labels,
        groups,
        fold_map,
        fit_predict=fake_fit,
    )
    assert set(held) == outer_held
    assert probabilities.shape == (len(outer_held), 3)
    assert len(selection) == 8
    assert sum(row["selected"] for row in selection) == 1
    assert len(calls) == 8 * 3 + 1
    assert calls[-1][1] == outer_held
    assert all(not target.intersection(outer_held) for _, target, _, _ in calls[:-1])
    for inner_fold in range(3):
        candidate_stacks = [calls[candidate * 3 + inner_fold][3] for candidate in range(8)]
        assert all(stack == candidate_stacks[0] for stack in candidate_stacks)


def test_role_checks_reject_protected_paths_and_nonzero_receipts(tmp_path: Path):
    protected = tmp_path / ".runs" / "polar_v2" / "locked_protocol" / "vcoco_test_clean.csv"
    with pytest.raises(RuntimeError, match="Forbidden"):
        relative_safe_path(tmp_path, protected)
    safe = {"protected_access": {key: 0 for key in PROTECTED_COUNTERS}}
    require_zero_protected_access(safe, name="synthetic")
    unsafe = {"protected_access": {key: 0 for key in PROTECTED_COUNTERS}}
    unsafe["protected_access"]["test_rows_or_arrays_read"] = 1
    with pytest.raises(RuntimeError, match="reports protected"):
        require_zero_protected_access(unsafe, name="synthetic")


def test_paired_statistics_hooks_are_grouped_and_plus_one_corrected():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    probabilities = np.eye(3)[labels] * 0.8 + 0.2 / 3.0
    rng = np.random.default_rng(7)
    bootstrap_indices = rng.integers(0, 3, size=(40, 3), dtype=np.uint16)
    swap_bits = rng.integers(0, 2, size=(50, 3), dtype=np.uint8)
    swap_signs_packbits = np.packbits(swap_bits, axis=1, bitorder="little")
    nll = paired_nll_bootstrap(
        labels,
        probabilities,
        probabilities,
        groups,
        bootstrap_indices=bootstrap_indices,
    )
    swap = paired_group_swap_test(
        labels,
        probabilities,
        probabilities,
        groups,
        swap_signs_packbits=swap_signs_packbits,
    )
    assert nll["clusters"] == 3
    assert nll["point_estimate"] == pytest.approx(0.0)
    assert swap["clusters"] == 3
    assert swap["plus_one_correction"] is True
    assert swap["macro_f1"]["point_estimate"] == pytest.approx(0.0)
    assert swap["macro_f1"]["two_sided_p"] == pytest.approx(1.0)
    assert swap["nll"]["two_sided_p"] == pytest.approx(1.0)


def test_shared_randomization_is_deterministic_persisted_and_reusable(tmp_path: Path):
    protocol = continuation_protocol()
    protocol["data_contract"]["combined_development_images"] = 3
    protocol["statistics"]["bootstrap_resamples"] = 5
    protocol["statistics"]["swap_monte_carlo_draws"] = 7
    groups = np.asarray(["group-c", "group-a", "group-b", "group-a"])
    bootstrap_path = tmp_path / "bootstrap_group_indices.npy"
    swap_path = tmp_path / "swap_signs_packbits.npy"
    receipt_path = tmp_path / "statistics_randomization_receipt.json"
    first = prepare_shared_randomization(
        groups, protocol, bootstrap_path, swap_path, receipt_path
    )
    second = prepare_shared_randomization(
        groups, protocol, bootstrap_path, swap_path, receipt_path
    )
    receipt, bootstrap, packed = load_shared_randomization(
        groups, protocol, bootstrap_path, swap_path, receipt_path
    )
    assert first == second == receipt
    assert receipt["group_ids"] == ["group-a", "group-b", "group-c"]
    assert bootstrap.shape == (5, 3)
    assert bootstrap.dtype == np.uint16
    assert packed.shape == (7, 1)
    assert packed.dtype == np.uint8
