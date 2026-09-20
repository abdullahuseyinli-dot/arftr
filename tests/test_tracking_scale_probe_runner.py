from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def runner():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("scale_runner_test", root / "experiments/run_okutama_tracking_scale_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_with_scores(full=.4, low=.4, high=.9):
    module = runner()
    rows = []
    for i in range(16):
        rows.append({"primary_eligible": i not in (3, 10), "scenario": str(i % 2),
                     "arms": {a: {"actor_survival_all_nine": v, "actor_joint_forward_reverse_survival": v,
                                  "cycle_median_normalized": .01} for a, v in zip(module.ARMS, (full, low, high), strict=True)}})
    return rows


def test_seed_deficient_centers_never_earn_primary_credit():
    result = runner().assess(rows_with_scores())
    assert result["primary_pass_counts_of16"]["crop_native_detail"] == 14
    assert result["mechanism_supported"]
    assert not result["expanded128_authorized"] and not result["activity_training_authorized"]
    assert not result["original_smoke_reclassified"]


def test_context_only_gain_is_not_detail_mechanism_success():
    result = runner().assess(rows_with_scores(full=.4, low=.9, high=.9))
    assert not result["mechanism_supported"]
    assert not result["checks"]["at_least3_more_native_passes_than_low"]


def test_one_bad_scenario_blocks_mechanism_support():
    rows = rows_with_scores()
    for row in rows:
        if row["scenario"] == "1":
            row["arms"]["crop_low_detail"]["actor_joint_forward_reverse_survival"] = .95
    result = runner().assess(rows)
    assert not result["checks"]["both_scenarios_positive_native_minus_low"]


def test_partial_population_and_missing_cycles_fail():
    module = runner()
    with pytest.raises(RuntimeError, match="partial"):
        module.assess(rows_with_scores()[:15])
    assert not module.sample_pass({"actor_survival_all_nine": 1., "actor_joint_forward_reverse_survival": 0., "cycle_median_normalized": None})


def test_bootstrap_fixed_seed_and_zero_effect():
    module = runner()
    assert module.center_bootstrap(np.zeros(14)) == [0., 0.]
    assert module.center_bootstrap(np.ones(14)) == [1., 1.]
    assert module.center_bootstrap(np.arange(14)) == module.center_bootstrap(np.arange(14))


def test_wrong_eligibility_count_is_rejected():
    rows = rows_with_scores()
    rows[3]["primary_eligible"] = True
    with pytest.raises(RuntimeError, match="exactly14"):
        runner().assess(rows)


def test_protocol_roundtrip_keeps_predeclared_synthetic_order():
    module = runner()
    protocol = module.read(module.PROTOCOL)
    reloaded = json.loads(json.dumps(protocol, sort_keys=True))
    assert reloaded["synthetic_controls"]["sizes"] == ["typical", "tiny"]


@pytest.mark.parametrize("crop_mode", [False, True])
def test_reverse_return_on_unsampled_pixel_border_is_missing(crop_mode):
    module = runner()
    crop = module.FixedCrop(3, 4, 10, 10) if crop_mode else None
    native_shape = (20, 20) if crop_mode else (10, 10)
    seed = np.asarray([[5., 6.]]) if crop_mode else np.asarray([[2., 2.]])

    class BorderReturnBackend:
        def predict(self, video, queries, reverse=False):
            positions = np.broadcast_to(queries[None, :, 1:], (3, len(queries), 2)).copy()
            if reverse:
                positions[1, :, 0] = 9.75  # old <size adapter accepts; beyond last sampled pixel center
            return positions, np.ones(positions.shape[:2], bool)

    item, local = module.track(BorderReturnBackend(), None, seed, np.asarray([-1., 0., 1.]),
        ("actor-000",), 1, (0, 1, 2), np.repeat(np.eye(3)[None], 3, axis=0), native_shape, crop)
    assert item.sequence.valid.all()
    assert not item.cycle_valid.any() and not local.cycle_valid.any()
    assert np.isnan(item.cycle_error).all()
