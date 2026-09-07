from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from replay_okutama_cptr_interventions import (
    BOOTSTRAP_SEED,
    RUN_STATUS,
    _exact_group_swap_pvalue,
    _holm_adjust,
    _slice_report,
    canonicalize_f3_persistence,
    compute_interventions,
    decision_statistics,
    recover_zero_epoch_q,
    validate_existing_checkpoint_output,
    validate_zero_epoch_state,
)

from experiments.audit_cptr_initialization_contract import synthetic_kwargs, synthetic_model
from hac.vcoco_v3_neural import decode_factorized_logits


def test_zero_epoch_q_recovery_selects_each_discrete_valid_fraction():
    static_posture = torch.tensor([[2.0, -0.5]]).repeat(9, 1)
    static_motion = torch.tensor([[1.0, -1.0]]).repeat(9, 1)
    legacy_posture = torch.tensor([[-0.2, 1.3]]).repeat(9, 1)
    legacy_motion = torch.tensor([[-0.7, 0.8]]).repeat(9, 1)
    q = torch.arange(9, dtype=torch.float32) / 8
    gate = torch.sigmoid(torch.tensor(5.0))
    posture = static_posture + gate * q[:, None] * (legacy_posture - static_posture)
    motion = static_motion + gate * q[:, None] * (legacy_motion - static_motion)
    historical = decode_factorized_logits(posture, motion)

    recovered, error = recover_zero_epoch_q(
        static_posture,
        static_motion,
        legacy_posture,
        legacy_motion,
        historical,
    )

    assert torch.equal(recovered, q)
    assert torch.count_nonzero(error) == 0


def test_interventions_preserve_new_residual_and_f3_returns_teacher_directly():
    model = synthetic_model()
    head = model.residual_heads["centre_short"]
    with torch.no_grad():
        head.posture.bias.copy_(torch.tensor([0.2, -0.1]))
        head.motion.bias.copy_(torch.tensor([-0.3, 0.4]))
    kwargs = synthetic_kwargs(4)

    with torch.inference_mode():
        result = compute_interventions(model, kwargs)
        static = model.static_fallback(kwargs["static_features"])
        teacher = model.legacy_temporal(
            kwargs["short_features"], torch.ones_like(kwargs["short_valid_mask"])
        )
        expected_f2_posture = teacher.posture_logits + result.learned_posture_delta
        expected_f2_motion = teacher.motion_logits + result.learned_motion_delta

    assert torch.equal(result.posture_logits["f2"], expected_f2_posture)
    assert torch.equal(result.motion_logits["f2"], expected_f2_motion)
    assert torch.equal(result.probabilities["f3"], teacher.probabilities)
    assert torch.equal(result.posture_logits["f3"], teacher.posture_logits)
    assert torch.equal(result.motion_logits["f3"], teacher.motion_logits)
    assert torch.equal(result.q, torch.tensor([0.5]))
    assert not torch.equal(result.probabilities["original"], result.probabilities["f2"])
    assert not torch.equal(result.probabilities["f1"], static.probabilities)


def test_persisted_f3_has_one_authoritative_probability_source():
    retained = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32)
    direct = np.asarray([[0.4, 0.4, 0.2]], dtype=np.float32)
    arrays = {
        "f3_probabilities": direct.copy(),
        "f3_posture_logits": np.asarray([[1.0, -1.0]], dtype=np.float32),
        "f3_motion_logits": np.asarray([[0.5, -0.5]], dtype=np.float32),
    }

    canonicalize_f3_persistence(arrays, retained)

    assert np.array_equal(arrays["f3_probabilities"], retained)
    assert np.array_equal(arrays["f3_current_direct_probabilities"], direct)
    assert "f3_posture_logits" not in arrays
    assert "f3_motion_logits" not in arrays
    assert "f3_current_direct_posture_logits" in arrays
    assert "f3_current_direct_motion_logits" in arrays


def test_zero_epoch_validator_rejects_a_nonzero_new_output():
    model = synthetic_model()
    validate_zero_epoch_state(model)
    with torch.no_grad():
        model.residual_heads["centre_short"].posture.bias[0] = 0.01
    try:
        validate_zero_epoch_state(model)
    except RuntimeError as error:
        assert "residual output is nonzero" in str(error)
    else:
        raise AssertionError("A nonzero residual output passed the zero-epoch contract")


def test_exact_group_swaps_and_holm_are_deterministic():
    labels = np.tile(np.arange(3), 3)
    groups = np.repeat(np.asarray(["a", "b", "c"]), 3)
    reference = np.full((9, 3), 0.05, dtype=np.float64)
    reference[np.arange(9), (labels + 1) % 3] = 0.90
    candidate = np.full((9, 3), 0.05, dtype=np.float64)
    candidate[np.arange(9), labels] = 0.90
    assignments = np.asarray(
        [[bool(value & (1 << bit)) for bit in range(3)] for value in range(8)]
    )

    result = _exact_group_swap_pvalue(
        labels,
        reference,
        candidate,
        groups,
        np.asarray(["a", "b", "c"]),
        assignments,
    )

    assert result["assignments"] == 8
    assert result["observed_macro_f1_delta"] == pytest.approx(1.0)
    assert result["one_sided_pvalue"] == pytest.approx(1 / 8)
    assert _holm_adjust({"f1": 0.01, "f2": 0.04}) == {"f1": 0.02, "f2": 0.04}


def test_slice_report_includes_brier_confusion_and_rescue_harm():
    labels = np.asarray([0, 1, 2])
    original = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    repaired = np.eye(3, dtype=np.float64)
    probabilities = {
        "original": original,
        "f1": repaired,
        "f2": repaired,
        "f3": repaired,
    }

    report = _slice_report(labels, probabilities, np.ones(3, dtype=bool))

    assert report["metrics"]["f2"]["brier"] == 0.0
    assert report["metrics"]["f2"]["confusion"] == np.eye(3, dtype=int).tolist()
    assert report["f2_error_flow_from_original"]["rescued"] == 2
    assert report["f2_error_flow_from_original"]["harmed"] == 0


def test_decision_statistics_persists_shared_bootstrap_and_exact_swaps(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("replay_okutama_cptr_interventions.BOOTSTRAP_RESAMPLES", 100)
    labels = np.tile(np.asarray([0, 1, 2]), 22)
    groups = np.repeat(np.asarray([f"scenario-{value:02d}" for value in range(11)]), 6)
    original = np.full((len(labels), 3), 0.05, dtype=np.float64)
    original[np.arange(len(labels)), (labels + 1) % 3] = 0.90
    repaired = np.full((len(labels), 3), 0.05, dtype=np.float64)
    repaired[np.arange(len(labels)), labels] = 0.90
    probabilities = {
        "original": original,
        "f1": repaired,
        "f2": repaired,
        "f3": repaired,
    }

    summary, path = decision_statistics(
        labels=labels,
        groups=groups,
        occluded=np.ones(len(labels), dtype=bool),
        transition=np.ones(len(labels), dtype=bool),
        probabilities=probabilities,
        output_dir=tmp_path,
        stage_name="r1a",
    )
    resumed_summary, resumed_path = decision_statistics(
        labels=labels,
        groups=groups,
        occluded=np.ones(len(labels), dtype=bool),
        transition=np.ones(len(labels), dtype=bool),
        probabilities=probabilities,
        output_dir=tmp_path,
        stage_name="r1a",
    )

    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["group_draw_indices"].shape == (100, 11)
        assert arrays["exact_swap_assignments"].shape == (2**11, 11)
        assert int(arrays["bootstrap_seed"]) == BOOTSTRAP_SEED
        assert arrays["group_order"].tolist() == sorted(np.unique(groups).tolist())
    assert summary["mechanistic_gate"]["passed"] is True
    assert len(summary["scenarios"]) == 11
    assert resumed_path == path
    assert resumed_summary == summary


def test_existing_checkpoint_resume_is_hash_and_source_bound(tmp_path):
    output = tmp_path / "replay.npz"
    np.savez_compressed(output, value=np.asarray([1.0]))
    sources = {"runner": "a" * 64}
    entry = {
        "fold": 0,
        "seed": 43,
        "fixed_epochs": 0,
        "files": {"candidate_checkpoint": {"sha256": "b" * 64}},
    }
    summary = {
        "status": RUN_STATUS,
        "fold": 0,
        "seed": 43,
        "fixed_epochs": 0,
        "source_sha256": sources,
        "frozen_input_sha256": {"candidate_checkpoint": "b" * 64},
        "artifact_sha256": {output.name: __import__(
            "replay_okutama_cptr_interventions"
        ).sha256_file(output)},
        "reproduction_gate_passed": True,
        "f3_statistical_identity_with_retained_teacher": True,
    }
    output.with_name("summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert validate_existing_checkpoint_output(
        output_path=output, entry=entry, replay_source_sha256=sources
    )["seed"] == 43
    with pytest.raises(RuntimeError, match="failed closed"):
        validate_existing_checkpoint_output(
            output_path=output,
            entry=entry,
            replay_source_sha256={"runner": "c" * 64},
        )
