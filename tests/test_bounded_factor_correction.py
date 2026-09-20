import numpy as np
import torch

from hac.bounded_factor_correction import (
    BoundedFactorCorrection,
    apply_numpy_correction,
    factor_logits,
    probability_factor_scores,
)


def test_factor_round_trip() -> None:
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.15, 0.05]])
    posture, motion = probability_factor_scores(probabilities)
    recovered = torch.softmax(factor_logits(posture, motion), dim=1)
    torch.testing.assert_close(recovered, probabilities)


def test_zero_initialized_separate_and_shared_models_preserve_anchor() -> None:
    features = torch.randn(7, 8)
    anchor = torch.softmax(torch.randn(7, 3), dim=1)
    for shared in (False, True):
        model = BoundedFactorCorrection(input_dim=8, width=4, shared_output=shared)
        output = model(features, anchor)
        torch.testing.assert_close(torch.softmax(output["logits"], 1), anchor)
        torch.testing.assert_close(output["correction"], torch.zeros(7, 2))


def test_bounded_model_enforces_limits_and_numpy_zero_is_exact() -> None:
    model = BoundedFactorCorrection(
        input_dim=4, width=3, posture_bound=0.25, motion_bound=0.5, bounded=True
    )
    with torch.no_grad():
        assert model.correction_head is not None
        model.correction_head.bias[:] = torch.tensor([100.0, -100.0])
    _, correction = model.heads(torch.randn(5, 4))
    assert torch.all(correction[:, 0].abs() <= 0.25)
    assert torch.all(correction[:, 1].abs() <= 0.5)
    anchor = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
    assert np.array_equal(apply_numpy_correction(anchor, np.zeros((1, 2))), anchor)


def test_unbounded_control_can_exceed_nominal_scale() -> None:
    model = BoundedFactorCorrection(input_dim=4, width=3, bounded=False)
    with torch.no_grad():
        assert model.correction_head is not None
        model.correction_head.bias[:] = 4.0
    _, correction = model.heads(torch.randn(2, 4))
    assert torch.all(correction > model.limits)
