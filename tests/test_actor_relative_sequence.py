import numpy as np
import pytest
import torch

from hac.actor_relative_sequence import ARMS, ActorRelativeStateSequence, sequence_loss


def payload(rows=3, slots=6):
    valid = torch.ones(rows, slots, dtype=torch.bool)
    return {
        "features": torch.randn(rows, slots, 8),
        "timestamps": torch.linspace(-1, 1, slots).expand(rows, -1),
        "streams": (torch.arange(slots)[None] >= slots // 2).expand(rows, -1).long(),
        "valid": valid,
        "center_indices": torch.full((rows,), slots // 2, dtype=torch.long),
        "labels": torch.arange(slots)[None].expand(rows, -1).remainder(3).long(),
        "physical_weights": torch.ones(rows, slots),
    }


@pytest.mark.parametrize("arm", ARMS)
def test_sequence_arm_shapes_and_finite_loss(arm):
    model = ActorRelativeStateSequence(arm, input_dim=8, width=8, temporal_blocks=2, parameter_limit=10_000)
    values = payload()
    output = model(values["features"], values["timestamps"], values["streams"], values["valid"], values["center_indices"])
    assert output["frame_logits"].shape == (3, 6, 3)
    assert output["center_logits"].shape == (3, 3)
    loss, parts = sequence_loss(model, **values, class_weight=torch.ones(3), relation_loss_weight=0.15)
    assert torch.isfinite(loss)
    assert set(parts) == {"frame", "center", "relation"}
    assert ("relation_logits" in output) == (arm == ARMS[-1])


def test_invalid_center_or_weight_is_rejected():
    model = ActorRelativeStateSequence(ARMS[-1], input_dim=8, width=8, temporal_blocks=2, parameter_limit=10_000)
    values = payload()
    values["valid"][0, 3] = False
    with pytest.raises(ValueError, match="center"):
        model(values["features"], values["timestamps"], values["streams"], values["valid"], values["center_indices"])
    values = payload()
    values["physical_weights"][0, 0] = 0
    with pytest.raises(ValueError, match="loss"):
        sequence_loss(model, **values, class_weight=torch.ones(3), relation_loss_weight=0.15)


def test_independent_head_ignores_time_and_other_frames():
    model = ActorRelativeStateSequence(ARMS[0], input_dim=8, width=8, temporal_blocks=2, parameter_limit=10_000).eval()
    values = payload(rows=1)
    first = model(values["features"], values["timestamps"], values["streams"], values["valid"], values["center_indices"])["center_logits"]
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in values.items()}
    changed["features"][0, 0] += 100
    changed["timestamps"] += 100
    second = model(changed["features"], changed["timestamps"], changed["streams"], changed["valid"], changed["center_indices"])["center_logits"]
    assert torch.allclose(first, second)


def test_sequence_inputs_have_no_nan_after_masking():
    model = ActorRelativeStateSequence(ARMS[2], input_dim=8, width=8, temporal_blocks=2, parameter_limit=10_000)
    values = payload(rows=1)
    values["valid"][0, 0] = False
    values["features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Malformed"):
        model(values["features"], values["timestamps"], values["streams"], values["valid"], values["center_indices"])
