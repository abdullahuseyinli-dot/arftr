import pytest
import torch

from hac.correspondence_field_reader import (
    ARMS,
    CONTEXT_DIM,
    MAX_POINTS,
    PAIR_COUNT,
    SUMMARY_DIM,
    CorrespondenceFieldReader,
    c2_parameter_target,
)


def payload(rows=2):
    generator = torch.Generator().manual_seed(516)
    points = torch.randn(rows, PAIR_COUNT, MAX_POINTS, 10, generator=generator)
    valid = torch.rand(rows, PAIR_COUNT, MAX_POINTS, generator=generator) > 0.2
    pairs = torch.randn(rows, PAIR_COUNT, 5, generator=generator)
    pairs[..., 0] = 0.1333333
    pairs[..., 1] = valid.sum(-1)
    pairs[..., 2:4] = pairs[..., 2:4].abs()
    pairs[..., 4] = 1
    return {
        "context": torch.randn(rows, CONTEXT_DIM, generator=generator),
        "points": points,
        "point_valid": valid,
        "pair_valid": torch.ones(rows, PAIR_COUNT, dtype=torch.bool),
        "pair_features": pairs,
        "midpoint_times": torch.linspace(-0.9333, 0.9333, PAIR_COUNT).expand(rows, -1),
        "summaries": torch.randn(rows, SUMMARY_DIM, generator=generator),
    }


def reader(arm):
    torch.manual_seed(147)
    return CorrespondenceFieldReader(arm, dropout=0).eval()


def cloned(values):
    return {name: value.clone() for name, value in values.items()}


@pytest.mark.parametrize("arm", ARMS)
def test_binary_interface_and_all_parameters_are_task_active(arm):
    model = reader(arm)
    values = payload()
    values["context"].requires_grad_()
    output = model(**values)
    assert output["motion_logits"].shape == (2,)
    assert output["observation_available"].tolist() == [True, True]
    probabilities = model.conditional_probabilities(**values)
    assert probabilities.shape == (2, 2)
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.nn.functional.binary_cross_entropy_with_logits(
        output["motion_logits"], torch.tensor([0.0, 1.0])
    ).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    # Both video streams and the appearance stream reach the learned decision.
    gradients = values["context"].grad.reshape(2, 3, 768).abs().sum((0, 2))
    assert torch.all(gradients > 0)
    assert abs(model.trainable_parameters / c2_parameter_target() - 1) <= 0.1
    assert model.capacity_report["width"] == 64


@pytest.mark.parametrize("arm", ("c1", "c2"))
def test_point_permutations_are_invariant_with_masked_points(arm):
    values = payload()
    model = reader(arm)
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    permutation = torch.rand(
        values["point_valid"].shape, generator=torch.Generator().manual_seed(84)
    ).argsort(-1)
    changed["points"] = changed["points"].gather(
        2, permutation[..., None].expand_as(changed["points"])
    )
    changed["point_valid"] = changed["point_valid"].gather(2, permutation)
    after = model(**changed)["motion_logits"]
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("arm", ("c1", "c2"))
def test_spatial_assignment_is_available_to_field_readers(arm):
    values = payload()
    values["point_valid"][:] = True
    model = reader(arm)
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    # Move positions independently of vectors. This is not a point-order change.
    changed["points"][..., :2] = changed["points"].flip(2)[..., :2]
    after = model(**changed)["motion_logits"]
    assert not torch.allclose(before, after, atol=1e-8, rtol=1e-7)


def test_unordered_reader_ignores_times_and_pair_order():
    model = reader("c1")
    values = payload()
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    permutation = torch.arange(PAIR_COUNT - 1, -1, -1)
    for name in ("points", "point_valid", "pair_valid", "pair_features"):
        changed[name] = changed[name][:, permutation]
    # Even invalid timestamps cannot influence C1's computation.
    changed["midpoint_times"][:] = torch.nan
    changed["summaries"][:] = torch.nan
    assert torch.allclose(before, model(**changed)["motion_logits"], atol=1e-6, rtol=1e-6)


def test_timed_reader_uses_field_time_associations_not_slot_indices():
    model = reader("c2")
    values = payload()
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    permutation = torch.arange(PAIR_COUNT - 1, -1, -1)
    for name in ("points", "point_valid", "pair_valid", "pair_features", "midpoint_times"):
        changed[name] = changed[name][:, permutation]
    together = model(**changed)["motion_logits"]
    assert torch.allclose(before, together, atol=1e-6, rtol=1e-6)
    changed["midpoint_times"] = values["midpoint_times"]
    reassigned = model(**changed)["motion_logits"]
    assert not torch.allclose(before, reassigned, atol=1e-8, rtol=1e-7)


def test_center_query_receives_gradients_from_every_pair_including_endpoints():
    values = payload(rows=1)
    values["points"].requires_grad_()
    model = reader("c2")
    model(**values)["motion_logits"].sum().backward()
    pair_gradient = values["points"].grad.abs().sum((0, 2, 3))
    assert pair_gradient.shape == (15,)
    assert torch.all(pair_gradient > 1e-9), pair_gradient


@pytest.mark.parametrize("arm", ARMS)
def test_all_missing_observations_are_finite_and_payload_independent(arm):
    values = payload()
    values["point_valid"][:] = False
    model = reader(arm)
    before = model(**values)
    changed = cloned(values)
    for name in ("points", "pair_features", "midpoint_times", "summaries"):
        changed[name][:] = torch.nan
    after = model(**changed)
    assert not after["observation_available"].any()
    assert torch.isfinite(after["motion_probability"]).all()
    assert torch.equal(before["motion_logits"], after["motion_logits"])


@pytest.mark.parametrize("arm", ("c1", "c2"))
def test_invalid_pair_payload_cannot_affect_prediction_or_gradients(arm):
    values = payload()
    values["pair_valid"][:, 0] = False
    model = reader(arm)
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    for name in ("points", "pair_features", "midpoint_times"):
        changed[name][:, 0] = torch.nan
    after = model(**changed)["motion_logits"]
    assert torch.equal(before, after)
    values["points"].requires_grad_()
    model(**values)["motion_logits"].sum().backward()
    assert torch.equal(values["points"].grad[:, 0], torch.zeros_like(values["points"].grad[:, 0]))


def test_c0_uses_summary_values_and_ignores_raw_fields():
    values = payload()
    model = reader("c0")
    before = model(**values)["motion_logits"]
    changed = cloned(values)
    for name in ("points", "pair_features", "midpoint_times"):
        changed[name][:] = torch.nan
    assert torch.equal(before, model(**changed)["motion_logits"])
    changed["summaries"] += 0.75
    assert not torch.allclose(before, model(**changed)["motion_logits"])


def test_invalid_available_observations_and_wrong_support_are_rejected():
    values = payload()
    model = reader("c2")
    values["points"][0, 0, 0] = torch.nan
    values["point_valid"][0, 0, 0] = True
    with pytest.raises(ValueError, match="nonfinite"):
        model(**values)
    values = payload()
    values["pair_features"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="elapsed"):
        model(**values)
    values = payload()
    values["point_valid"] = values["point_valid"][:, :-1]
    with pytest.raises(ValueError, match="15 pairs"):
        model(**values)
    values = payload()
    values["context"] = values["context"][:, :768]
    with pytest.raises(ValueError, match="context"):
        model(**values)


def test_declared_width_capacity_and_summary_contract_are_fixed():
    counts = {arm: reader(arm).trainable_parameters for arm in ARMS}
    assert counts == {"c0": 250177, "c1": 253121, "c2": 245697}
    assert len(reader("c2").attention_blocks) == 2
    values = payload()
    values["summaries"] = torch.zeros(2, 40)
    with pytest.raises(ValueError, match="summaries"):
        reader("c0")(**values)
    with pytest.raises(ValueError, match="arm"):
        CorrespondenceFieldReader("unknown")
