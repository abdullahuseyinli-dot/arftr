import pytest
import torch

from hac.sear_controls import CONTROL_ARMS, DenseCenterControl, UnrestrictedTemplateControl


@pytest.mark.parametrize("arm", CONTROL_ARMS)
def test_controls_shapes_probabilities_and_gradients(arm):
    torch.manual_seed(3)
    model = DenseCenterControl(
        arm,
        input_dim=12,
        video_dim=16,
        width=16,
        layers=1,
        heads=4,
        dropout=0,
        parameter_limit=100_000,
    )
    patches = torch.randn(3, 9, 12)
    video = torch.randn(3, 16)
    valid = torch.ones(3, 9, dtype=torch.bool)
    valid[1, -2:] = False
    output = model(
        patches,
        video,
        valid=valid,
        center_cls=torch.randn(3, 12) if arm == "center_cls" else None,
    )
    assert output["logits"].shape == (3, 3)
    assert torch.allclose(output["probabilities"].sum(-1), torch.ones(3))
    loss = output["logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("arm", ("dense_cnn", "dense_transformer"))
def test_dense_controls_exact_video_fallback(arm):
    model = DenseCenterControl(
        arm,
        input_dim=8,
        video_dim=10,
        width=8,
        layers=1,
        heads=2,
        dropout=0,
        parameter_limit=100_000,
    ).eval()
    patches = torch.full((2, 9, 8), float("nan"))
    valid = torch.zeros(2, 9, dtype=torch.bool)
    video = torch.randn(2, 10)
    output = model(patches, video, valid=valid)
    assert torch.equal(output["local_logits"], torch.zeros_like(output["local_logits"]))
    assert torch.equal(output["logits"], output["video_logits"])


@pytest.mark.parametrize("arm", ("dense_cnn", "dense_transformer"))
def test_zero_matchability_is_exact_video_fallback(arm):
    model = DenseCenterControl(
        arm,
        input_dim=8,
        video_dim=10,
        width=8,
        layers=1,
        heads=2,
        dropout=0,
        parameter_limit=100_000,
    ).eval()
    patches = torch.full((2, 9, 8), float("nan"))
    video = torch.randn(2, 10)
    output = model(patches, video, matchability=torch.zeros(2, 9))
    assert torch.equal(output["local_logits"], torch.zeros_like(output["local_logits"]))
    assert torch.equal(output["logits"], output["video_logits"])


def test_center_cls_missing_is_exact_video_fallback():
    model = DenseCenterControl(
        "center_cls",
        input_dim=8,
        video_dim=10,
        width=8,
        layers=1,
        heads=2,
        dropout=0,
        parameter_limit=100_000,
    ).eval()
    patches = torch.randn(2, 9, 8)
    center = torch.full((2, 8), float("nan"))
    video = torch.randn(2, 10)
    output = model(
        patches, video, center_cls=center, center_cls_valid=torch.zeros(2, dtype=torch.bool)
    )
    assert torch.equal(output["local_logits"], torch.zeros_like(output["local_logits"]))
    assert torch.equal(output["logits"], output["video_logits"])


def test_dense_transformer_uses_coordinates():
    model = DenseCenterControl(
        "dense_transformer",
        input_dim=8,
        video_dim=10,
        width=8,
        layers=1,
        heads=2,
        dropout=0,
        parameter_limit=100_000,
    ).eval()
    patches = torch.randn(2, 9, 8)
    video = torch.randn(2, 10)
    coordinates = torch.randn(9, 2)
    first = model(patches, video, coordinates=coordinates)["local_logits"]
    second = model(patches, video, coordinates=coordinates.flip(0))["local_logits"]
    assert not torch.equal(first, second)


def test_invalid_arm_and_shapes_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        DenseCenterControl("invented")
    model = DenseCenterControl("dense_cnn", input_dim=8, video_dim=10, width=8, layers=1, heads=2)
    with pytest.raises(ValueError, match="square"):
        model(torch.randn(2, 8, 8), torch.randn(2, 10))


def unrestricted_model():
    return UnrestrictedTemplateControl(
        input_dim=12,
        video_dim=15,
        rank=8,
        slots=3,
        templates_per_class=2,
        iterations=2,
        width=16,
        dropout=0,
        parameter_limit=100_000,
    )


def test_unrestricted_template_control_is_finite_and_active():
    model = unrestricted_model()
    output = model(torch.randn(4, 9, 12), torch.randn(4, 15))
    assert output["class_template_slot_masses"].shape == (4, 3, 2, 3)
    assert output["class_template_nuisance_transform"].shape == (4, 3, 2, 2, 2)
    loss = output["logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_unrestricted_control_can_hide_evidence_for_one_class_template():
    model = unrestricted_model().eval()
    patches, video = torch.randn(2, 9, 12), torch.randn(2, 15)
    initial = model(patches, video)
    with torch.no_grad():
        model.visibility_bias[1, 0].fill_(-30)
    changed = model(patches, video)
    assert torch.equal(
        initial["class_template_slot_masses"][:, [0, 2]],
        changed["class_template_slot_masses"][:, [0, 2]],
    )
    assert torch.all(
        changed["class_template_slot_masses"][:, 1, 0]
        < initial["class_template_slot_masses"][:, 1, 0]
    )


def test_unrestricted_control_exact_video_fallback():
    model = unrestricted_model().eval()
    patches = torch.full((2, 9, 12), float("nan"))
    video = torch.randn(2, 15)
    output = model(patches, video, valid=torch.zeros(2, 9, dtype=torch.bool))
    assert torch.equal(output["local_logits"], torch.zeros_like(output["local_logits"]))
    assert torch.equal(output["logits"], output["video_logits"])


def test_unrestricted_zero_norm_prototype_has_neutral_cost_two():
    model = unrestricted_model().eval()
    with torch.no_grad():
        model.template_descriptors.zero_()
        model.raw_cost_weights[..., 1].fill_(-100)
    result = model(torch.randn(2, 9, 12), torch.randn(2, 15))
    expected = 2 * result["class_template_slot_masses"].sum(-1)
    appearance_weight = result["class_template_cost_weights"][..., 0]
    torch.testing.assert_close(result["template_energies"], expected * appearance_weight[None])


def test_common_video_branch_is_exactly_interchangeable():
    cnn = DenseCenterControl(
        "dense_cnn", input_dim=8, video_dim=10, width=8, layers=1, heads=2, dropout=0
    ).eval()
    transformer = DenseCenterControl(
        "dense_transformer",
        input_dim=8,
        video_dim=10,
        width=8,
        layers=1,
        heads=2,
        dropout=0,
    ).eval()
    transformer.video_head.load_state_dict(cnn.video_head.state_dict())
    video = torch.randn(3, 10)
    assert torch.equal(cnn.video_head(video), transformer.video_head(video))


def test_default_dense_mechanisms_are_capacity_matched():
    # Widths were chosen label-blind to match A5's 551,558 active parameters.
    cnn = DenseCenterControl("dense_cnn", width=100, heads=4)
    transformer = DenseCenterControl("dense_transformer", width=60, heads=4)
    unrestricted = UnrestrictedTemplateControl()
    target = 551_558
    for model in (cnn, transformer, unrestricted):
        assert abs(model.trainable_parameters / target - 1) < 0.05
