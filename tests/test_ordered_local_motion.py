from __future__ import annotations

import pytest
import torch

from hac.ordered_local_motion import ARMS, OrderedLocalMotion


def model(arm: str) -> OrderedLocalMotion:
    torch.manual_seed(11)
    return OrderedLocalMotion(12, arm, width=16, rank=4, heads=4, dropout=0.0)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(2, 8, 9, 12, generator=generator),
        torch.arange(8).float()[None].expand(2, -1) * (8 / 30),
        torch.zeros(2, 6),
    )


@pytest.mark.parametrize("arm", ARMS)
def test_finite_probability_and_classification_gradient(arm: str) -> None:
    net = model(arm)
    output = net(*inputs())
    assert output["probabilities"].shape == (2, 3)
    assert torch.allclose(output["probabilities"].sum(-1), torch.ones(2))
    loss = -output["probabilities"][:, 0].log().mean()
    loss.backward()
    assert net.projection[1].weight.grad.abs().sum() > 0
    if "relational" in arm:
        assert net.left_projection.weight.grad.abs().sum() > 0
        assert net.time_kernel[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ARMS)
def test_invalid_nan_tokens_are_inert(arm: str) -> None:
    tokens, times, quality = inputs()
    valid = torch.ones(tokens.shape[:3], dtype=torch.bool)
    valid[:, 1, 2] = False
    valid[:, 6, 7] = False
    net = model(arm).eval()
    before = net(tokens, times, quality, valid)
    tokens[~valid] = torch.nan
    after = net(tokens, times, quality, valid)
    assert torch.equal(before["probabilities"], after["probabilities"])


@pytest.mark.parametrize("arm", ("ordered_relational", "correspondence_relational"))
def test_repeated_static_grid_has_no_relational_motion(arm: str) -> None:
    tokens, times, quality = inputs()
    tokens = tokens[:, :1].expand(-1, 8, -1, -1).clone()
    output = model(arm)(tokens, times, quality)
    assert torch.allclose(
        output["interactions"], torch.zeros_like(output["interactions"]), atol=1e-10
    )
    assert torch.allclose(
        output["correspondence_displacement"],
        torch.zeros_like(output["correspondence_displacement"]),
        atol=1e-7,
    )


def test_relational_operator_preserves_lag_and_region_structure() -> None:
    tokens, times, quality = inputs()
    net = model("ordered_relational").eval()
    original = net(tokens, times, quality)["interactions"]
    permutation = torch.tensor([0, 6, 2, 7, 4, 1, 5, 3])
    shuffled = net(tokens[:, permutation], times, quality)["interactions"]
    assert original.shape == (2, 3, 9, 9)
    assert not torch.allclose(original, shuffled)
    spatial = torch.tensor([2, 0, 1, 5, 3, 4, 8, 6, 7])
    region_shuffle = net(tokens[:, :, spatial], times, quality)["interactions"]
    assert torch.allclose(region_shuffle, original[:, :, spatial][:, :, :, spatial], atol=1e-6)


def test_physical_time_conditioning_does_not_divide_changes_by_short_dt() -> None:
    tokens, times, quality = inputs()
    net = model("ordered_relational").eval()
    with torch.no_grad():
        for parameter in net.time_kernel.parameters():
            parameter.zero_()
    long = net(tokens, times, quality)["interactions"]
    short = net(tokens, times / 4, quality)["interactions"]
    assert torch.allclose(long, short)


def test_learned_time_gain_does_not_cancel_on_uniform_timestamp_grids() -> None:
    tokens, times, quality = inputs()
    net = model("ordered_relational").eval()
    with torch.no_grad():
        for parameter in net.time_kernel.parameters():
            parameter.zero_()
        net.time_kernel[0].weight[0, 0] = 1.0
        net.time_kernel[-1].weight[0, 0] = 1.0
    long = net(tokens, times, quality)["interactions"]
    short = net(tokens, times / 4, quality)["interactions"]
    assert long.norm() > short.norm()
    assert not torch.allclose(long, short)


def test_correspondence_has_no_future_forecasting_or_label_argument() -> None:
    tokens, times, quality = inputs()
    with pytest.raises(TypeError, match="labels"):
        model("correspondence_relational")(tokens, times, quality, labels=torch.zeros(2))
    with pytest.raises(ValueError, match="increase"):
        model("ordered_relational")(tokens, times.flip(1), quality)


@pytest.mark.parametrize("arm", ARMS)
def test_full_width_parameter_bound(arm: str) -> None:
    net = OrderedLocalMotion(768, arm)
    assert net.trainable_parameters < 1_000_000


def test_materializer_fallback_is_exact_and_rejects_nonzero_invalid_long() -> None:
    import numpy as np
    from run_okutama_ordered_motion import select_clip_block

    short = np.arange(24, dtype=np.float16).reshape(3, 2, 2, 2)
    long = short + 5
    valid = np.asarray([True, False, True])
    long[1] = 0
    selected = select_clip_block(short, long, valid)
    assert np.array_equal(selected[1], short[1])
    assert np.array_equal(selected[valid], long[valid])
    long[1, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="zero sentinel"):
        select_clip_block(short, long, valid)


def test_training_scaler_never_reads_held_tokens_or_quality() -> None:
    import numpy as np
    from run_okutama_ordered_motion import scaling

    tokens = np.arange(48, dtype=np.float16).reshape(4, 2, 2, 3)
    quality = np.arange(8, dtype=np.float32).reshape(4, 2)
    train = np.asarray([0, 1])
    expected = scaling(tokens, quality, train)
    tokens[2:] = np.nan
    quality[2:] = np.nan
    observed = scaling(tokens, quality, train)
    for key in expected:
        assert np.array_equal(observed[key], expected[key])


def test_cpu_synthetic_fit_replays_and_rejects_scenario_overlap(tmp_path) -> None:
    import json

    import numpy as np
    from run_okutama_ordered_motion import PROTOCOL, train_fit

    protocol = json.loads(PROTOCOL.read_text())
    protocol.update(width=16, rank=4, heads=4, layers=1, batch_size=3, dropout=0.0)
    generator = np.random.default_rng(17)
    tokens = generator.normal(size=(6, 8, 9, 768)).astype(np.float16)
    data = {
        "quality": generator.normal(size=(6, 6)).astype(np.float32),
        "times": np.broadcast_to(np.arange(8, dtype=np.float32), (6, 8)).copy(),
        "labels": np.asarray([0, 1, 2, 0, 1, 2]),
        "scenarios": np.asarray(["a", "a", "a", "b", "b", "b"]),
        "sample_ids": np.asarray([f"toy-{i}" for i in range(6)]),
    }
    request = {
        "arm": "ordered_relational",
        "seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    train, held = np.arange(3), np.arange(3, 6)
    values, receipt = train_fit(
        tmp_path, tokens, data, protocol, request, train, held, "cpu", epochs=1
    )
    repeated, repeated_receipt = train_fit(
        tmp_path, tokens, data, protocol, request, train, held, "cpu", epochs=1
    )
    assert np.array_equal(values, repeated)
    assert receipt == repeated_receipt
    assert receipt["outer_labels_used_for_training_or_selection"] is False
    data["scenarios"][:] = "same"
    with pytest.raises(RuntimeError, match="overlaps"):
        train_fit(
            tmp_path / "overlap", tokens, data, protocol, request, train, held, "cpu", epochs=1
        )
