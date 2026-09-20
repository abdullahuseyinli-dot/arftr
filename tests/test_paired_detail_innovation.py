from __future__ import annotations

import numpy as np
import torch

from hac.fine_local_motion import expand_coarse_tokens
from hac.paired_detail_innovation import PairedDetailInnovation, pdi_loss


def fixture():
    generator = torch.Generator().manual_seed(19)
    coarse = torch.randn(6, 8, 9, 12, generator=generator)
    repeated = expand_coarse_tokens(coarse)
    fine = repeated + 0.1 * torch.randn(repeated.shape, generator=generator)
    times = torch.arange(8).float()[None].expand(6, -1) / 3
    quality = torch.randn(6, 6, generator=generator)
    anchor = torch.softmax(torch.randn(6, 3, generator=generator), dim=1)
    return coarse, repeated, fine, times, quality, anchor


def model():
    torch.manual_seed(23)
    return PairedDetailInnovation(
        input_dim=12, rank=4, width=16, layers=1, heads=4, parameter_limit=100_000
    )


def test_zero_initialized_model_preserves_anchor_for_every_arm():
    coarse, _repeated, fine, times, quality, anchor = fixture()
    net = model().eval()
    for arm in ("coarse_residual", "paired_spatial", "paired_spatial_temporal"):
        out = net(fine, coarse, times, quality, anchor, arm=arm)
        assert torch.equal(out["actions"][:, 0], anchor)
        assert torch.all(out["correction"] == 0)
        assert torch.equal(out["probabilities"], anchor)


def test_null_detail_is_exact_after_optimizer_updates_and_weights_are_shared():
    coarse, repeated, _fine, times, quality, anchor = fixture()
    net = model().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    assert len({id(p) for p in net.parameters()}) == len(list(net.parameters()))
    labels = torch.arange(6) % 3
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = net(repeated, coarse, times, quality, anchor, arm="paired_spatial_temporal")
        loss, _ = pdi_loss(out, anchor, labels, torch.ones(3))
        loss.backward()
        optimizer.step()
    out = net(repeated, coarse, times, quality, anchor, arm="paired_spatial_temporal")
    assert torch.allclose(out["detail_moments"], torch.zeros_like(out["detail_moments"]), atol=1e-10)
    assert torch.equal(out["representation"], torch.zeros_like(out["representation"]))
    assert torch.equal(out["correction"], torch.zeros_like(out["correction"]))
    assert torch.equal(out["probabilities"], anchor)


def test_fine_detail_activates_shared_path_and_all_parameters_receive_gradients():
    coarse, _repeated, fine, times, quality, anchor = fixture()
    net = model().train()
    labels = torch.arange(6) % 3
    first = net(fine, coarse, times, quality, anchor, arm="paired_spatial_temporal")
    loss, _ = pdi_loss(first, anchor, labels, torch.ones(3))
    loss.backward()
    assert net.correction_head.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = net(fine, coarse, times, quality, anchor, arm="paired_spatial_temporal")
    loss, _ = pdi_loss(second, anchor, labels, torch.ones(3))
    loss.backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert net.detail_projection[0].weight.grad.abs().sum() > 0
    assert net.projection.weight.grad.abs().sum() > 0


def test_scaffold_is_nonzero_and_spatial_arm_masks_only_motion_detail():
    coarse, _repeated, fine, times, quality, anchor = fixture()
    net = model().eval()
    spatial = net(fine, coarse, times, quality, anchor, arm="paired_spatial")
    full = net(fine, coarse, times, quality, anchor, arm="paired_spatial_temporal")
    assert spatial["coarse_scaffold"].abs().sum() > 0
    assert spatial["detail_moments"][:, 0].abs().sum() > 0
    assert torch.all(spatial["detail_moments"][:, 1:] == 0)
    assert full["detail_moments"][:, 1:].abs().sum() > 0


def test_fixed_bound_and_simplex_after_nonzero_head():
    coarse, _repeated, fine, times, quality, anchor = fixture()
    net = model().eval()
    with torch.no_grad():
        net.correction_head.weight.fill_(2.0)
    output = net(fine, coarse, times, quality, anchor, arm="paired_spatial_temporal")
    assert output["correction"].abs().max() <= 0.5
    assert torch.allclose(output["actions"].sum(2), torch.ones(6, 4), atol=1e-6)
    assert torch.equal(output["actions"][:, 0], anchor)
    assert net.trainable_parameters < 100_000


def test_forward_has_no_label_or_support_input():
    coarse, _repeated, fine, times, quality, anchor = fixture()
    try:
        model()(fine, coarse, times, quality, anchor, arm="paired_spatial", labels=np.zeros(6))
    except TypeError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("labels unexpectedly accepted")
