from __future__ import annotations

import numpy as np
from PIL import Image

from hac.center_completion_data import (
    MASK_BANDS,
    SOURCE_SIZE,
    apply_raw_mask,
    full_support_patch_mask,
    perturb_hidden_raw_pixels,
    prepare_center,
    raw_actor_crop,
    raw_fill_mask,
)
from hac.center_evidence_completion import (
    AffineTransport,
    bilinear_transport,
    fit_visible_anchor_transport,
    robust_affine_fit,
    transport_consensus,
    visible_anchor_matches,
)


def _row(box: tuple[float, float, float, float]) -> dict[str, str]:
    return {
        "bbox_xmin": str(box[0]),
        "bbox_ymin": str(box[1]),
        "bbox_xmax": str(box[2]),
        "bbox_ymax": str(box[3]),
    }


def test_hidden_pixel_perturbation_is_destroyed_before_preprocessing() -> None:
    x = np.arange(SOURCE_SIZE[0], dtype=np.uint16)[None, :]
    y = np.arange(SOURCE_SIZE[1], dtype=np.uint16)[:, None]
    rgb = np.stack(
        [
            np.broadcast_to(x % 256, (SOURCE_SIZE[1], SOURCE_SIZE[0])),
            np.broadcast_to(y % 256, (SOURCE_SIZE[1], SOURCE_SIZE[0])),
            (x + y) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    crop = raw_actor_crop(image, (480.0, 180.0, 800.0, 600.0))
    for mask_id in MASK_BANDS:
        mask = raw_fill_mask(crop, mask_id)
        perturbed = perturb_hidden_raw_pixels(crop.rgb, mask)
        assert not np.array_equal(perturbed[mask], crop.rgb[mask])
        assert np.array_equal(
            apply_raw_mask(perturbed, mask),
            apply_raw_mask(crop.rgb, mask),
        )


def test_center_targets_have_full_support_and_do_not_overlap() -> None:
    image = Image.new("RGB", SOURCE_SIZE, (40, 80, 120))
    prepared = prepare_center(image, _row((500.0, 150.0, 780.0, 650.0)))
    assert prepared.unmasked_pixels.shape == (3, 384, 384)
    assert prepared.masked_pixels.shape == (2, 3, 384, 384)
    assert prepared.target_patch_masks.shape == (2, 27, 27)
    assert prepared.visible_patch_masks.shape == (2, 27, 27)
    assert prepared.target_patch_masks.reshape(2, -1).any(axis=1).all()
    assert not np.logical_and(*prepared.target_patch_masks).any()
    assert (prepared.target_patch_masks <= prepared.valid_patch_mask[None]).all()
    assert not np.logical_and(
        prepared.target_patch_masks, prepared.visible_patch_masks
    ).any()


def test_patch_support_requires_complete_fourteen_pixel_footprint() -> None:
    raw = np.zeros((384, 384), dtype=bool)
    raw[:14, :14] = True
    patches = full_support_patch_mask(raw)
    assert patches.sum() == 1
    assert patches[0, 0]


def test_visible_matches_exclude_hidden_center_tokens() -> None:
    rng = np.random.default_rng(3)
    center = rng.normal(size=(27, 27, 16)).astype(np.float32)
    neighbor = center.copy()
    target = np.zeros((27, 27), dtype=bool)
    target[8:19, 10:17] = True
    matches = visible_anchor_matches(center, neighbor, target, radius=0)
    matched_yx = matches.center_xy[:, ::-1].astype(int)
    assert len(matches.center_xy) == 27 * 27 - int(target.sum())
    assert not target[matched_yx[:, 0], matched_yx[:, 1]].any()


def test_robust_affine_recovers_translation() -> None:
    yy, xx = np.mgrid[2:25:4, 2:25:4]
    center = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    neighbor = center + np.array([1.25, -0.75])
    matrix, residual = robust_affine_fit(center, neighbor)
    expected = np.array([[1, 0, 1.25], [0, 1, -0.75]], dtype=np.float64)
    assert np.allclose(matrix, expected, atol=1e-10)
    assert residual < 1e-10


def test_huber_irls_reduces_outlier_influence_relative_to_least_squares() -> None:
    yy, xx = np.mgrid[2:25:4, 2:25:4]
    center = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    clean = center + np.array([1.25, -0.75])
    neighbor = clean.copy()
    neighbor[:3] += np.array([8.0, -7.0])
    matrix, _ = robust_affine_fit(center, neighbor)
    design = np.column_stack([center, np.ones(len(center))])
    least_squares = np.linalg.lstsq(design, neighbor, rcond=None)[0].T
    robust_error = np.median(np.linalg.norm(design @ matrix.T - clean, axis=1))
    least_squares_error = np.median(
        np.linalg.norm(design @ least_squares.T - clean, axis=1)
    )
    assert robust_error < least_squares_error


def test_transport_uses_dustbin_when_matches_are_insufficient() -> None:
    center = np.zeros((27, 27, 8), dtype=np.float32)
    neighbor = np.zeros_like(center)
    target = np.ones((27, 27), dtype=bool)
    transform, matches = fit_visible_anchor_transport(center, neighbor, target)
    values, observed = bilinear_transport(neighbor, target, transform)
    assert len(matches.center_xy) == 0
    assert not transform.valid
    assert transform.reason == "too_few_matches"
    assert not observed.any()
    assert np.count_nonzero(values) == 0


def test_bilinear_transport_samples_affine_target_coordinates() -> None:
    yy, xx = np.mgrid[:27, :27]
    neighbor = np.stack([xx, yy], axis=2).astype(np.float32)
    target = np.zeros((27, 27), dtype=bool)
    target[10, 11] = True
    transform = AffineTransport(
        matrix=np.array([[1, 0, 0.5], [0, 1, 0.25]], dtype=np.float64),
        match_count=20,
        weighted_residual=0.1,
        valid=True,
        reason="ok",
    )
    values, observed = bilinear_transport(neighbor, target, transform)
    assert observed.tolist() == [True]
    assert np.allclose(values[0], [11.5, 10.25])


def test_consensus_accepts_agreeing_donors_even_with_high_identity_entropy() -> None:
    donors = np.ones((4, 2, 3), dtype=np.float32)
    observed = np.ones((4, 2), dtype=bool)
    logits = np.zeros((4, 2), dtype=np.float64)
    center_only = np.zeros((2, 3), dtype=np.float32)
    consensus = transport_consensus(donors, observed, logits, center_only)
    assert consensus.available.all()
    assert np.array_equal(consensus.features, np.ones((2, 3), dtype=np.float32))
    assert np.allclose(consensus.donor_weight_entropy, np.log(4))


def test_consensus_all_invalid_copies_center_only_exactly() -> None:
    donors = np.zeros((4, 2, 3), dtype=np.float32)
    observed = np.zeros((4, 2), dtype=bool)
    logits = np.full((4, 2), -np.inf)
    center_only = np.arange(6, dtype=np.float32).reshape(2, 3)
    consensus = transport_consensus(donors, observed, logits, center_only)
    assert not consensus.available.any()
    assert np.array_equal(consensus.features, center_only)
