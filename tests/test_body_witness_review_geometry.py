from __future__ import annotations

import pytest

from experiments.derive_okutama_body_witness_review_css_inverse import v5_display_x_scale


def test_v5_css_inverse_scale_is_one_when_height_cap_is_inactive():
    assert v5_display_x_scale(
        640, 400, element_width=640, capped_element_height=538.4375
    ) == 1.0


def test_v5_css_inverse_recovers_height_capped_horizontal_compression():
    scale = v5_display_x_scale(
        109, 138, element_width=640, capped_element_height=538.4375
    )
    assert scale == pytest.approx(425.2875905797102 / 640)


@pytest.mark.parametrize("dimension", (0, -1, float("nan")))
def test_v5_css_inverse_rejects_invalid_geometry(dimension):
    with pytest.raises(ValueError, match="positive and finite"):
        v5_display_x_scale(
            dimension, 100, element_width=640, capped_element_height=538.4375
        )
