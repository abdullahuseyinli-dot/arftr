import numpy as np

from hac.body_witness_data import (PADDING_RGB, apply_raw_body_mask, crop_raw_body,
                                   make_body_crop_geometry, raw_body_masks)
from hac.rgb_witness_data import witness_geometry


def test_raw_masks_are_disjoint_and_hidden_corruption_cannot_change_visible_view():
    rng = np.random.default_rng(4)
    rgb = rng.integers(0,256,size=(120,160,3),dtype=np.uint8)
    geometry = make_body_crop_geometry((40.2,20.4,90.8,100.1),extent=1.25,source_size=(160,120))
    raw = crop_raw_body(rgb,geometry)
    masks = raw_body_masks(geometry)
    assert not (masks["upper_visible"] & masks["lower_visible"]).any()
    upper = apply_raw_body_mask(raw,geometry,"upper_visible")
    changed = raw.copy(); changed[masks["lower_visible"]] ^= 255
    assert np.array_equal(upper,apply_raw_body_mask(changed,geometry,"upper_visible"))
    assert np.all(upper[~masks["upper_visible"]] == np.asarray(PADDING_RGB,dtype=np.uint8))


def test_geometry_six_fields_and_valid_fraction():
    value = witness_geometry((40,20,100,100),(30,10,110,110),(160,120))
    assert value.shape == (6,)
    assert np.isfinite(value).all()
    assert 0 < value[-1] <= 1
