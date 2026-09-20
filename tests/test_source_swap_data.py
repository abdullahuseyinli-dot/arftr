"""Source swap must leave historical fallback inputs and other streams intact."""

import numpy as np
import pytest

from hac.source_swap_data import derive_features, immutable_json, masked_long_source


def tokens():
    rng = np.random.default_rng(13)
    short = rng.normal(size=(4, 8, 9, 768)).astype(np.float16)
    long = rng.normal(size=short.shape).astype(np.float16)
    dino = rng.normal(size=(4, 16, 1, 768)).astype(np.float16)
    valid = np.array([True, False, True, False])
    long[~valid] = 0
    long_dino = dino.copy()
    long_dino[~valid] = 0
    regenerated = short.copy()
    regenerated[2] = long[2]
    source_fallback = np.array([False, False, True, True])
    return short, long, dino, long_dino, valid, regenerated, source_fallback


def test_long_only_swap_retains_both_fallback_kinds_and_other_modalities():
    short, long, dino, long_dino, valid, regenerated, source = tokens()
    before, base_before = derive_features(short, long, dino, long_dino, valid)
    replacement = masked_long_source(regenerated, long, valid, source)
    assert not replacement[~valid].any()
    after, base_after = derive_features(short, replacement, dino, long_dino, valid)
    np.testing.assert_array_equal(before[1:], after[1:])
    np.testing.assert_array_equal(before[:, :768], after[:, :768])
    np.testing.assert_array_equal(before[:, 1536:], after[:, 1536:])
    for name in base_before:
        np.testing.assert_array_equal(base_before[name][1:], base_after[name][1:])
        assert np.any(base_before[name][0] != base_after[name][0])
    assert before.shape == (4, 3072)
    assert [value.shape[1] for value in base_before.values()] == [768, 4608, 9984, 10752]


def test_invalid_fallback_cache_or_dtype_refused():
    _, long, _, _, valid, regenerated, source = tokens()
    regenerated[2, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="fallback differs"):
        masked_long_source(regenerated, long, valid, source)
    with pytest.raises(ValueError, match="dtype"):
        masked_long_source(regenerated.astype(np.float32), long, valid, source)


def test_immutable_json_preserves_original(tmp_path):
    path = tmp_path / "receipt.json"
    immutable_json(path, {"identity": 1})
    original = path.read_bytes()
    immutable_json(path, {"identity": 1})
    with pytest.raises(RuntimeError, match="changed"):
        immutable_json(path, {"identity": 2})
    assert path.read_bytes() == original
