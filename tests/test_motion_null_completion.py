import copy

import numpy as np
import pytest

from experiments.complete_okutama_motion_null_contrast import parity_measurements


def fixture():
    return {"sample_ids": np.array(["a", "b"]), "anchor_probabilities": np.array([[0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]),
            "candidate_probabilities": np.array([[0.1, 0.69, 0.21], [0.1, 0.21, 0.69]]), "delta": np.array([0.02, -0.02])}


def test_numeric_only_exception_records_original_failure():
    old = fixture()
    saved = copy.deepcopy(old)
    saved["delta"][0] += 0.00002
    result = parity_measurements(saved, old, 1e-6)
    assert result["accepted_by_user_exception"]
    assert not result["original_numeric_parity_pass"]
    assert result["class_prediction_disagreements"] == 0


@pytest.mark.parametrize("change", ["class", "anchor", "identity", "nonfinite", "bound", "mass"])
def test_numerical_exception_never_waives_other_checks(change):
    old = fixture()
    saved = copy.deepcopy(old)
    if change == "class":
        saved["candidate_probabilities"][0] = [0.1, 0.2, 0.7]
    elif change == "anchor":
        saved["anchor_probabilities"][0, 0] += 0.01
    elif change == "identity":
        saved["sample_ids"][0] = "x"
    elif change == "nonfinite":
        saved["delta"][0] = np.nan
    elif change == "bound":
        saved["delta"][0] = 0.51
    else:
        saved["candidate_probabilities"][0, 0] += 0.01
    with pytest.raises(RuntimeError):
        parity_measurements(saved, old, 1e-6)
