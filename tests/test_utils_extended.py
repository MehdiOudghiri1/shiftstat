from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shiftstat.exceptions import ValidationError
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
    labels_from_probabilities,
    predictive_entropy,
)
from shiftstat.utils.random import check_random_state, random_state_to_int
from shiftstat.utils.split import make_reference_target_split


def test_probability_helpers_accept_binary_matrix_and_reject_multiclass() -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])

    positive = extract_positive_class_probabilities(probabilities)

    assert np.allclose(positive, [0.2, 0.9])
    assert np.allclose(confidence_from_probabilities(positive), [0.8, 0.9])
    assert np.array_equal(labels_from_probabilities(positive, threshold=0.5), [0, 1])
    assert predictive_entropy(positive).shape == (2,)
    with pytest.raises(ValidationError):
        extract_positive_class_probabilities(np.ones((3, 3)))


def test_random_state_helpers_cover_supported_and_invalid_inputs() -> None:
    seeded = check_random_state(123)
    existing = np.random.RandomState(124)

    assert isinstance(seeded, np.random.RandomState)
    assert check_random_state(existing) is existing
    assert random_state_to_int(None) is None
    assert random_state_to_int(np.int64(5)) == 5
    assert isinstance(random_state_to_int(existing), int)
    with pytest.raises(TypeError):
        check_random_state("seed")
    with pytest.raises(TypeError):
        random_state_to_int("seed")


def test_reference_target_split_supports_labeled_and_unlabeled_data() -> None:
    X = pd.DataFrame({"x": range(10), "z": range(10, 20)})
    y = np.array([0, 1] * 5)

    unlabeled = make_reference_target_split(X, target_size=0.2, random_state=3)
    labeled = make_reference_target_split(X, y, target_size=0.3, random_state=4, stratify=y)

    assert len(unlabeled.X_ref) == 8
    assert unlabeled.y_ref is None
    assert len(labeled.X_target) == 3
    assert labeled.y_ref is not None
    assert labeled.y_target is not None
    assert set(labeled.y_target.tolist()) <= {0, 1}
