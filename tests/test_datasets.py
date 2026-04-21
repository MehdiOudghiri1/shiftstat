from __future__ import annotations

import numpy as np
import pytest

from shiftstat.datasets import (
    make_configurable_shift_classification,
    make_covariate_shift_regression,
    make_mixed_type_shift,
    make_severity_controlled_shift,
)


def test_regression_and_mixed_synthetic_datasets_have_expected_shapes() -> None:
    regression = make_covariate_shift_regression(
        n_samples_ref=30,
        n_samples_target=20,
        n_features=4,
        random_state=11,
    )
    mixed = make_mixed_type_shift(n_samples_ref=25, n_samples_target=15, random_state=12)

    assert regression.task == "regression"
    assert regression.X_ref.shape == (30, 4)
    assert regression.y_target.dtype.kind == "f"
    assert mixed.task == "classification"
    assert {"cat_0", "cat_1"}.issubset(mixed.X_target.columns)


@pytest.mark.parametrize("pattern", ["covariate", "subgroup", "calibration", "mixed"])
def test_configurable_shift_patterns_are_seeded(pattern: str) -> None:
    first = make_configurable_shift_classification(
        n_samples_ref=40,
        n_samples_target=35,
        n_numeric_features=3,
        n_categorical_features=1,
        shift_pattern=pattern,
        random_state=13,
    )
    second = make_configurable_shift_classification(
        n_samples_ref=40,
        n_samples_target=35,
        n_numeric_features=3,
        n_categorical_features=1,
        shift_pattern=pattern,
        random_state=13,
    )

    assert np.array_equal(first.y_ref, second.y_ref)
    assert np.allclose(first.target_predictions, second.target_predictions)


def test_configurable_shift_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="class_imbalance"):
        make_configurable_shift_classification(class_imbalance=1.0)
    with pytest.raises(ValueError, match="n_numeric_features"):
        make_configurable_shift_classification(n_numeric_features=1)
    with pytest.raises(ValueError, match="n_categorical_features"):
        make_configurable_shift_classification(n_categorical_features=0)
    with pytest.raises(ValueError, match="n_categories"):
        make_configurable_shift_classification(n_categories=1)
    with pytest.raises(ValueError, match="shift_pattern"):
        make_configurable_shift_classification(shift_pattern="unknown")


def test_severity_controlled_shift_delegates_to_classification_generator() -> None:
    data = make_severity_controlled_shift(severity=0.7, random_state=14)

    assert data.task == "classification"
    assert data.X_ref.shape[1] == data.X_target.shape[1]
