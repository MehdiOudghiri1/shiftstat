from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shiftstat.exceptions import SchemaMismatchError, ValidationError
from shiftstat.utils import (
    align_tabular_inputs,
    ensure_1d,
    ensure_2d_tabular,
    extract_feature_names,
    infer_feature_types,
    validate_same_length,
)


def test_ensure_2d_tabular_accepts_dataframe_and_array() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert ensure_2d_tabular(df).shape == (2, 2)
    assert ensure_2d_tabular(array).shape == (2, 2)


def test_ensure_2d_tabular_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValidationError):
        ensure_2d_tabular(np.array([1, 2, 3]))


def test_validate_same_length_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValidationError):
        validate_same_length(np.array([1, 2]), np.array([1, 2, 3]))


def test_align_tabular_inputs_reorders_dataframe_columns() -> None:
    ref = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    new = pd.DataFrame({"x2": [5, 6], "x1": [7, 8]})
    ref_aligned, new_aligned = align_tabular_inputs(ref, new)
    assert list(ref_aligned.columns) == ["x1", "x2"]
    assert list(new_aligned.columns) == ["x1", "x2"]


def test_align_tabular_inputs_rejects_schema_mismatch() -> None:
    ref = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    new = pd.DataFrame({"x1": [5, 6], "x3": [7, 8]})
    with pytest.raises(SchemaMismatchError):
        align_tabular_inputs(ref, new)


def test_extract_feature_names_and_types() -> None:
    frame = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    assert extract_feature_names(frame) == ["num", "cat"]
    feature_types = infer_feature_types(frame)
    assert feature_types == {"num": "continuous", "cat": "categorical"}


def test_ensure_1d_converts_series() -> None:
    series = pd.Series([1, 2, 3])
    assert ensure_1d(series).shape == (3,)
