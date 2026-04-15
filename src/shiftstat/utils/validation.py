"""Validation helpers for vectors and tabular data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from shiftstat.exceptions import ValidationError
from shiftstat.typing import ArrayLike, TabularLike, VectorLike


def ensure_2d_tabular(X: TabularLike, *, copy: bool = False) -> pd.DataFrame | np.ndarray:
    """Validate a tabular input as a two-dimensional array or DataFrame."""

    if isinstance(X, pd.DataFrame):
        if X.shape[1] == 0:
            raise ValidationError("Tabular input must contain at least one feature.")
        return X.copy() if copy else X

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValidationError(f"Expected a 2D tabular input, got shape {array.shape}.")
    if array.shape[1] == 0:
        raise ValidationError("Tabular input must contain at least one feature.")
    return array.copy() if copy else array


def ensure_1d(x: VectorLike | ArrayLike, *, name: str = "vector") -> np.ndarray:
    """Validate an input as a one-dimensional NumPy array."""

    if isinstance(x, pd.Series):
        values = x.to_numpy()
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes, np.ndarray)):
        values = np.asarray(list(x))
    else:
        values = np.asarray(x)
    if values.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional, got shape {values.shape}.")
    return values


def validate_same_length(*arrays: ArrayLike) -> None:
    """Validate that all provided arrays have the same first-dimension length."""

    lengths = []
    for array in arrays:
        if isinstance(array, pd.DataFrame):
            lengths.append(len(array))
        else:
            lengths.append(len(np.asarray(array)))
    if len(set(lengths)) > 1:
        raise ValidationError(f"Expected equal lengths, received lengths {lengths}.")

