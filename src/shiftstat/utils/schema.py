"""Schema and feature metadata helpers for tabular inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from shiftstat.exceptions import SchemaMismatchError, ValidationError
from shiftstat.typing import FeatureTypes, TabularLike
from shiftstat.utils.validation import ensure_2d_tabular


@dataclass(frozen=True)
class TabularSchema:
    """Lightweight tabular schema metadata."""

    feature_names: list[str]
    feature_types: FeatureTypes
    n_features: int


def extract_feature_names(X: TabularLike) -> list[str]:
    """Extract feature names from a DataFrame or synthesize names for arrays."""

    X_validated = ensure_2d_tabular(X)
    if isinstance(X_validated, pd.DataFrame):
        return [str(column) for column in X_validated.columns]
    return [f"feature_{index}" for index in range(X_validated.shape[1])]


def infer_feature_types(
    X: TabularLike,
    *,
    categorical_features: list[str] | list[int] | None = None,
) -> FeatureTypes:
    """Infer feature types as `continuous` or `categorical`."""

    X_validated = ensure_2d_tabular(X)
    feature_names = extract_feature_names(X_validated)
    feature_types: FeatureTypes = {}
    categorical_name_set: set[str] = set()

    if categorical_features is not None:
        if isinstance(X_validated, pd.DataFrame):
            categorical_name_set = {str(name) for name in categorical_features}
        else:
            categorical_name_set = {
                feature_names[int(index)]
                for index in categorical_features
                if int(index) < len(feature_names)
            }

    if isinstance(X_validated, pd.DataFrame):
        for feature_name in feature_names:
            dtype = X_validated[feature_name].dtype
            if (
                feature_name in categorical_name_set
                or is_bool_dtype(dtype)
                or not is_numeric_dtype(dtype)
            ):
                feature_types[feature_name] = "categorical"
            else:
                feature_types[feature_name] = "continuous"
        return feature_types

    for column_index, feature_name in enumerate(feature_names):
        column = X_validated[:, column_index]
        if feature_name in categorical_name_set:
            feature_types[feature_name] = "categorical"
            continue
        if np.issubdtype(column.dtype, np.number) and column.dtype != object:
            feature_types[feature_name] = "continuous"
            continue
        unique_values = pd.Series(column).dropna().unique()
        is_categorical = len(unique_values) <= min(10, max(2, X_validated.shape[0] // 10))
        feature_types[feature_name] = "categorical" if is_categorical else "continuous"
    return feature_types


def validate_tabular_pair_schema(
    X_ref: TabularLike,
    X_new: TabularLike,
    *,
    require_feature_name_match: bool = True,
) -> tuple[TabularSchema, TabularSchema]:
    """Validate that a pair of tabular inputs is schema-compatible."""

    X_ref_validated = ensure_2d_tabular(X_ref)
    X_new_validated = ensure_2d_tabular(X_new)

    if X_ref_validated.shape[1] != X_new_validated.shape[1]:
        raise SchemaMismatchError(
            "Reference and new data must have the same number of features. "
            f"Received {X_ref_validated.shape[1]} and {X_new_validated.shape[1]}."
        )

    ref_names = extract_feature_names(X_ref_validated)
    new_names = extract_feature_names(X_new_validated)
    if require_feature_name_match and ref_names != new_names:
        raise SchemaMismatchError(
            "Feature names do not align between reference and new data. "
            "Use `align_tabular_inputs` before fitting."
        )

    ref_schema = TabularSchema(ref_names, infer_feature_types(X_ref_validated), len(ref_names))
    new_schema = TabularSchema(new_names, infer_feature_types(X_new_validated), len(new_names))
    return ref_schema, new_schema


def align_tabular_inputs(
    X_ref: TabularLike,
    X_new: TabularLike,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Align two tabular inputs on the same feature order."""

    X_ref_validated = ensure_2d_tabular(X_ref)
    X_new_validated = ensure_2d_tabular(X_new)

    if isinstance(X_ref_validated, pd.DataFrame) and isinstance(X_new_validated, pd.DataFrame):
        if set(X_ref_validated.columns) != set(X_new_validated.columns):
            raise SchemaMismatchError("Reference and new DataFrames must contain the same columns.")
        ordered = list(X_ref_validated.columns)
        return X_ref_validated.loc[:, ordered], X_new_validated.loc[:, ordered]

    if isinstance(X_ref_validated, pd.DataFrame) != isinstance(X_new_validated, pd.DataFrame):
        raise ValidationError(
            "Reference and new data must both be DataFrames or both be NumPy-like arrays."
        )

    return X_ref_validated, X_new_validated
