"""Importance weighting estimators and utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.exceptions import NotFittedError
from shiftstat.metrics import (
    compute_effective_sample_size,
    weighted_accuracy,
    weighted_brier_score,
    weighted_log_loss,
    weighted_mae,
    weighted_mse,
)
from shiftstat.plotting.reweight import plot_effective_sample_size, plot_importance_weight_histogram
from shiftstat.reports import ReweightingReport
from shiftstat.typing import FeatureTypes, TabularLike
from shiftstat.utils.random import random_state_to_int
from shiftstat.utils.schema import (
    align_tabular_inputs,
    extract_feature_names,
    infer_feature_types,
    validate_tabular_pair_schema,
)
from shiftstat.utils.validation import ensure_1d, validate_same_length


def _to_dataframe(X: TabularLike) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()  # type: ignore[no-any-return]
    return pd.DataFrame(np.asarray(X), columns=extract_feature_names(X))  # type: ignore[no-any-return]


def _build_preprocessor(feature_types: FeatureTypes) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    categorical_columns = [name for name, kind in feature_types.items() if kind == "categorical"]
    continuous_columns = [name for name, kind in feature_types.items() if kind == "continuous"]
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )
    if continuous_columns:
        transformers.append(("continuous", "passthrough", continuous_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Compute a weighted empirical mean."""

    values_arr = ensure_1d(values, name="values").astype(float)
    if sample_weight is None:
        return float(np.mean(values_arr))
    weights = ensure_1d(sample_weight, name="sample_weight").astype(float)
    validate_same_length(values_arr, weights)
    return float(np.average(values_arr, weights=weights))


def weighted_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    loss: str | Callable[[np.ndarray, np.ndarray], np.ndarray] = "mse",
) -> float:
    """Compute weighted empirical risk for a named or callable loss."""

    y_true_arr = ensure_1d(y_true, name="y_true")
    y_pred_arr = ensure_1d(y_pred, name="y_pred")
    validate_same_length(y_true_arr, y_pred_arr)

    if callable(loss):
        losses = np.asarray(loss(y_true_arr, y_pred_arr), dtype=float)
        if losses.ndim != 1:
            raise ValueError("Callable loss must return a one-dimensional loss array.")
        validate_same_length(losses, y_true_arr)
        return weighted_mean(losses, sample_weight)

    if loss == "mse":
        return weighted_mse(y_true_arr, y_pred_arr, sample_weight=sample_weight)
    if loss == "mae":
        return weighted_mae(y_true_arr, y_pred_arr, sample_weight=sample_weight)
    if loss == "log_loss":
        return weighted_log_loss(y_true_arr, y_pred_arr, sample_weight=sample_weight)
    if loss == "brier":
        return weighted_brier_score(y_true_arr, y_pred_arr, sample_weight=sample_weight)
    if loss == "accuracy":
        return 1.0 - weighted_accuracy(y_true_arr, y_pred_arr, sample_weight=sample_weight)
    raise ValueError(f"Unsupported loss specification: {loss}.")


class ImportanceWeighter:
    """Estimate covariate-shift importance weights via domain discrimination.

    Parameters
    ----------
    method:
        Weighting strategy. Use `"domain_classifier"` for a flexible random-forest
        classifier or `"logistic"` for a logistic baseline.
    categorical_features:
        Optional categorical feature specification.
    clip_min:
        Lower bound for weight clipping.
    clip_max:
        Upper bound for weight clipping.
    normalize:
        If `True`, normalize weights to have mean one.
    random_state:
        Reproducible seed for the weighting model and evaluation split.
    """

    def __init__(
        self,
        *,
        method: str = "domain_classifier",
        categorical_features: list[str] | list[int] | None = None,
        clip_min: float = 1e-3,
        clip_max: float = 20.0,
        normalize: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.method = method
        self.categorical_features = categorical_features
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize = normalize
        self.random_state = random_state

    def fit(self, X_ref: TabularLike, X_target: TabularLike) -> ImportanceWeighter:
        """Fit a domain classifier used to estimate importance weights."""

        X_ref_aligned, X_target_aligned = align_tabular_inputs(X_ref, X_target)
        validate_tabular_pair_schema(X_ref_aligned, X_target_aligned)

        self.X_ref_ = X_ref_aligned
        self.X_target_ = X_target_aligned
        self.feature_names_in_ = extract_feature_names(X_ref_aligned)
        self.feature_types_ = infer_feature_types(
            X_ref_aligned,
            categorical_features=self.categorical_features,
        )

        ref_df = _to_dataframe(X_ref_aligned)
        target_df = _to_dataframe(X_target_aligned)
        combined = pd.concat([ref_df, target_df], axis=0, ignore_index=True)
        y_domain = np.concatenate(
            [np.zeros(len(ref_df), dtype=int), np.ones(len(target_df), dtype=int)]
        )
        self.prior_ref_ = len(ref_df) / len(combined)
        self.prior_target_ = len(target_df) / len(combined)

        estimator = self._make_estimator()
        preprocessor = _build_preprocessor(self.feature_types_)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        self.model_ = clone(pipeline)
        self.model_.fit(combined, y_domain)

        seed = random_state_to_int(self.random_state)
        eval_model = clone(pipeline)
        X_train, X_test, y_train, y_test = train_test_split(
            combined,
            y_domain,
            stratify=y_domain,
            test_size=0.3,
            random_state=seed,
        )
        eval_model.fit(X_train, y_train)
        scores = eval_model.predict_proba(X_test)[:, 1]
        self.source_auc_ = float(roc_auc_score(y_test, scores))

        self.reference_weights_ = self.predict_weights()
        self.effective_sample_size_ = compute_effective_sample_size(self.reference_weights_)
        return self

    def predict_weights(self, X: TabularLike | None = None) -> np.ndarray:
        """Estimate target-over-reference importance weights."""

        self._check_is_fitted()
        X_input = self.X_ref_ if X is None else X
        X_df = _to_dataframe(X_input)
        target_probability = np.clip(self.model_.predict_proba(X_df)[:, 1], 1e-6, 1 - 1e-6)
        odds = target_probability / (1 - target_probability)
        weights = odds * (self.prior_ref_ / self.prior_target_)
        weights = np.clip(weights, self.clip_min, self.clip_max)
        if self.normalize:
            weights = weights / np.mean(weights)
        return np.asarray(weights, dtype=float)

    def fit_predict(self, X_ref: TabularLike, X_target: TabularLike) -> np.ndarray:
        """Fit the weighting model and return reference-domain weights."""

        return self.fit(X_ref, X_target).predict_weights()

    def summary(self) -> dict[str, float | str]:
        """Return a lightweight dictionary summary of the fitted weighting model."""

        self._check_is_fitted()
        return {
            "method": self.method,
            "n_reference_samples": float(len(self.reference_weights_)),
            "mean_weight": float(np.mean(self.reference_weights_)),
            "std_weight": float(np.std(self.reference_weights_)),
            "max_weight": float(np.max(self.reference_weights_)),
            "effective_sample_size": float(self.effective_sample_size_),
            "source_auc": float(self.source_auc_),
        }

    def to_report(self) -> ReweightingReport:
        """Return a report abstraction for the fitted importance weights."""

        self._check_is_fitted()
        return ReweightingReport.from_weighter(self)

    def plot(self, kind: str = "histogram", **kwargs: Any) -> Any:
        """Dispatch plotting for weighting diagnostics."""

        self._check_is_fitted()
        if kind == "histogram":
            return plot_importance_weight_histogram(self.reference_weights_, **kwargs)
        if kind == "effective_sample_size":
            return plot_effective_sample_size(
                original_sample_size=len(self.reference_weights_),
                effective_sample_size=self.effective_sample_size_,
                **kwargs,
            )
        raise ValueError(f"Unsupported plot kind: {kind}.")

    def _make_estimator(self) -> Any:
        seed = random_state_to_int(self.random_state)
        if self.method == "domain_classifier":
            return RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=seed,
            )
        if self.method == "logistic":
            return LogisticRegression(max_iter=3000, random_state=seed)
        raise ValueError(f"Unsupported weighting method: {self.method}.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("ImportanceWeighter must be fitted before use.")

    def __repr__(self) -> str:
        fitted = hasattr(self, "model_")
        return (
            "ImportanceWeighter("
            f"method={self.method!r}, clip_min={self.clip_min}, clip_max={self.clip_max}, "
            f"normalize={self.normalize}, fitted={fitted})"
        )
