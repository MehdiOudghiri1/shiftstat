"""Weighted metrics used throughout ShiftStat."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from shiftstat.utils.validation import ensure_1d, validate_same_length


def _normalize_weights(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = ensure_1d(sample_weight, name="sample_weight").astype(float)
    if len(weights) != n_samples:
        raise ValueError("sample_weight must match the number of samples.")
    return weights


def weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute weighted classification accuracy."""

    y_true_arr = ensure_1d(y_true, name="y_true")
    y_pred_arr = ensure_1d(y_pred, name="y_pred")
    validate_same_length(y_true_arr, y_pred_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    predicted_labels = (
        (y_pred_arr >= 0.5).astype(int) if y_pred_arr.dtype.kind in {"f", "c"} else y_pred_arr
    )
    return float(np.average((predicted_labels == y_true_arr).astype(float), weights=weights))


def weighted_log_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Compute weighted binary log loss."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    y_pred_arr = np.clip(
        ensure_1d(y_pred_proba, name="y_pred_proba").astype(float),
        epsilon,
        1 - epsilon,
    )
    validate_same_length(y_true_arr, y_pred_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    losses = -(y_true_arr * np.log(y_pred_arr) + (1 - y_true_arr) * np.log(1 - y_pred_arr))
    return float(np.average(losses, weights=weights))


def weighted_brier_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute weighted Brier score for binary probabilities."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    y_pred_arr = ensure_1d(y_pred_proba, name="y_pred_proba").astype(float)
    validate_same_length(y_true_arr, y_pred_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    losses = (y_pred_arr - y_true_arr) ** 2
    return float(np.average(losses, weights=weights))


def weighted_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute weighted mean squared error."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    y_pred_arr = ensure_1d(y_pred, name="y_pred").astype(float)
    validate_same_length(y_true_arr, y_pred_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    return float(np.average((y_pred_arr - y_true_arr) ** 2, weights=weights))


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute weighted mean absolute error."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    y_pred_arr = ensure_1d(y_pred, name="y_pred").astype(float)
    validate_same_length(y_true_arr, y_pred_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    return float(np.average(np.abs(y_pred_arr - y_true_arr), weights=weights))


def compute_effective_sample_size(sample_weight: np.ndarray) -> float:
    """Compute Kish effective sample size."""

    weights = ensure_1d(sample_weight, name="sample_weight").astype(float)
    numerator = np.sum(weights) ** 2
    denominator = np.sum(weights**2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def source_separability_score(source_labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute dataset source separability as ROC AUC."""

    return float(roc_auc_score(ensure_1d(source_labels), ensure_1d(scores)))
