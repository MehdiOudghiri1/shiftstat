"""Reliability and calibration metrics for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shiftstat.metrics.weighted import weighted_brier_score, weighted_log_loss
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
    labels_from_probabilities,
    predictive_entropy,
)
from shiftstat.utils.validation import ensure_1d, validate_same_length


def _normalize_weights(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = ensure_1d(sample_weight, name="sample_weight").astype(float)
    if len(weights) != n_samples:
        raise ValueError("sample_weight must match the number of samples.")
    return weights


def _bin_edges(probabilities: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError(f"Unsupported binning strategy: {strategy}.")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(probabilities, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    unique_edges = np.unique(edges)
    if len(unique_edges) < 2:
        return np.array([0.0, 1.0], dtype=float)
    return np.asarray(unique_edges, dtype=float)


def probability_bin_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize empirical calibration within probability bins."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    prob_arr = extract_positive_class_probabilities(y_prob)
    validate_same_length(y_true_arr, prob_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    edges = _bin_edges(prob_arr, n_bins=n_bins, strategy=strategy)
    assignments = np.digitize(prob_arr, edges[1:-1], right=True)

    records: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        bin_weight_sum = float(np.sum(bin_weights))
        mean_confidence = float(np.average(prob_arr[mask], weights=bin_weights))
        empirical_probability = float(np.average(y_true_arr[mask], weights=bin_weights))
        records.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "weight_sum": bin_weight_sum,
                "mean_confidence": mean_confidence,
                "empirical_probability": empirical_probability,
                "absolute_gap": float(abs(mean_confidence - empirical_probability)),
                "signed_gap": float(mean_confidence - empirical_probability),
            }
        )
    return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Estimate a calibration curve as bin-wise confidence versus frequency."""

    summary = probability_bin_summary(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=sample_weight,
    )
    selected = summary.loc[
        :,
        ["bin", "mean_confidence", "empirical_probability", "count", "weight_sum"],
    ]
    return selected  # type: ignore[no-any-return]


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute Expected Calibration Error."""

    summary = probability_bin_summary(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=sample_weight,
    )
    if summary.empty:
        return 0.0
    total_weight = float(summary["weight_sum"].sum())
    return float(np.average(summary["absolute_gap"], weights=summary["weight_sum"] / total_weight))


def weighted_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute weighted Expected Calibration Error."""

    return expected_calibration_error(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=sample_weight,
    )


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute Maximum Calibration Error."""

    summary = probability_bin_summary(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=sample_weight,
    )
    if summary.empty:
        return 0.0
    return float(summary["absolute_gap"].max())


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope via logistic regression."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
    prob_arr = extract_positive_class_probabilities(y_prob)
    validate_same_length(y_true_arr, prob_arr)
    if np.unique(y_true_arr).size < 2:
        return float("nan"), float("nan")

    weights = _normalize_weights(sample_weight, len(y_true_arr))
    logit = np.log(prob_arr / (1.0 - prob_arr)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, y_true_arr, sample_weight=weights)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    return intercept, slope


def predictive_entropy_summary(
    y_prob: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize predictive entropy under optional sample weighting."""

    entropy = predictive_entropy(y_prob)
    weights = _normalize_weights(sample_weight, len(entropy))
    centered = entropy - np.average(entropy, weights=weights)
    return {
        "mean": float(np.average(entropy, weights=weights)),
        "std": float(np.sqrt(np.average(centered**2, weights=weights))),
        "min": float(np.min(entropy)),
        "max": float(np.max(entropy)),
    }


def confidence_conditioned_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize error rates across confidence buckets."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
    prob_arr = extract_positive_class_probabilities(y_prob)
    validate_same_length(y_true_arr, prob_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    confidence = confidence_from_probabilities(prob_arr)
    errors = (labels_from_probabilities(prob_arr) != y_true_arr).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    assignments = np.digitize(confidence, edges[1:-1], right=True)

    records: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        records.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "weight_sum": float(np.sum(bin_weights)),
                "mean_confidence": float(np.average(confidence[mask], weights=bin_weights)),
                "error_rate": float(np.average(errors[mask], weights=bin_weights)),
            }
        )
    return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]


def uncertainty_bucket_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 5,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize performance across predictive-entropy buckets."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
    prob_arr = extract_positive_class_probabilities(y_prob)
    validate_same_length(y_true_arr, prob_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    entropy = predictive_entropy(prob_arr)
    confidence = confidence_from_probabilities(prob_arr)
    errors = (labels_from_probabilities(prob_arr) != y_true_arr).astype(float)
    edges = np.quantile(entropy, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([float(np.min(entropy)), float(np.max(entropy) + 1e-12)], dtype=float)
    assignments = np.digitize(entropy, edges[1:-1], right=True)

    records: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        records.append(
            {
                "bin": index,
                "lower_entropy": float(edges[index]),
                "upper_entropy": float(edges[index + 1]),
                "count": int(np.sum(mask)),
                "weight_sum": float(np.sum(bin_weights)),
                "mean_entropy": float(np.average(entropy[mask], weights=bin_weights)),
                "mean_confidence": float(np.average(confidence[mask], weights=bin_weights)),
                "error_rate": float(np.average(errors[mask], weights=bin_weights)),
            }
        )
    return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]


def negative_log_likelihood(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute negative log likelihood for binary classification."""

    probabilities = extract_positive_class_probabilities(y_prob)
    return weighted_log_loss(y_true, probabilities, sample_weight=sample_weight)


def brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Approximate the Murphy decomposition of the Brier score."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
    prob_arr = extract_positive_class_probabilities(y_prob)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    summary = probability_bin_summary(
        y_true_arr,
        prob_arr,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=weights,
    )
    if summary.empty:
        return {
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
        }

    total_weight = float(np.sum(weights))
    prevalence = float(np.average(y_true_arr, weights=weights))
    reliability = float(
        np.sum(
            summary["weight_sum"]
            * (summary["mean_confidence"] - summary["empirical_probability"]) ** 2
        )
        / total_weight
    )
    resolution = float(
        np.sum(summary["weight_sum"] * (summary["empirical_probability"] - prevalence) ** 2)
        / total_weight
    )
    uncertainty = float(prevalence * (1.0 - prevalence))
    return {
        "brier_score": weighted_brier_score(y_true_arr, prob_arr, sample_weight=weights),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a compact bundle of calibration metrics."""

    intercept, slope = calibration_slope_intercept(y_true, y_prob, sample_weight=sample_weight)
    decomposition = brier_decomposition(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        sample_weight=sample_weight,
    )
    prob_arr = extract_positive_class_probabilities(y_prob)
    return {
        "ece": expected_calibration_error(
            y_true,
            prob_arr,
            n_bins=n_bins,
            strategy=strategy,
            sample_weight=sample_weight,
        ),
        "mce": maximum_calibration_error(
            y_true,
            prob_arr,
            n_bins=n_bins,
            strategy=strategy,
            sample_weight=sample_weight,
        ),
        "negative_log_likelihood": negative_log_likelihood(
            y_true,
            prob_arr,
            sample_weight=sample_weight,
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        **decomposition,
    }
