"""Metrics for selective prediction and abstention under shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftstat.metrics.reliability import expected_calibration_error, maximum_calibration_error
from shiftstat.metrics.weighted import (
    weighted_accuracy,
    weighted_brier_score,
    weighted_log_loss,
)
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
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


def _selection_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
    prob_arr = extract_positive_class_probabilities(y_prob)
    mask_arr = ensure_1d(accepted_mask, name="accepted_mask").astype(bool)
    validate_same_length(y_true_arr, prob_arr, mask_arr)
    weights = _normalize_weights(sample_weight, len(y_true_arr))
    return y_true_arr, prob_arr, mask_arr, weights


def retained_coverage(
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute retained coverage after abstention."""

    mask_arr = ensure_1d(accepted_mask, name="accepted_mask").astype(bool)
    weights = _normalize_weights(sample_weight, len(mask_arr))
    return float(np.sum(weights[mask_arr]) / max(np.sum(weights), 1e-12))


def abstention_rate(
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute the abstention rate."""

    return float(1.0 - retained_coverage(accepted_mask, sample_weight=sample_weight))


def selective_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute accuracy on the accepted subset."""

    y_true_arr, prob_arr, mask_arr, weights = _selection_inputs(
        y_true,
        y_prob,
        accepted_mask,
        sample_weight=sample_weight,
    )
    if not np.any(mask_arr):
        return float("nan")
    return float(
        weighted_accuracy(
            y_true_arr[mask_arr],
            prob_arr[mask_arr],
            sample_weight=weights[mask_arr],
        )
    )


def selective_risk(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    loss: str = "error_rate",
) -> float:
    """Compute selective risk on the accepted subset."""

    y_true_arr, prob_arr, mask_arr, weights = _selection_inputs(
        y_true,
        y_prob,
        accepted_mask,
        sample_weight=sample_weight,
    )
    if not np.any(mask_arr):
        return float("nan")

    if loss == "error_rate":
        return float(
            1.0
            - weighted_accuracy(
                y_true_arr[mask_arr],
                prob_arr[mask_arr],
                sample_weight=weights[mask_arr],
            )
        )
    if loss == "log_loss":
        return float(
            weighted_log_loss(
                y_true_arr[mask_arr],
                prob_arr[mask_arr],
                sample_weight=weights[mask_arr],
            )
        )
    if loss == "brier_score":
        return float(
            weighted_brier_score(
                y_true_arr[mask_arr],
                prob_arr[mask_arr],
                sample_weight=weights[mask_arr],
            )
        )
    raise ValueError("loss must be one of {'error_rate', 'log_loss', 'brier_score'}.")


def selective_calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute calibration summaries on the accepted subset."""

    y_true_arr, prob_arr, mask_arr, weights = _selection_inputs(
        y_true,
        y_prob,
        accepted_mask,
        sample_weight=sample_weight,
    )
    if not np.any(mask_arr):
        return {"ece": float("nan"), "mce": float("nan")}
    return {
        "ece": float(
            expected_calibration_error(
                y_true_arr[mask_arr],
                prob_arr[mask_arr],
                n_bins=n_bins,
                strategy=strategy,
                sample_weight=weights[mask_arr],
            )
        ),
        "mce": float(
            maximum_calibration_error(
                y_true_arr[mask_arr],
                prob_arr[mask_arr],
                n_bins=n_bins,
                strategy=strategy,
                sample_weight=weights[mask_arr],
            )
        ),
    }


def selective_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, float]:
    """Compute a compact summary of selective prediction metrics."""

    y_true_arr, prob_arr, mask_arr, weights = _selection_inputs(
        y_true,
        y_prob,
        accepted_mask,
        sample_weight=sample_weight,
    )
    confidence = confidence_from_probabilities(prob_arr)
    entropy = predictive_entropy(prob_arr)
    coverage = retained_coverage(mask_arr, sample_weight=weights)
    calibration = selective_calibration_summary(
        y_true_arr,
        prob_arr,
        mask_arr,
        sample_weight=weights,
        n_bins=n_bins,
        strategy=strategy,
    )

    if np.any(mask_arr):
        accepted_weights = weights[mask_arr]
        selective_log = weighted_log_loss(
            y_true_arr[mask_arr],
            prob_arr[mask_arr],
            sample_weight=accepted_weights,
        )
        selective_brier = weighted_brier_score(
            y_true_arr[mask_arr],
            prob_arr[mask_arr],
            sample_weight=accepted_weights,
        )
        mean_confidence_accepted = float(np.average(confidence[mask_arr], weights=accepted_weights))
        mean_entropy_accepted = float(np.average(entropy[mask_arr], weights=accepted_weights))
    else:
        selective_log = float("nan")
        selective_brier = float("nan")
        mean_confidence_accepted = float("nan")
        mean_entropy_accepted = float("nan")

    rejected_mask = ~mask_arr
    if np.any(rejected_mask):
        rejected_weights = weights[rejected_mask]
        mean_confidence_rejected = float(
            np.average(confidence[rejected_mask], weights=rejected_weights)
        )
        mean_entropy_rejected = float(np.average(entropy[rejected_mask], weights=rejected_weights))
    else:
        mean_confidence_rejected = float("nan")
        mean_entropy_rejected = float("nan")

    full_accuracy = weighted_accuracy(y_true_arr, prob_arr, sample_weight=weights)
    full_log_loss = weighted_log_loss(y_true_arr, prob_arr, sample_weight=weights)
    full_brier_score = weighted_brier_score(y_true_arr, prob_arr, sample_weight=weights)
    full_calibration = {
        "ece": expected_calibration_error(
            y_true_arr,
            prob_arr,
            n_bins=n_bins,
            strategy=strategy,
            sample_weight=weights,
        ),
        "mce": maximum_calibration_error(
            y_true_arr,
            prob_arr,
            n_bins=n_bins,
            strategy=strategy,
            sample_weight=weights,
        ),
    }

    accepted_weight = float(np.sum(weights[mask_arr]))
    rejected_weight = float(np.sum(weights[rejected_mask]))
    return {
        "coverage": coverage,
        "abstention_rate": float(1.0 - coverage),
        "accepted_samples": float(np.sum(mask_arr)),
        "rejected_samples": float(np.sum(rejected_mask)),
        "accepted_weight": accepted_weight,
        "rejected_weight": rejected_weight,
        "selective_accuracy": float(
            selective_accuracy(y_true_arr, prob_arr, mask_arr, sample_weight=weights)
        ),
        "selective_risk": float(
            selective_risk(y_true_arr, prob_arr, mask_arr, sample_weight=weights)
        ),
        "selective_log_loss": float(selective_log),
        "selective_brier_score": float(selective_brier),
        "selective_ece": float(calibration["ece"]),
        "selective_mce": float(calibration["mce"]),
        "full_accuracy": float(full_accuracy),
        "full_risk": float(1.0 - full_accuracy),
        "full_log_loss": float(full_log_loss),
        "full_brier_score": float(full_brier_score),
        "full_ece": float(full_calibration["ece"]),
        "full_mce": float(full_calibration["mce"]),
        "risk_reduction": float((1.0 - full_accuracy) - selective_risk(y_true_arr, prob_arr, mask_arr, sample_weight=weights)),
        "log_loss_reduction": float(full_log_loss - selective_log),
        "ece_reduction": float(full_calibration["ece"] - calibration["ece"]),
        "mean_confidence_accepted": mean_confidence_accepted,
        "mean_confidence_rejected": mean_confidence_rejected,
        "mean_entropy_accepted": mean_entropy_accepted,
        "mean_entropy_rejected": mean_entropy_rejected,
    }


def candidate_thresholds(
    selection_scores: np.ndarray,
    *,
    max_points: int = 101,
) -> np.ndarray:
    """Construct candidate thresholds for score-based abstention policies."""

    scores = ensure_1d(selection_scores, name="selection_scores").astype(float)
    unique_scores = np.unique(scores)
    if len(unique_scores) <= max_points:
        thresholds = unique_scores
    else:
        quantiles = np.linspace(0.0, 1.0, max_points)
        thresholds = np.unique(np.quantile(scores, quantiles))
    thresholds = thresholds.astype(float)
    full_coverage_threshold = float(np.min(scores) - 1e-12)
    return np.unique(np.concatenate([[full_coverage_threshold], thresholds]))


def risk_coverage_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selection_scores: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    max_points: int = 101,
) -> pd.DataFrame:
    """Compute a risk-coverage table over score thresholds."""

    scores = ensure_1d(selection_scores, name="selection_scores").astype(float)
    y_true_arr = ensure_1d(y_true, name="y_true")
    prob_arr = extract_positive_class_probabilities(y_prob)
    validate_same_length(y_true_arr, prob_arr, scores)
    weights = _normalize_weights(sample_weight, len(scores))

    threshold_values = candidate_thresholds(scores, max_points=max_points) if thresholds is None else np.asarray(thresholds, dtype=float)
    records: list[dict[str, Any]] = []
    for threshold in np.sort(np.unique(threshold_values)):
        accepted = scores >= float(threshold)
        summary = selective_summary(
            y_true_arr,
            prob_arr,
            accepted,
            sample_weight=weights,
            n_bins=n_bins,
            strategy=strategy,
        )
        records.append(
            {
                "threshold": float(threshold),
                "coverage": float(summary["coverage"]),
                "abstention_rate": float(summary["abstention_rate"]),
                "accepted_samples": float(summary["accepted_samples"]),
                "rejected_samples": float(summary["rejected_samples"]),
                "selective_accuracy": float(summary["selective_accuracy"]),
                "selective_risk": float(summary["selective_risk"]),
                "selective_log_loss": float(summary["selective_log_loss"]),
                "selective_brier_score": float(summary["selective_brier_score"]),
                "selective_ece": float(summary["selective_ece"]),
                "selective_mce": float(summary["selective_mce"]),
            }
        )

    frame = pd.DataFrame.from_records(records).sort_values("coverage", ascending=False)
    frame = frame.drop_duplicates(subset=["coverage"], keep="first").reset_index(drop=True)
    return frame  # type: ignore[no-any-return]
