"""Reliability diagnostics under reference-to-target shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftstat.calibration.evaluator import CalibrationEvaluator, compare_calibration
from shiftstat.calibration.results import CalibrationResult
from shiftstat.metrics import (
    confidence_conditioned_error,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    predictive_entropy_summary,
    probability_bin_summary,
    uncertainty_bucket_summary,
    weighted_accuracy,
    weighted_brier_score,
)
from shiftstat.plotting.calibration import plot_calibration_comparison
from shiftstat.plotting.reliability import plot_confidence_error_curve
from shiftstat.reliability.results import (
    ReliabilityDegradationSummary,
    ReliabilityProfile,
    ReliabilityShiftReport,
)
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
)
from shiftstat.utils.validation import ensure_1d, validate_same_length


class ReliabilityAnalyzer:
    """Analyze predictive reliability degradation between reference and target data."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.calibration_evaluator = CalibrationEvaluator(n_bins=n_bins, strategy=strategy)

    def fit(
        self,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        y_target: np.ndarray,
        p_target: np.ndarray,
        *,
        reference_weight: np.ndarray | None = None,
        target_weight: np.ndarray | None = None,
        weighted_reference_name: str = "reference_weighted",
    ) -> ReliabilityAnalyzer:
        """Fit reliability diagnostics for reference and target predictions."""

        self.reference_profile_ = self._build_profile(
            y_ref,
            p_ref,
            sample_weight=None,
            name="reference",
        )
        self.target_profile_ = self._build_profile(
            y_target,
            p_target,
            sample_weight=target_weight,
            name="target",
        )
        self.weighted_reference_profile_ = None
        if reference_weight is not None:
            self.weighted_reference_profile_ = self._build_profile(
                y_ref,
                p_ref,
                sample_weight=reference_weight,
                name=weighted_reference_name,
            )

        calibration_results = {
            "reference": self._profile_calibration_result(self.reference_profile_),
            "target": self._profile_calibration_result(self.target_profile_),
        }
        if self.weighted_reference_profile_ is not None:
            calibration_results["reference_weighted"] = self._profile_calibration_result(
                self.weighted_reference_profile_
            )
        self.calibration_comparison_ = compare_calibration(calibration_results)
        self.degradation_summary_ = self._summarize_degradation(
            self.reference_profile_,
            self.target_profile_,
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return a compact degradation summary."""

        self._check_is_fitted()
        return pd.DataFrame.from_records([self.degradation_summary_.to_dict()])

    def to_report(
        self,
        *,
        dataset_shift_overview: dict[str, Any] | None = None,
        recalibrated_target_profile: ReliabilityProfile | None = None,
        weighting_summary: dict[str, Any] | None = None,
        recalibration_summary: dict[str, Any] | None = None,
    ) -> ReliabilityShiftReport:
        """Return a markdown- and machine-readable reliability report."""

        self._check_is_fitted()
        return ReliabilityShiftReport(
            dataset_shift_overview=dataset_shift_overview,
            reference_profile=self.reference_profile_,
            target_profile=self.target_profile_,
            degradation_summary=self.degradation_summary_,
            weighted_reference_profile=self.weighted_reference_profile_,
            recalibrated_target_profile=recalibrated_target_profile,
            weighting_summary=weighting_summary,
            recalibration_summary=recalibration_summary,
            calibration_comparison=_frame_records(self.calibration_comparison_),
        )

    def plot(self, kind: str = "diagram", **kwargs: Any) -> Any:
        """Render reliability diagnostics."""

        self._check_is_fitted()
        if kind == "diagram":
            return plot_calibration_comparison(
                self.reference_profile_,
                self.target_profile_,
                **kwargs,
            )
        if kind == "confidence_error":
            return plot_confidence_error_curve(self.target_profile_, **kwargs)
        raise ValueError(f"Unsupported plot kind: {kind}.")

    def _build_profile(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        sample_weight: np.ndarray | None,
        name: str,
    ) -> ReliabilityProfile:
        y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
        prob_arr = extract_positive_class_probabilities(y_prob)
        validate_same_length(y_true_arr, prob_arr)

        calibration_result = self.calibration_evaluator.evaluate(
            y_true_arr,
            prob_arr,
            sample_weight=sample_weight,
            name=name,
        )
        confidence = confidence_from_probabilities(prob_arr)
        entropy_summary = predictive_entropy_summary(prob_arr, sample_weight=sample_weight)
        accuracy = weighted_accuracy(y_true_arr, prob_arr, sample_weight=sample_weight)

        return ReliabilityProfile(
            name=name,
            n_samples=len(y_true_arr),
            weighted=sample_weight is not None,
            accuracy=float(accuracy),
            error_rate=float(1.0 - accuracy),
            log_loss=float(
                negative_log_likelihood(y_true_arr, prob_arr, sample_weight=sample_weight)
            ),
            brier_score=float(
                weighted_brier_score(y_true_arr, prob_arr, sample_weight=sample_weight)
            ),
            ece=float(
                expected_calibration_error(
                    y_true_arr,
                    prob_arr,
                    n_bins=self.n_bins,
                    strategy=self.strategy,
                    sample_weight=sample_weight,
                )
            ),
            mce=float(
                maximum_calibration_error(
                    y_true_arr,
                    prob_arr,
                    n_bins=self.n_bins,
                    strategy=self.strategy,
                    sample_weight=sample_weight,
                )
            ),
            calibration_intercept=float(calibration_result.calibration_intercept),
            calibration_slope=float(calibration_result.calibration_slope),
            mean_confidence=float(np.average(confidence, weights=sample_weight)),
            mean_outcome=float(np.average(y_true_arr, weights=sample_weight)),
            mean_entropy=float(entropy_summary["mean"]),
            confidence_gap=float(np.average(confidence, weights=sample_weight) - accuracy),
            calibration_table=probability_bin_summary(
                y_true_arr,
                prob_arr,
                n_bins=self.n_bins,
                strategy=self.strategy,
                sample_weight=sample_weight,
            ).pipe(_frame_records),
            confidence_error_table=confidence_conditioned_error(
                y_true_arr,
                prob_arr,
                n_bins=self.n_bins,
                sample_weight=sample_weight,
            ).pipe(_frame_records),
            uncertainty_table=uncertainty_bucket_summary(
                y_true_arr,
                prob_arr,
                sample_weight=sample_weight,
            ).pipe(_frame_records),
        )

    def _profile_calibration_result(self, profile: ReliabilityProfile) -> CalibrationResult:
        return CalibrationResult(
            name=profile.name,
            n_samples=profile.n_samples,
            weighted=profile.weighted,
            ece=profile.ece,
            mce=profile.mce,
            brier_score=profile.brier_score,
            negative_log_likelihood=profile.log_loss,
            calibration_intercept=profile.calibration_intercept,
            calibration_slope=profile.calibration_slope,
            mean_confidence=profile.mean_confidence,
            mean_outcome=profile.mean_outcome,
            brier_reliability=float("nan"),
            brier_resolution=float("nan"),
            brier_uncertainty=float("nan"),
            bin_summary=profile.calibration_table,
        )

    def _summarize_degradation(
        self,
        reference: ReliabilityProfile,
        target: ReliabilityProfile,
    ) -> ReliabilityDegradationSummary:
        delta_accuracy = float(target.accuracy - reference.accuracy)
        delta_error_rate = float(target.error_rate - reference.error_rate)
        delta_log_loss = float(target.log_loss - reference.log_loss)
        delta_brier = float(target.brier_score - reference.brier_score)
        delta_ece = float(target.ece - reference.ece)
        delta_mce = float(target.mce - reference.mce)
        delta_confidence = float(target.mean_confidence - reference.mean_confidence)
        delta_gap = float(target.confidence_gap - reference.confidence_gap)
        delta_entropy = float(target.mean_entropy - reference.mean_entropy)

        high_confidence_bins = target.confidence_frame()
        high_confidence_error = False
        if not high_confidence_bins.empty:
            mask = high_confidence_bins["mean_confidence"] >= 0.8
            if mask.any():
                high_confidence_error = bool(
                    (high_confidence_bins.loc[mask, "error_rate"] > 0.2).any()
                )

        return ReliabilityDegradationSummary(
            reference_name=reference.name,
            target_name=target.name,
            delta_accuracy=delta_accuracy,
            delta_error_rate=delta_error_rate,
            delta_log_loss=delta_log_loss,
            delta_brier_score=delta_brier,
            delta_ece=delta_ece,
            delta_mce=delta_mce,
            delta_mean_confidence=delta_confidence,
            delta_confidence_gap=delta_gap,
            delta_mean_entropy=delta_entropy,
            fragility_flags={
                "calibration_degradation": delta_ece > 0.02,
                "confidence_inflation": delta_confidence > 0.03 and delta_accuracy < 0.0,
                "loss_degradation": delta_log_loss > 0.05,
                "high_confidence_errors": high_confidence_error,
            },
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "degradation_summary_"):
            raise ValueError("ReliabilityAnalyzer must be fitted before accessing results.")


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to typed JSON-friendly records."""

    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
