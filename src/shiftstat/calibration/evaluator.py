"""Calibration evaluation workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftstat.calibration.results import CalibrationResult
from shiftstat.metrics import (
    brier_decomposition,
    calibration_summary,
    probability_bin_summary,
)
from shiftstat.plotting.calibration import plot_reliability_diagram
from shiftstat.utils.probabilities import extract_positive_class_probabilities


class CalibrationEvaluator:
    """Evaluate binary classification calibration with optional weighting."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        name: str = "dataset",
    ) -> CalibrationResult:
        """Evaluate calibration for a single set of binary probabilities."""

        prob_arr = extract_positive_class_probabilities(y_prob)
        summary = calibration_summary(
            y_true,
            prob_arr,
            n_bins=self.n_bins,
            strategy=self.strategy,
            sample_weight=sample_weight,
        )
        curve = probability_bin_summary(
            y_true,
            prob_arr,
            n_bins=self.n_bins,
            strategy=self.strategy,
            sample_weight=sample_weight,
        )
        decomposition = brier_decomposition(
            y_true,
            prob_arr,
            n_bins=self.n_bins,
            strategy=self.strategy,
            sample_weight=sample_weight,
        )
        return CalibrationResult(
            name=name,
            n_samples=len(prob_arr),
            weighted=sample_weight is not None,
            ece=float(summary["ece"]),
            mce=float(summary["mce"]),
            brier_score=float(summary["brier_score"]),
            negative_log_likelihood=float(summary["negative_log_likelihood"]),
            calibration_intercept=float(summary["calibration_intercept"]),
            calibration_slope=float(summary["calibration_slope"]),
            mean_confidence=float(np.average(prob_arr, weights=sample_weight)),
            mean_outcome=float(np.average(np.asarray(y_true, dtype=float), weights=sample_weight)),
            brier_reliability=float(decomposition["reliability"]),
            brier_resolution=float(decomposition["resolution"]),
            brier_uncertainty=float(decomposition["uncertainty"]),
            bin_summary=_frame_records(curve),
        )

    def compare(self, results: dict[str, CalibrationResult]) -> pd.DataFrame:
        """Compare multiple calibration results in a single table."""

        return compare_calibration(results)

    def plot(self, result: CalibrationResult, *, kind: str = "diagram", **kwargs: Any) -> Any:
        """Render a calibration plot for a fitted result object."""

        if kind == "diagram":
            return plot_reliability_diagram(result, **kwargs)
        raise ValueError(f"Unsupported plot kind: {kind}.")


def compare_calibration(results: dict[str, CalibrationResult]) -> pd.DataFrame:
    """Compare multiple calibration results in a tabular summary."""

    records = []
    for name, result in results.items():
        records.append(
            {
                "name": name,
                "ece": result.ece,
                "mce": result.mce,
                "brier_score": result.brier_score,
                "negative_log_likelihood": result.negative_log_likelihood,
                "calibration_intercept": result.calibration_intercept,
                "calibration_slope": result.calibration_slope,
                "mean_confidence": result.mean_confidence,
                "mean_outcome": result.mean_outcome,
                "weighted": result.weighted,
            }
        )
    return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to typed JSON-friendly records."""

    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
