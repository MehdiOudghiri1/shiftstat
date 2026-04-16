"""Estimator-style interface for tabular shift detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftstat.detect.results import DatasetShiftSummary, FeatureShiftResult
from shiftstat.detect.statistics import (
    apply_multiple_testing_correction,
    categorical_shift_test,
    classifier_two_sample_test,
    compute_severity_score,
    continuous_shift_test,
)
from shiftstat.exceptions import NotFittedError
from shiftstat.plotting.detect import (
    plot_feature_drift,
    plot_shift_severity_heatmap,
    plot_source_discrimination_roc,
)
from shiftstat.reports import DetectionReport
from shiftstat.typing import TabularLike
from shiftstat.utils.schema import (
    align_tabular_inputs,
    extract_feature_names,
    infer_feature_types,
    validate_tabular_pair_schema,
)


class ShiftDetector:
    """Detect feature-wise and dataset-level distribution shift in tabular data.

    Parameters
    ----------
    alpha:
        Significance level used after any multiple-testing correction.
    multiple_testing:
        One of `{"benjamini-hochberg", "bonferroni", "none"}`.
    categorical_features:
        Optional categorical feature specification. For DataFrames, use column
        names. For arrays, use column indices.
    n_bins:
        Number of bins used for PSI calculations on continuous features.
    random_state:
        Random seed for classifier two-sample testing.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        multiple_testing: str = "benjamini-hochberg",
        categorical_features: list[str] | list[int] | None = None,
        n_bins: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.alpha = alpha
        self.multiple_testing = multiple_testing
        self.categorical_features = categorical_features
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, X_ref: TabularLike, X_new: TabularLike) -> ShiftDetector:
        """Fit the detector on reference and new tabular samples."""

        X_ref_aligned, X_new_aligned = align_tabular_inputs(X_ref, X_new)
        validate_tabular_pair_schema(X_ref_aligned, X_new_aligned)

        self.feature_names_in_ = extract_feature_names(X_ref_aligned)
        self.feature_types_ = infer_feature_types(
            X_ref_aligned,
            categorical_features=self.categorical_features,
        )
        self.X_ref_ = X_ref_aligned
        self.X_new_ = X_new_aligned

        raw_results = []
        for column_index, feature_name in enumerate(self.feature_names_in_):
            if isinstance(X_ref_aligned, pd.DataFrame) and isinstance(X_new_aligned, pd.DataFrame):
                ref_values = X_ref_aligned[feature_name].to_numpy()
                new_values = X_new_aligned[feature_name].to_numpy()
            else:
                ref_values = np.asarray(X_ref_aligned)[:, column_index]
                new_values = np.asarray(X_new_aligned)[:, column_index]

            feature_type = self.feature_types_[feature_name]
            if feature_type == "continuous":
                result = continuous_shift_test(ref_values, new_values, n_bins=self.n_bins)
            else:
                result = categorical_shift_test(ref_values, new_values)
            raw_results.append(
                (feature_name, feature_type, result, len(ref_values), len(new_values))
            )

        adjusted_p_values = apply_multiple_testing_correction(
            [result.p_value for _, _, result, _, _ in raw_results],
            method=self.multiple_testing,
        )

        self.per_feature_results_ = []
        for (feature_name, feature_type, result, reference_size, new_size), adjusted_p_value in zip(
            raw_results,
            adjusted_p_values,
            strict=False,
        ):
            severity = compute_severity_score(
                statistic=result.statistic,
                adjusted_p_value=adjusted_p_value,
                psi=result.psi,
                wasserstein_distance=result.wasserstein_distance,
            )
            self.per_feature_results_.append(
                FeatureShiftResult(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    test_name=result.test_name,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    adjusted_p_value=adjusted_p_value,
                    reject_null=adjusted_p_value <= self.alpha,
                    psi=result.psi,
                    wasserstein_distance=result.wasserstein_distance,
                    severity_score=severity,
                    reference_size=reference_size,
                    new_size=new_size,
                )
            )

        self.per_feature_results_.sort(key=lambda item: item.severity_score, reverse=True)
        self.classifier_result_ = classifier_two_sample_test(
            X_ref_aligned,
            X_new_aligned,
            feature_types=self.feature_types_,
            random_state=self.random_state,
        )
        mean_psi = (
            float(np.mean([item.psi for item in self.per_feature_results_]))
            if self.per_feature_results_
            else 0.0
        )
        max_severity = (
            max(item.severity_score for item in self.per_feature_results_)
            if self.per_feature_results_
            else 0.0
        )
        min_adjusted_p = (
            min(item.adjusted_p_value for item in self.per_feature_results_)
            if self.per_feature_results_
            else 1.0
        )
        n_shifted = sum(item.reject_null for item in self.per_feature_results_)
        self.dataset_summary_ = DatasetShiftSummary(
            n_features=len(self.per_feature_results_),
            n_shifted_features=n_shifted,
            shifted_fraction=n_shifted / max(len(self.per_feature_results_), 1),
            mean_psi=mean_psi,
            max_severity=max_severity,
            min_adjusted_p_value=min_adjusted_p,
            classifier_auc=self.classifier_result_.auc,
            overall_shift_detected=(n_shifted > 0 or self.classifier_result_.auc >= 0.6),
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summary sorted by feature severity."""

        self._check_is_fitted()
        records = [result.to_dict() for result in self.per_feature_results_]
        return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]

    def to_report(self) -> DetectionReport:
        """Return a report abstraction for markdown, dict, and table export."""

        self._check_is_fitted()
        return DetectionReport.from_detector(self)

    def plot(self, kind: str = "feature_drift", **kwargs: Any) -> Any:
        """Dispatch plotting based on a named plot kind."""

        self._check_is_fitted()
        if kind == "feature_drift":
            return plot_feature_drift(self.summary(), **kwargs)
        if kind == "severity_heatmap":
            return plot_shift_severity_heatmap(self.summary(), **kwargs)
        if kind == "roc":
            return plot_source_discrimination_roc(self.classifier_result_, **kwargs)
        raise ValueError(f"Unsupported plot kind: {kind}.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "per_feature_results_"):
            raise NotFittedError("ShiftDetector must be fitted before accessing results.")

    def __repr__(self) -> str:
        fitted = hasattr(self, "per_feature_results_")
        return (
            "ShiftDetector("
            f"alpha={self.alpha}, multiple_testing={self.multiple_testing!r}, "
            f"n_bins={self.n_bins}, fitted={fitted})"
        )
