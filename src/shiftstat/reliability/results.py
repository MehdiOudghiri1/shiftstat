"""Result containers for reliability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReliabilityProfile:
    """Reliability profile for a binary classifier on one distribution."""

    name: str
    n_samples: int
    weighted: bool
    accuracy: float
    error_rate: float
    log_loss: float
    brier_score: float
    ece: float
    mce: float
    calibration_intercept: float
    calibration_slope: float
    mean_confidence: float
    mean_outcome: float
    mean_entropy: float
    confidence_gap: float
    calibration_table: list[dict[str, Any]]
    confidence_error_table: list[dict[str, Any]]
    uncertainty_table: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "weighted": self.weighted,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "mce": self.mce,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "mean_confidence": self.mean_confidence,
            "mean_outcome": self.mean_outcome,
            "mean_entropy": self.mean_entropy,
            "confidence_gap": self.confidence_gap,
            "calibration_table": self.calibration_table,
            "confidence_error_table": self.confidence_error_table,
            "uncertainty_table": self.uncertainty_table,
        }

    def calibration_frame(self) -> pd.DataFrame:
        """Return the calibration table as a DataFrame."""

        return pd.DataFrame.from_records(self.calibration_table)  # type: ignore[no-any-return]

    def confidence_frame(self) -> pd.DataFrame:
        """Return the confidence-conditioned error table as a DataFrame."""

        return pd.DataFrame.from_records(self.confidence_error_table)  # type: ignore[no-any-return]

    def uncertainty_frame(self) -> pd.DataFrame:
        """Return the uncertainty bucket table as a DataFrame."""

        return pd.DataFrame.from_records(self.uncertainty_table)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class ReliabilityDegradationSummary:
    """Summary of reliability degradation from reference to target."""

    reference_name: str
    target_name: str
    delta_accuracy: float
    delta_error_rate: float
    delta_log_loss: float
    delta_brier_score: float
    delta_ece: float
    delta_mce: float
    delta_mean_confidence: float
    delta_confidence_gap: float
    delta_mean_entropy: float
    fragility_flags: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "reference_name": self.reference_name,
            "target_name": self.target_name,
            "delta_accuracy": self.delta_accuracy,
            "delta_error_rate": self.delta_error_rate,
            "delta_log_loss": self.delta_log_loss,
            "delta_brier_score": self.delta_brier_score,
            "delta_ece": self.delta_ece,
            "delta_mce": self.delta_mce,
            "delta_mean_confidence": self.delta_mean_confidence,
            "delta_confidence_gap": self.delta_confidence_gap,
            "delta_mean_entropy": self.delta_mean_entropy,
            "fragility_flags": self.fragility_flags,
        }


@dataclass(frozen=True)
class ReliabilityShiftReport:
    """Markdown- and machine-readable report for reliability under shift."""

    dataset_shift_overview: dict[str, Any] | None
    reference_profile: ReliabilityProfile
    target_profile: ReliabilityProfile
    degradation_summary: ReliabilityDegradationSummary
    weighted_reference_profile: ReliabilityProfile | None = None
    recalibrated_target_profile: ReliabilityProfile | None = None
    weighting_summary: dict[str, Any] | None = None
    recalibration_summary: dict[str, Any] | None = None
    calibration_comparison: list[dict[str, Any]] | None = None

    def to_markdown(self) -> str:
        """Render a scientific markdown report."""

        lines = ["## Reliability under shift", ""]
        if self.dataset_shift_overview is not None:
            shifted_features = self.dataset_shift_overview.get("n_shifted_features", "n/a")
            classifier_auc = self.dataset_shift_overview.get("classifier_auc", float("nan"))
            lines.extend(
                [
                    "### Dataset shift overview",
                    "",
                    f"- Shifted features: {shifted_features}",
                    f"- Classifier two-sample AUC: {classifier_auc:.3f}",
                    "",
                ]
            )
        lines.extend(
            [
                "### Model performance under shift",
                "",
                f"- Reference accuracy: {self.reference_profile.accuracy:.3f}",
                f"- Target accuracy: {self.target_profile.accuracy:.3f}",
                f"- Delta log loss: {self.degradation_summary.delta_log_loss:.3f}",
                f"- Delta Brier score: {self.degradation_summary.delta_brier_score:.3f}",
                "",
                "### Calibration degradation",
                "",
                f"- Reference ECE: {self.reference_profile.ece:.3f}",
                f"- Target ECE: {self.target_profile.ece:.3f}",
                f"- Delta ECE: {self.degradation_summary.delta_ece:.3f}",
                f"- Delta mean confidence: {self.degradation_summary.delta_mean_confidence:.3f}",
                "",
            ]
        )
        if self.weighted_reference_profile is not None and self.weighting_summary is not None:
            effective_n = self.weighting_summary.get("effective_sample_size", float("nan"))
            lines.extend(
                [
                    "### Effect of weighting",
                    "",
                    f"- Weighted reference ECE: {self.weighted_reference_profile.ece:.3f}",
                    f"- Effective sample size: {effective_n:.1f}",
                    "",
                ]
            )
        if self.recalibrated_target_profile is not None and self.recalibration_summary is not None:
            recalibration_method = self.recalibration_summary.get("method", "n/a")
            lines.extend(
                [
                    "### Effect of recalibration",
                    "",
                    f"- Recalibration method: {recalibration_method}",
                    f"- Target ECE after recalibration: {self.recalibrated_target_profile.ece:.3f}",
                    (
                        "- Target log loss after recalibration: "
                        f"{self.recalibrated_target_profile.log_loss:.3f}"
                    ),
                    "",
                ]
            )
        active_flags = [
            name
            for name, is_active in self.degradation_summary.fragility_flags.items()
            if is_active
        ]
        lines.extend(
            [
                "### Interpretation notes",
                "",
                (
                    "- Active fragility indicators: "
                    f"{', '.join(active_flags) if active_flags else 'none'}"
                ),
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "dataset_shift_overview": self.dataset_shift_overview,
            "reference_profile": self.reference_profile.to_dict(),
            "target_profile": self.target_profile.to_dict(),
            "degradation_summary": self.degradation_summary.to_dict(),
            "weighted_reference_profile": (
                None
                if self.weighted_reference_profile is None
                else self.weighted_reference_profile.to_dict()
            ),
            "recalibrated_target_profile": (
                None
                if self.recalibrated_target_profile is None
                else self.recalibrated_target_profile.to_dict()
            ),
            "weighting_summary": self.weighting_summary,
            "recalibration_summary": self.recalibration_summary,
            "calibration_comparison": self.calibration_comparison,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return a compact one-row degradation summary."""

        return pd.DataFrame.from_records([self.degradation_summary.to_dict()])  # type: ignore[no-any-return]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return exportable report tables."""

        tables = {
            "reference_calibration": self.reference_profile.calibration_frame(),
            "target_calibration": self.target_profile.calibration_frame(),
            "degradation_summary": self.to_frame(),
        }
        if self.weighted_reference_profile is not None:
            tables["weighted_reference_calibration"] = (
                self.weighted_reference_profile.calibration_frame()
            )
        if self.recalibrated_target_profile is not None:
            tables["recalibrated_target_calibration"] = (
                self.recalibrated_target_profile.calibration_frame()
            )
        if self.calibration_comparison is not None:
            tables["calibration_comparison"] = pd.DataFrame.from_records(
                self.calibration_comparison
            )
        return tables


@dataclass(frozen=True)
class ShiftEvaluationResult:
    """Structured output of a model evaluation workflow under shift."""

    estimator_name: str
    reference_profile: ReliabilityProfile
    target_profile: ReliabilityProfile
    degradation_summary: ReliabilityDegradationSummary
    dataset_shift_overview: dict[str, Any] | None = None
    weighted_reference_profile: ReliabilityProfile | None = None
    recalibrated_target_profile: ReliabilityProfile | None = None
    weighting_summary: dict[str, Any] | None = None
    recalibration_summary: dict[str, Any] | None = None
    calibration_comparison: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable representation."""

        return {
            "estimator_name": self.estimator_name,
            "reference_profile": self.reference_profile.to_dict(),
            "target_profile": self.target_profile.to_dict(),
            "degradation_summary": self.degradation_summary.to_dict(),
            "dataset_shift_overview": self.dataset_shift_overview,
            "weighted_reference_profile": (
                None
                if self.weighted_reference_profile is None
                else self.weighted_reference_profile.to_dict()
            ),
            "recalibrated_target_profile": (
                None
                if self.recalibrated_target_profile is None
                else self.recalibrated_target_profile.to_dict()
            ),
            "weighting_summary": self.weighting_summary,
            "recalibration_summary": self.recalibration_summary,
            "calibration_comparison": self.calibration_comparison,
        }

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact tabular summary of the evaluation."""

        record = {
            "estimator_name": self.estimator_name,
            "reference_accuracy": self.reference_profile.accuracy,
            "target_accuracy": self.target_profile.accuracy,
            "reference_ece": self.reference_profile.ece,
            "target_ece": self.target_profile.ece,
            "delta_accuracy": self.degradation_summary.delta_accuracy,
            "delta_ece": self.degradation_summary.delta_ece,
            "delta_log_loss": self.degradation_summary.delta_log_loss,
        }
        if self.recalibrated_target_profile is not None:
            record["recalibrated_target_ece"] = self.recalibrated_target_profile.ece
            record["recalibrated_target_log_loss"] = self.recalibrated_target_profile.log_loss
        return pd.DataFrame.from_records([record])  # type: ignore[no-any-return]

    def to_report(self) -> ReliabilityShiftReport:
        """Convert the workflow result to a report object."""

        return ReliabilityShiftReport(
            dataset_shift_overview=self.dataset_shift_overview,
            reference_profile=self.reference_profile,
            target_profile=self.target_profile,
            degradation_summary=self.degradation_summary,
            weighted_reference_profile=self.weighted_reference_profile,
            recalibrated_target_profile=self.recalibrated_target_profile,
            weighting_summary=self.weighting_summary,
            recalibration_summary=self.recalibration_summary,
            calibration_comparison=self.calibration_comparison,
        )
