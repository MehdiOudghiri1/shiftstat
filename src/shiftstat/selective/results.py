"""Result containers for selective prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from shiftstat.reliability.results import ReliabilityProfile


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Risk-coverage curve for one abstention policy on one dataset."""

    name: str
    policy_name: str
    score_method: str
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "policy_name": self.policy_name,
            "score_method": self.score_method,
            "records": self.records,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the curve as a DataFrame."""

        return pd.DataFrame.from_records(self.records)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class SelectiveProfile:
    """Selective prediction summary for a single dataset."""

    name: str
    policy_name: str
    score_method: str
    threshold: float
    weighted: bool
    n_samples: int
    accepted_samples: int
    rejected_samples: int
    coverage: float
    abstention_rate: float
    selective_accuracy: float
    selective_risk: float
    selective_log_loss: float
    selective_brier_score: float
    selective_ece: float
    selective_mce: float
    full_accuracy: float
    full_risk: float
    full_log_loss: float
    full_brier_score: float
    full_ece: float
    full_mce: float
    risk_reduction: float
    log_loss_reduction: float
    ece_reduction: float
    mean_confidence_accepted: float
    mean_confidence_rejected: float
    mean_entropy_accepted: float
    mean_entropy_rejected: float
    tuning_summary: dict[str, Any] | None
    score_distribution_table: list[dict[str, Any]]
    confidence_distribution_table: list[dict[str, Any]]
    accepted_reliability_profile: ReliabilityProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "policy_name": self.policy_name,
            "score_method": self.score_method,
            "threshold": self.threshold,
            "weighted": self.weighted,
            "n_samples": self.n_samples,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "selective_accuracy": self.selective_accuracy,
            "selective_risk": self.selective_risk,
            "selective_log_loss": self.selective_log_loss,
            "selective_brier_score": self.selective_brier_score,
            "selective_ece": self.selective_ece,
            "selective_mce": self.selective_mce,
            "full_accuracy": self.full_accuracy,
            "full_risk": self.full_risk,
            "full_log_loss": self.full_log_loss,
            "full_brier_score": self.full_brier_score,
            "full_ece": self.full_ece,
            "full_mce": self.full_mce,
            "risk_reduction": self.risk_reduction,
            "log_loss_reduction": self.log_loss_reduction,
            "ece_reduction": self.ece_reduction,
            "mean_confidence_accepted": self.mean_confidence_accepted,
            "mean_confidence_rejected": self.mean_confidence_rejected,
            "mean_entropy_accepted": self.mean_entropy_accepted,
            "mean_entropy_rejected": self.mean_entropy_rejected,
            "tuning_summary": self.tuning_summary,
            "score_distribution_table": self.score_distribution_table,
            "confidence_distribution_table": self.confidence_distribution_table,
            "accepted_reliability_profile": (
                None
                if self.accepted_reliability_profile is None
                else self.accepted_reliability_profile.to_dict()
            ),
        }

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact one-row summary."""

        return pd.DataFrame.from_records(
            [
                {
                    "name": self.name,
                    "policy_name": self.policy_name,
                    "score_method": self.score_method,
                    "threshold": self.threshold,
                    "coverage": self.coverage,
                    "abstention_rate": self.abstention_rate,
                    "selective_accuracy": self.selective_accuracy,
                    "selective_risk": self.selective_risk,
                    "selective_log_loss": self.selective_log_loss,
                    "selective_ece": self.selective_ece,
                    "risk_reduction": self.risk_reduction,
                    "log_loss_reduction": self.log_loss_reduction,
                    "ece_reduction": self.ece_reduction,
                }
            ]
        )  # type: ignore[no-any-return]

    def score_distribution_frame(self) -> pd.DataFrame:
        """Return score distributions for accepted and rejected samples."""

        return pd.DataFrame.from_records(self.score_distribution_table)  # type: ignore[no-any-return]

    def confidence_distribution_frame(self) -> pd.DataFrame:
        """Return confidence distributions for accepted and rejected samples."""

        return pd.DataFrame.from_records(self.confidence_distribution_table)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class SelectiveDeploymentReport:
    """Operational report for selective deployment under shift."""

    dataset_shift_overview: dict[str, Any] | None
    policy_summary: dict[str, Any]
    reference_profile: SelectiveProfile
    target_profile: SelectiveProfile
    subgroup_abstention_summary: list[dict[str, Any]]
    subgroup_abstention_disparities: list[dict[str, Any]]
    risk_coverage_curve: list[dict[str, Any]]
    threshold_comparison: list[dict[str, Any]] | None = None
    recalibrated_target_profile: SelectiveProfile | None = None
    caveats: list[str] | None = None
    operational_implications: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "dataset_shift_overview": self.dataset_shift_overview,
            "policy_summary": self.policy_summary,
            "reference_profile": self.reference_profile.to_dict(),
            "target_profile": self.target_profile.to_dict(),
            "subgroup_abstention_summary": self.subgroup_abstention_summary,
            "subgroup_abstention_disparities": self.subgroup_abstention_disparities,
            "risk_coverage_curve": self.risk_coverage_curve,
            "threshold_comparison": self.threshold_comparison,
            "recalibrated_target_profile": (
                None
                if self.recalibrated_target_profile is None
                else self.recalibrated_target_profile.to_dict()
            ),
            "caveats": self.caveats,
            "operational_implications": self.operational_implications,
        }

    def to_markdown(self, *, top_k: int = 5) -> str:
        """Render a concise markdown report."""

        lines = ["## Selective deployment report", ""]
        if self.dataset_shift_overview is not None:
            lines.extend(
                [
                    "### Shift overview",
                    "",
                    f"- Classifier two-sample AUC: {self.dataset_shift_overview['classifier_auc']:.3f}",
                    f"- Shifted features: {self.dataset_shift_overview['n_shifted_features']}",
                    "",
                ]
            )

        lines.extend(
            [
                "### Policy summary",
                "",
                f"- Policy: {self.policy_summary['policy_name']}",
                f"- Score method: {self.policy_summary['score_method']}",
                f"- Threshold: {self.policy_summary['threshold']:.3f}",
                "",
                "### Target before/after abstention",
                "",
                f"- Target full accuracy: {self.target_profile.full_accuracy:.3f}",
                f"- Target selective accuracy: {self.target_profile.selective_accuracy:.3f}",
                f"- Coverage retained: {self.target_profile.coverage:.3f}",
                f"- Abstention rate: {self.target_profile.abstention_rate:.3f}",
                f"- Risk reduction: {self.target_profile.risk_reduction:.3f}",
                f"- ECE reduction: {self.target_profile.ece_reduction:.3f}",
                "",
                "### Subgroup abstention summary",
                "",
            ]
        )

        subgroup_frame = self.subgroup_abstention_frame().head(top_k)
        if subgroup_frame.empty:
            lines.append("- No subgroup abstention rows available.")
        else:
            for _, row in subgroup_frame.iterrows():
                lines.append(
                    (
                        f"- {row['slice_name']} -> {row['group']}: "
                        f"coverage {row['target_coverage']:.3f}, "
                        f"abstention {row['target_abstention_rate']:.3f}, "
                        f"accepted accuracy {row['target_selective_accuracy']:.3f}"
                    )
                )

        if self.recalibrated_target_profile is not None:
            lines.extend(
                [
                    "",
                    "### Recalibrated target comparison",
                    "",
                    (
                        "- Recalibrated selective ECE: "
                        f"{self.recalibrated_target_profile.selective_ece:.3f}"
                    ),
                    (
                        "- Recalibrated selective log loss: "
                        f"{self.recalibrated_target_profile.selective_log_loss:.3f}"
                    ),
                ]
            )

        if self.threshold_comparison is not None:
            comparison = self.threshold_comparison_frame()
            if not comparison.empty:
                lines.extend(["", "### Threshold tuning comparison", ""])
                for _, row in comparison.iterrows():
                    lines.append(
                        (
                            f"- weighted={row['weighted']}: threshold {row['threshold']:.3f}, "
                            f"coverage {row['achieved_coverage']:.3f}, "
                            f"selective risk {row['achieved_selective_risk']:.3f}"
                        )
                    )

        disparity = self.subgroup_disparity_frame().head(1)
        if not disparity.empty:
            row = disparity.iloc[0]
            lines.extend(
                [
                    "",
                    "### Largest subgroup disparity",
                    "",
                    (
                        f"- {row['metric']}: {row['worst_group']} vs {row['best_group']} "
                        f"(gap {row['absolute_gap']:.3f})"
                    ),
                ]
            )

        if self.caveats:
            lines.extend(["", "### Caveats", ""])
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
        if self.operational_implications:
            lines.extend(["", "### Operational implications", ""])
            for implication in self.operational_implications:
                lines.append(f"- {implication}")
        return "\n".join(lines)

    def subgroup_abstention_frame(self) -> pd.DataFrame:
        """Return subgroup abstention summaries."""

        return pd.DataFrame.from_records(self.subgroup_abstention_summary)  # type: ignore[no-any-return]

    def subgroup_disparity_frame(self) -> pd.DataFrame:
        """Return subgroup abstention disparities."""

        return pd.DataFrame.from_records(self.subgroup_abstention_disparities)  # type: ignore[no-any-return]

    def risk_coverage_frame(self) -> pd.DataFrame:
        """Return the risk-coverage curve."""

        return pd.DataFrame.from_records(self.risk_coverage_curve)  # type: ignore[no-any-return]

    def threshold_comparison_frame(self) -> pd.DataFrame:
        """Return weighted versus unweighted threshold comparisons."""

        if self.threshold_comparison is None:
            return pd.DataFrame()
        return pd.DataFrame.from_records(self.threshold_comparison)  # type: ignore[no-any-return]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return exportable report tables."""

        tables = {
            "reference_profile": self.reference_profile.summary_frame(),
            "target_profile": self.target_profile.summary_frame(),
            "subgroup_abstention": self.subgroup_abstention_frame(),
            "subgroup_disparities": self.subgroup_disparity_frame(),
            "risk_coverage_curve": self.risk_coverage_frame(),
        }
        if self.threshold_comparison is not None:
            tables["threshold_comparison"] = self.threshold_comparison_frame()
        if self.recalibrated_target_profile is not None:
            tables["recalibrated_target_profile"] = self.recalibrated_target_profile.summary_frame()
        return tables


@dataclass(frozen=True)
class SelectiveShiftEvaluationResult:
    """Structured output for selective evaluation under shift."""

    estimator_name: str
    dataset_shift_overview: dict[str, Any] | None
    policy_summary: dict[str, Any]
    reference_full_profile: ReliabilityProfile
    target_full_profile: ReliabilityProfile
    reference_selective_profile: SelectiveProfile
    target_selective_profile: SelectiveProfile
    target_risk_coverage_curve: RiskCoverageCurve
    subgroup_abstention_summary: list[dict[str, Any]]
    subgroup_abstention_disparities: list[dict[str, Any]]
    threshold_comparison: list[dict[str, Any]] | None = None
    weighting_summary: dict[str, Any] | None = None
    recalibration_summary: dict[str, Any] | None = None
    recalibrated_target_selective_profile: SelectiveProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable representation."""

        return {
            "estimator_name": self.estimator_name,
            "dataset_shift_overview": self.dataset_shift_overview,
            "policy_summary": self.policy_summary,
            "reference_full_profile": self.reference_full_profile.to_dict(),
            "target_full_profile": self.target_full_profile.to_dict(),
            "reference_selective_profile": self.reference_selective_profile.to_dict(),
            "target_selective_profile": self.target_selective_profile.to_dict(),
            "target_risk_coverage_curve": self.target_risk_coverage_curve.to_dict(),
            "subgroup_abstention_summary": self.subgroup_abstention_summary,
            "subgroup_abstention_disparities": self.subgroup_abstention_disparities,
            "threshold_comparison": self.threshold_comparison,
            "weighting_summary": self.weighting_summary,
            "recalibration_summary": self.recalibration_summary,
            "recalibrated_target_selective_profile": (
                None
                if self.recalibrated_target_selective_profile is None
                else self.recalibrated_target_selective_profile.to_dict()
            ),
        }

    def summary_frame(self) -> pd.DataFrame:
        """Return a compact summary of target deployment tradeoffs."""

        record = {
            "estimator_name": self.estimator_name,
            "policy_name": self.policy_summary["policy_name"],
            "score_method": self.policy_summary["score_method"],
            "threshold": self.policy_summary["threshold"],
            "target_coverage": self.target_selective_profile.coverage,
            "target_abstention_rate": self.target_selective_profile.abstention_rate,
            "target_selective_accuracy": self.target_selective_profile.selective_accuracy,
            "target_selective_risk": self.target_selective_profile.selective_risk,
            "target_risk_reduction": self.target_selective_profile.risk_reduction,
            "target_selective_ece": self.target_selective_profile.selective_ece,
            "target_ece_reduction": self.target_selective_profile.ece_reduction,
        }
        if self.recalibrated_target_selective_profile is not None:
            record["recalibrated_target_selective_ece"] = (
                self.recalibrated_target_selective_profile.selective_ece
            )
            record["recalibrated_target_selective_log_loss"] = (
                self.recalibrated_target_selective_profile.selective_log_loss
            )
        return pd.DataFrame.from_records([record])  # type: ignore[no-any-return]

    def to_report(self) -> SelectiveDeploymentReport:
        """Convert the workflow result to a deployment report."""

        caveats = [
            "Abstention can reduce observed accepted-set risk while leaving rejected-set failures operationally important.",
            "Coverage-risk tradeoffs depend on the validation distribution used for threshold tuning.",
        ]
        operational_implications = [
            (
                "The deployed policy retains "
                f"{self.target_selective_profile.coverage:.3f} coverage at "
                f"{self.target_selective_profile.selective_risk:.3f} selective risk."
            )
        ]
        if self.target_selective_profile.risk_reduction <= 0.0:
            caveats.append(
                "Accepted-set risk did not improve on the target sample, so abstention may be hiding failures rather than helping."
            )
        else:
            operational_implications.append(
                f"Accepted-set risk fell by {self.target_selective_profile.risk_reduction:.3f} after abstention."
            )
        if self.target_selective_profile.ece_reduction < 0.0:
            caveats.append(
                "Accepted-set calibration worsened even though the policy abstained on part of the target distribution."
            )
        disparity_frame = pd.DataFrame.from_records(self.subgroup_abstention_disparities)
        if not disparity_frame.empty:
            top = disparity_frame.sort_values("absolute_gap", ascending=False).iloc[0]
            operational_implications.append(
                f"Largest subgroup rejection disparity appears on {top['metric']} with gap {top['absolute_gap']:.3f}."
            )
        return SelectiveDeploymentReport(
            dataset_shift_overview=self.dataset_shift_overview,
            policy_summary=self.policy_summary,
            reference_profile=self.reference_selective_profile,
            target_profile=self.target_selective_profile,
            subgroup_abstention_summary=self.subgroup_abstention_summary,
            subgroup_abstention_disparities=self.subgroup_abstention_disparities,
            risk_coverage_curve=self.target_risk_coverage_curve.records,
            threshold_comparison=self.threshold_comparison,
            recalibrated_target_profile=self.recalibrated_target_selective_profile,
            caveats=caveats,
            operational_implications=operational_implications,
        )
