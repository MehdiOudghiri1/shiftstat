"""Selective prediction evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shiftstat.metrics import risk_coverage_table, selective_summary
from shiftstat.reliability.analyzer import ReliabilityAnalyzer
from shiftstat.selective.policy import AbstentionPolicy
from shiftstat.selective.results import RiskCoverageCurve, SelectiveProfile
from shiftstat.subgroup import group_by_feature
from shiftstat.utils.probabilities import confidence_from_probabilities, extract_positive_class_probabilities
from shiftstat.utils.schema import extract_feature_names


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()  # type: ignore[no-any-return]
    array = np.asarray(X)
    return pd.DataFrame(array, columns=extract_feature_names(X))  # type: ignore[no-any-return]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _distribution_table(
    values: np.ndarray,
    accepted_mask: np.ndarray,
    *,
    n_bins: int = 10,
    value_name: str,
) -> list[dict[str, Any]]:
    series = np.asarray(values, dtype=float)
    mask = np.asarray(accepted_mask, dtype=bool)
    edges = np.linspace(float(np.min(series)), float(np.max(series)) + 1e-12, n_bins + 1)
    assignments = np.digitize(series, edges[1:-1], right=True)
    records: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        bin_mask = assignments == index
        if not np.any(bin_mask):
            continue
        for status, status_mask in [("accepted", mask), ("rejected", ~mask)]:
            joint_mask = bin_mask & status_mask
            if not np.any(joint_mask):
                continue
            count = int(np.sum(joint_mask))
            records.append(
                {
                    value_name: status,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": count,
                    "share": float(count / max(len(series), 1)),
                }
            )
    return records


def _collapse_intersection(groups: list[pd.Series]) -> pd.Series:
    output = groups[0].astype(str).copy()
    for group in groups[1:]:
        output = output.str.cat(group.astype(str), sep=" & ")
    return output.astype(str)


def abstention_summary_by_groups(
    X: Any,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: AbstentionPolicy,
    *,
    sample_weight: np.ndarray | None = None,
    dataset_name: str = "dataset",
    subgroup_features: list[str] | list[int] | None = None,
    intersectional_features: list[tuple[str | int, ...]] | None = None,
    subgroup_bins: int = 4,
    subgroup_strategy: str = "quantile",
    max_categories: int = 8,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Summarize abstention behavior across user-defined slices."""

    frame = _as_frame(X)
    features = subgroup_features or extract_feature_names(frame)
    intersections = intersectional_features or []
    probabilities = extract_positive_class_probabilities(y_prob)
    accepted = policy.accept_mask(probabilities, X=frame)
    scores = policy.score(probabilities, X=frame)

    grouping_specs: list[tuple[str, pd.Series]] = []
    for feature in features:
        labels = group_by_feature(
            frame,
            feature,
            n_bins=subgroup_bins,
            strategy=subgroup_strategy,
            max_categories=max_categories,
        )
        name = str(feature) if isinstance(feature, str) else str(frame.columns[int(feature)])
        grouping_specs.append((name, labels))

    for feature_tuple in intersections:
        grouped = [
            group_by_feature(
                frame,
                feature,
                n_bins=subgroup_bins,
                strategy=subgroup_strategy,
                max_categories=max_categories,
            )
            for feature in feature_tuple
        ]
        labels = _collapse_intersection(grouped)
        group_names = [
            str(feature) if isinstance(feature, str) else str(frame.columns[int(feature)])
            for feature in feature_tuple
        ]
        grouping_specs.append((" x ".join(group_names), labels))

    records: list[dict[str, Any]] = []
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    confidence = confidence_from_probabilities(probabilities)
    for slice_name, groups in grouping_specs:
        for group_value in sorted(groups.astype(str).unique().tolist()):
            mask = groups.astype(str) == group_value
            accepted_mask = accepted[mask.to_numpy()]
            subset_weight = None if weights is None else weights[mask.to_numpy()]
            summary = selective_summary(
                np.asarray(y_true)[mask.to_numpy()],
                probabilities[mask.to_numpy()],
                accepted_mask,
                sample_weight=subset_weight,
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "slice_name": slice_name,
                    "group": str(group_value),
                    "n_samples": int(mask.sum()),
                    "accepted_samples": int(np.sum(accepted_mask)),
                    "rejected_samples": int(np.sum(~accepted_mask)),
                    "coverage": float(summary["coverage"]),
                    "abstention_rate": float(summary["abstention_rate"]),
                    "selective_accuracy": float(summary["selective_accuracy"]),
                    "selective_risk": float(summary["selective_risk"]),
                    "full_accuracy": float(summary["full_accuracy"]),
                    "risk_reduction": float(summary["risk_reduction"]),
                    "mean_selection_score": float(np.mean(scores[mask.to_numpy()])),
                    "mean_confidence": float(np.mean(confidence[mask.to_numpy()])),
                    "support_ok": bool(int(mask.sum()) >= min_group_size),
                }
            )
    return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]


class SelectivePredictor:
    """Apply and evaluate abstention policies on predictive probabilities."""

    def __init__(
        self,
        policy: AbstentionPolicy,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        distribution_bins: int = 10,
    ) -> None:
        self.policy = policy
        self.n_bins = n_bins
        self.strategy = strategy
        self.distribution_bins = distribution_bins

    def fit(
        self,
        y_true: np.ndarray | None,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        sample_weight: np.ndarray | None = None,
        threshold: float | None = None,
        target_coverage: float | None = None,
        target_risk: float | None = None,
        max_thresholds: int = 101,
    ) -> SelectivePredictor:
        """Fit the abstention policy and resolve the deployment threshold."""

        self.policy.fit(
            y_true,
            y_prob,
            X=X,
            sample_weight=sample_weight,
            threshold=threshold,
            target_coverage=target_coverage,
            target_risk=target_risk,
            max_thresholds=max_thresholds,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )
        return self

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        sample_weight: np.ndarray | None = None,
        name: str = "dataset",
    ) -> SelectiveProfile:
        """Evaluate the fitted policy on one dataset."""

        probabilities = extract_positive_class_probabilities(y_prob)
        accepted = self.policy.accept_mask(probabilities, X=X)
        scores = self.policy.score(probabilities, X=X)
        analyzer = ReliabilityAnalyzer(n_bins=self.n_bins, strategy=self.strategy)
        summary = selective_summary(
            y_true,
            probabilities,
            accepted,
            sample_weight=sample_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )

        accepted_profile = None
        if np.any(accepted):
            accepted_weights = None if sample_weight is None else np.asarray(sample_weight)[accepted]
            accepted_profile = analyzer._build_profile(
                np.asarray(y_true)[accepted],
                probabilities[accepted],
                sample_weight=accepted_weights,
                name=f"{name}_accepted",
            )

        return SelectiveProfile(
            name=name,
            policy_name=self.policy.name,
            score_method=self.policy.method,
            threshold=float(self.policy.threshold_ if hasattr(self.policy, "threshold_") else self.policy.threshold),
            weighted=sample_weight is not None,
            n_samples=int(len(probabilities)),
            accepted_samples=int(summary["accepted_samples"]),
            rejected_samples=int(summary["rejected_samples"]),
            coverage=float(summary["coverage"]),
            abstention_rate=float(summary["abstention_rate"]),
            selective_accuracy=float(summary["selective_accuracy"]),
            selective_risk=float(summary["selective_risk"]),
            selective_log_loss=float(summary["selective_log_loss"]),
            selective_brier_score=float(summary["selective_brier_score"]),
            selective_ece=float(summary["selective_ece"]),
            selective_mce=float(summary["selective_mce"]),
            full_accuracy=float(summary["full_accuracy"]),
            full_risk=float(summary["full_risk"]),
            full_log_loss=float(summary["full_log_loss"]),
            full_brier_score=float(summary["full_brier_score"]),
            full_ece=float(summary["full_ece"]),
            full_mce=float(summary["full_mce"]),
            risk_reduction=float(summary["risk_reduction"]),
            log_loss_reduction=float(summary["log_loss_reduction"]),
            ece_reduction=float(summary["ece_reduction"]),
            mean_confidence_accepted=float(summary["mean_confidence_accepted"]),
            mean_confidence_rejected=float(summary["mean_confidence_rejected"]),
            mean_entropy_accepted=float(summary["mean_entropy_accepted"]),
            mean_entropy_rejected=float(summary["mean_entropy_rejected"]),
            tuning_summary=self.policy.summary(),
            score_distribution_table=_distribution_table(
                scores,
                accepted,
                n_bins=self.distribution_bins,
                value_name="status",
            ),
            confidence_distribution_table=_distribution_table(
                confidence_from_probabilities(probabilities),
                accepted,
                n_bins=self.distribution_bins,
                value_name="status",
            ),
            accepted_reliability_profile=accepted_profile,
        )

    def risk_coverage_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        sample_weight: np.ndarray | None = None,
        name: str = "dataset",
        max_points: int = 101,
    ) -> RiskCoverageCurve:
        """Compute a risk-coverage curve for the current policy score."""

        probabilities = extract_positive_class_probabilities(y_prob)
        scores = self.policy.score(probabilities, X=X)
        frame = risk_coverage_table(
            y_true,
            probabilities,
            scores,
            sample_weight=sample_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
            max_points=max_points,
        )
        return RiskCoverageCurve(
            name=name,
            policy_name=self.policy.name,
            score_method=self.policy.method,
            records=_frame_records(frame),
        )
