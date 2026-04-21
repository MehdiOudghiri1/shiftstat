"""Subgroup-aware reliability diagnostics under deployment shift."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import SmallSampleWarning
from ..reliability.analyzer import ReliabilityAnalyzer
from ..utils.probabilities import extract_positive_class_probabilities
from ..utils.schema import align_tabular_inputs, extract_feature_names, infer_feature_types
from ..utils.validation import ensure_1d, validate_same_length
from .results import SubgroupReport


@dataclass(frozen=True)
class _GroupingContext:
    name: str
    reference_groups: pd.Series
    target_groups: pd.Series


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()  # type: ignore[no-any-return]
    array = np.asarray(X)
    return pd.DataFrame(array, columns=extract_feature_names(X))  # type: ignore[no-any-return]


def _resolve_feature_name(X: pd.DataFrame, feature: str | int) -> str:
    if isinstance(feature, str):
        if feature not in X.columns:
            raise ValueError(f"Unknown feature {feature!r}.")
        return feature
    feature_index = int(feature)
    if feature_index < 0 or feature_index >= X.shape[1]:
        raise ValueError(f"Feature index {feature_index} is out of bounds.")
    return str(X.columns[feature_index])


def _format_interval(feature_name: str, interval: pd.Interval) -> str:
    left_bracket = "[" if interval.closed_left else "("
    right_bracket = "]" if interval.closed_right else ")"
    return (
        f"{feature_name} in {left_bracket}{interval.left:.3f}, {interval.right:.3f}{right_bracket}"
    )


def _collapse_rare_groups(
    groups: pd.Series,
    *,
    max_groups: int,
    min_group_size: int,
    other_label: str,
) -> pd.Series:
    counts = groups.value_counts(dropna=False)
    keep = counts[counts >= min_group_size].index.tolist()[: max(max_groups - 1, 1)]
    collapsed = groups.where(groups.isin(keep), other_label)
    return collapsed.astype(str)


def group_by_feature(
    X: Any,
    feature: str | int,
    *,
    reference: Any | None = None,
    categorical: bool | None = None,
    categorical_features: list[str] | list[int] | None = None,
    n_bins: int = 4,
    strategy: str = "quantile",
    max_categories: int = 8,
    missing_label: str = "missing",
) -> pd.Series:
    """Create interpretable subgroup labels from one feature."""

    X_frame = _as_frame(X)
    reference_frame = X_frame if reference is None else _as_frame(reference)
    feature_name = _resolve_feature_name(X_frame, feature)
    reference_name = _resolve_feature_name(reference_frame, feature_name)

    feature_types = infer_feature_types(
        reference_frame,
        categorical_features=categorical_features,
    )
    if categorical is None:
        categorical = feature_types[reference_name] == "categorical"

    series = X_frame[feature_name]
    reference_series = reference_frame[reference_name]

    if categorical:
        base = reference_series.astype("string").fillna(missing_label)
        categories = base.value_counts().index.tolist()[:max_categories]
        labels = series.astype("string").fillna(missing_label)
        labels = labels.where(labels.isin(categories), "other")
        return labels.map(lambda value: f"{feature_name}={value}").astype(str)

    numeric_reference = pd.to_numeric(reference_series, errors="coerce")
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_reference = numeric_reference.dropna()
    if valid_reference.empty:
        return pd.Series([f"{feature_name}={missing_label}"] * len(series), index=series.index)

    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")

    if strategy == "quantile":
        edges = np.quantile(valid_reference, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(float(valid_reference.min()), float(valid_reference.max()), n_bins + 1)
    edges = np.unique(edges)
    if len(edges) < 2:
        constant_value = float(valid_reference.iloc[0])
        return pd.Series(
            [f"{feature_name}={constant_value:.3f}"] * len(series),
            index=series.index,
        )
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    binned = pd.cut(
        numeric_series,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    labels = binned.map(
        lambda interval: (
            f"{feature_name}={missing_label}"
            if pd.isna(interval)
            else _format_interval(feature_name, interval)
        )
    )
    return labels.astype(str)


def group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: Any,
    *,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    dataset_name: str = "dataset",
    slice_name: str = "group",
    min_group_size: int = 30,
    min_class_count: int = 8,
    warn_on_small_groups: bool = True,
) -> pd.DataFrame:
    """Compute performance and calibration metrics by subgroup."""

    y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
    prob_arr = extract_positive_class_probabilities(y_prob)
    group_arr = pd.Series(ensure_1d(groups, name="groups"), name="group").astype(str)
    validate_same_length(y_true_arr, prob_arr, group_arr)

    if sample_weight is not None:
        weight_arr = ensure_1d(sample_weight, name="sample_weight").astype(float)
        validate_same_length(y_true_arr, weight_arr)
    else:
        weight_arr = None

    analyzer = ReliabilityAnalyzer(n_bins=n_bins, strategy=strategy)
    total_n = len(y_true_arr)
    total_weight = float(np.sum(weight_arr)) if weight_arr is not None else float(total_n)

    records: list[dict[str, Any]] = []
    flagged_groups: list[str] = []
    for group_value in sorted(group_arr.unique().tolist()):
        mask = group_arr == group_value
        y_group = y_true_arr[mask.to_numpy()]
        p_group = prob_arr[mask.to_numpy()]
        weights_group = weight_arr[mask.to_numpy()] if weight_arr is not None else None

        n_samples = int(mask.sum())
        n_positive = int(np.sum(y_group == 1))
        n_negative = int(np.sum(y_group == 0))
        support_ok = n_samples >= min_group_size and min(n_positive, n_negative) >= min_class_count
        if not support_ok:
            flagged_groups.append(str(group_value))

        profile = analyzer._build_profile(
            y_group,
            p_group,
            sample_weight=weights_group,
            name=str(group_value),
        )
        weight_sum = float(np.sum(weights_group)) if weights_group is not None else float(n_samples)
        records.append(
            {
                "dataset": dataset_name,
                "slice_name": slice_name,
                "group": str(group_value),
                "n_samples": n_samples,
                "weight_sum": weight_sum,
                "sample_share": float(n_samples / max(total_n, 1)),
                "weight_share": float(weight_sum / max(total_weight, 1e-12)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "support_ok": support_ok,
                "sample_size_flag": not support_ok,
                "accuracy": profile.accuracy,
                "error_rate": profile.error_rate,
                "log_loss": profile.log_loss,
                "brier_score": profile.brier_score,
                "ece": profile.ece,
                "mce": profile.mce,
                "calibration_intercept": profile.calibration_intercept,
                "calibration_slope": profile.calibration_slope,
                "mean_confidence": profile.mean_confidence,
                "mean_outcome": profile.mean_outcome,
                "mean_prediction": float(np.average(p_group, weights=weights_group)),
                "confidence_gap": profile.confidence_gap,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if flagged_groups and warn_on_small_groups:
        warnings.warn(
            (
                f"{slice_name} includes {len(flagged_groups)} subgroup(s) below the default "
                f"support thresholds ({min_group_size} samples and {min_class_count} examples "
                f"from each class). Interpret slice-level estimates cautiously."
            ),
            SmallSampleWarning,
            stacklevel=2,
        )
    return frame  # type: ignore[no-any-return]


class SubgroupAnalyzer:
    """Analyze reliability degradation across user-defined or derived subgroups."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        subgroup_bins: int = 4,
        subgroup_strategy: str = "quantile",
        max_categories: int = 8,
        max_intersection_groups: int = 20,
        min_group_size: int = 30,
        min_class_count: int = 8,
        categorical_features: list[str] | list[int] | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.subgroup_bins = subgroup_bins
        self.subgroup_strategy = subgroup_strategy
        self.max_categories = max_categories
        self.max_intersection_groups = max_intersection_groups
        self.min_group_size = min_group_size
        self.min_class_count = min_class_count
        self.categorical_features = categorical_features

    def fit(
        self,
        X_ref: Any,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        X_target: Any,
        y_target: np.ndarray,
        p_target: np.ndarray,
        *,
        subgroup_features: list[str] | list[int] | None = None,
        intersectional_features: list[tuple[str | int, ...]] | None = None,
        reference_weight: np.ndarray | None = None,
        target_weight: np.ndarray | None = None,
    ) -> SubgroupAnalyzer:
        """Fit subgroup reliability diagnostics on reference and target samples."""

        X_ref_aligned, X_target_aligned = align_tabular_inputs(X_ref, X_target)
        ref_frame = _as_frame(X_ref_aligned)
        target_frame = _as_frame(X_target_aligned)
        combined_frame = pd.concat([ref_frame, target_frame], axis=0, ignore_index=True)

        feature_names = extract_feature_names(ref_frame)
        features = subgroup_features or feature_names
        intersections = intersectional_features or []

        cached_groups: dict[str, tuple[pd.Series, pd.Series]] = {}
        performance_frames: list[pd.DataFrame] = []
        degradation_frames: list[pd.DataFrame] = []
        stability_rows: list[dict[str, Any]] = []
        warnings_list: list[str] = []

        for feature in features:
            context = self._build_grouping_context(
                ref_frame,
                target_frame,
                combined_frame,
                feature,
                cached_groups=cached_groups,
            )
            performance_frame, degradation_frame, stability_row = self._evaluate_grouping(
                y_ref,
                p_ref,
                y_target,
                p_target,
                context,
                reference_weight=reference_weight,
                target_weight=target_weight,
            )
            performance_frames.append(performance_frame)
            degradation_frames.append(degradation_frame)
            stability_rows.append(stability_row)
            warnings_list.extend(_warnings_from_frame(degradation_frame, context.name))

        for feature_tuple in intersections:
            context = self._build_intersection_context(
                ref_frame,
                target_frame,
                combined_frame,
                feature_tuple,
                cached_groups=cached_groups,
            )
            performance_frame, degradation_frame, stability_row = self._evaluate_grouping(
                y_ref,
                p_ref,
                y_target,
                p_target,
                context,
                reference_weight=reference_weight,
                target_weight=target_weight,
            )
            performance_frames.append(performance_frame)
            degradation_frames.append(degradation_frame)
            stability_rows.append(stability_row)
            warnings_list.extend(_warnings_from_frame(degradation_frame, context.name))

        self.performance_table_ = pd.concat(performance_frames, ignore_index=True)
        self.calibration_table_ = self.performance_table_.loc[
            :,
            [
                "dataset",
                "slice_name",
                "group",
                "n_samples",
                "sample_share",
                "support_ok",
                "sample_size_flag",
                "ece",
                "mce",
                "calibration_intercept",
                "calibration_slope",
                "mean_prediction",
                "mean_outcome",
                "confidence_gap",
            ],
        ].copy()
        self.degradation_table_ = pd.concat(degradation_frames, ignore_index=True).sort_values(
            ["severity_score", "delta_error_rate", "delta_ece"],
            ascending=[False, False, False],
            ignore_index=True,
        )
        self.shift_exposure_table_ = self.degradation_table_.loc[
            :,
            [
                "slice_name",
                "group",
                "reference_n_samples",
                "target_n_samples",
                "reference_sample_share",
                "target_sample_share",
                "exposure_gap",
                "exposure_ratio",
                "supported_both",
            ],
        ].copy()
        self.stability_table_ = pd.DataFrame.from_records(stability_rows)
        self.warnings_ = _deduplicate_preserve_order(warnings_list)
        self.aggregate_summary_ = {
            "n_groupings": int(len(performance_frames)),
            "n_degradation_rows": int(len(self.degradation_table_)),
            "supported_target_coverage": float(
                self.stability_table_["target_supported_coverage"].mean()
                if not self.stability_table_.empty
                else 0.0
            ),
            "flagged_group_rows": int(self.performance_table_["sample_size_flag"].sum()),
        }
        self.worst_group_summary_ = self._build_worst_group_summary()
        return self

    def performance_table(self, *, dataset: str | None = None) -> pd.DataFrame:
        """Return subgroup performance metrics."""

        self._check_is_fitted()
        frame = self.performance_table_.copy()
        if dataset is None:
            return frame  # type: ignore[no-any-return]
        return frame.loc[frame["dataset"] == dataset].reset_index(drop=True)  # type: ignore[no-any-return]

    def calibration_table(self, *, dataset: str | None = None) -> pd.DataFrame:
        """Return subgroup calibration metrics."""

        self._check_is_fitted()
        frame = self.calibration_table_.copy()
        if dataset is None:
            return frame  # type: ignore[no-any-return]
        return frame.loc[frame["dataset"] == dataset].reset_index(drop=True)  # type: ignore[no-any-return]

    def shift_exposure_summary(self) -> pd.DataFrame:
        """Return target-versus-reference subgroup exposure summaries."""

        self._check_is_fitted()
        return self.shift_exposure_table_.copy()  # type: ignore[no-any-return]

    def degradation_ranking(self) -> pd.DataFrame:
        """Return subgroup degradations ranked by severity."""

        self._check_is_fitted()
        return self.degradation_table_.copy()  # type: ignore[no-any-return]

    def stability_diagnostics(self) -> pd.DataFrame:
        """Return slice-level support and coverage diagnostics."""

        self._check_is_fitted()
        return self.stability_table_.copy()  # type: ignore[no-any-return]

    def worst_group_summary(self) -> dict[str, Any]:
        """Return a compact summary of worst-group outcomes."""

        self._check_is_fitted()
        return dict(self.worst_group_summary_)

    def to_report(self) -> SubgroupReport:
        """Return a structured subgroup report."""

        self._check_is_fitted()
        return SubgroupReport(
            aggregate_summary=self.aggregate_summary_,
            performance_table=_frame_records(self.performance_table_),
            calibration_table=_frame_records(self.calibration_table_),
            shift_exposure_table=_frame_records(self.shift_exposure_table_),
            degradation_table=_frame_records(self.degradation_table_),
            stability_table=_frame_records(self.stability_table_),
            worst_group_summary=self.worst_group_summary_,
            warnings=self.warnings_,
        )

    def _build_grouping_context(
        self,
        ref_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        combined_frame: pd.DataFrame,
        feature: str | int,
        *,
        cached_groups: dict[str, tuple[pd.Series, pd.Series]],
    ) -> _GroupingContext:
        feature_name = _feature_spec_name(feature, ref_frame)
        if feature_name not in cached_groups:
            ref_groups = group_by_feature(
                ref_frame,
                feature,
                reference=combined_frame,
                categorical_features=self.categorical_features,
                n_bins=self.subgroup_bins,
                strategy=self.subgroup_strategy,
                max_categories=self.max_categories,
            )
            target_groups = group_by_feature(
                target_frame,
                feature,
                reference=combined_frame,
                categorical_features=self.categorical_features,
                n_bins=self.subgroup_bins,
                strategy=self.subgroup_strategy,
                max_categories=self.max_categories,
            )
            cached_groups[feature_name] = (ref_groups, target_groups)
        ref_groups, target_groups = cached_groups[feature_name]
        return _GroupingContext(feature_name, ref_groups, target_groups)

    def _build_intersection_context(
        self,
        ref_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        combined_frame: pd.DataFrame,
        features: tuple[str | int, ...],
        *,
        cached_groups: dict[str, tuple[pd.Series, pd.Series]],
    ) -> _GroupingContext:
        contexts = [
            self._build_grouping_context(
                ref_frame,
                target_frame,
                combined_frame,
                feature,
                cached_groups=cached_groups,
            )
            for feature in features
        ]
        intersection_name = " x ".join(context.name for context in contexts)
        ref_groups = contexts[0].reference_groups.copy()
        target_groups = contexts[0].target_groups.copy()
        for context in contexts[1:]:
            ref_groups = ref_groups.str.cat(context.reference_groups, sep=" & ")
            target_groups = target_groups.str.cat(context.target_groups, sep=" & ")
        ref_groups = _collapse_rare_groups(
            ref_groups,
            max_groups=self.max_intersection_groups,
            min_group_size=self.min_group_size,
            other_label=f"{intersection_name}=other_intersection",
        )
        target_groups = _collapse_rare_groups(
            target_groups,
            max_groups=self.max_intersection_groups,
            min_group_size=self.min_group_size,
            other_label=f"{intersection_name}=other_intersection",
        )
        return _GroupingContext(intersection_name, ref_groups, target_groups)

    def _evaluate_grouping(
        self,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        y_target: np.ndarray,
        p_target: np.ndarray,
        context: _GroupingContext,
        *,
        reference_weight: np.ndarray | None,
        target_weight: np.ndarray | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        reference_frame = group_metrics(
            y_ref,
            p_ref,
            context.reference_groups,
            sample_weight=reference_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
            dataset_name="reference",
            slice_name=context.name,
            min_group_size=self.min_group_size,
            min_class_count=self.min_class_count,
        )
        target_frame = group_metrics(
            y_target,
            p_target,
            context.target_groups,
            sample_weight=target_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
            dataset_name="target",
            slice_name=context.name,
            min_group_size=self.min_group_size,
            min_class_count=self.min_class_count,
        )
        performance_frame = pd.concat([reference_frame, target_frame], ignore_index=True)

        degradation_frame = reference_frame.merge(
            target_frame,
            on=["slice_name", "group"],
            how="outer",
            suffixes=("_reference", "_target"),
        )
        for prefix in ["reference", "target"]:
            degradation_frame[f"{prefix}_n_samples"] = degradation_frame[
                f"n_samples_{prefix}"
            ].fillna(0.0)
            degradation_frame[f"{prefix}_sample_share"] = degradation_frame[
                f"sample_share_{prefix}"
            ].fillna(0.0)
            degradation_frame[f"{prefix}_weight_share"] = degradation_frame[
                f"weight_share_{prefix}"
            ].fillna(0.0)
            degradation_frame[f"{prefix}_support_ok"] = degradation_frame[
                f"support_ok_{prefix}"
            ].fillna(False)

        degradation_frame["supported_both"] = (
            degradation_frame["reference_support_ok"] & degradation_frame["target_support_ok"]
        )
        degradation_frame["exposure_gap"] = (
            degradation_frame["target_sample_share"] - degradation_frame["reference_sample_share"]
        )
        degradation_frame["exposure_ratio"] = degradation_frame.apply(
            lambda row: _safe_ratio(
                row["target_sample_share"],
                row["reference_sample_share"],
            ),
            axis=1,
        )
        degradation_frame["delta_accuracy"] = (
            degradation_frame["accuracy_target"] - degradation_frame["accuracy_reference"]
        )
        degradation_frame["delta_error_rate"] = (
            degradation_frame["error_rate_target"] - degradation_frame["error_rate_reference"]
        )
        degradation_frame["delta_log_loss"] = (
            degradation_frame["log_loss_target"] - degradation_frame["log_loss_reference"]
        )
        degradation_frame["delta_brier_score"] = (
            degradation_frame["brier_score_target"] - degradation_frame["brier_score_reference"]
        )
        degradation_frame["delta_ece"] = (
            degradation_frame["ece_target"] - degradation_frame["ece_reference"]
        )
        degradation_frame["delta_mce"] = (
            degradation_frame["mce_target"] - degradation_frame["mce_reference"]
        )
        degradation_frame["delta_confidence_gap"] = (
            degradation_frame["confidence_gap_target"]
            - degradation_frame["confidence_gap_reference"]
        )
        degradation_frame["error_risk_ratio"] = degradation_frame.apply(
            lambda row: _safe_ratio(row["error_rate_target"], row["error_rate_reference"]),
            axis=1,
        )
        degradation_frame["ece_risk_ratio"] = degradation_frame.apply(
            lambda row: _safe_ratio(row["ece_target"], row["ece_reference"]),
            axis=1,
        )
        degradation_frame["severity_score"] = (
            np.maximum(degradation_frame["delta_error_rate"], 0.0)
            + np.maximum(degradation_frame["delta_log_loss"], 0.0)
            + np.maximum(degradation_frame["delta_ece"], 0.0)
        ) * degradation_frame["target_sample_share"].fillna(0.0)

        stability_row = {
            "slice_name": context.name,
            "n_groups": int(len(degradation_frame)),
            "supported_groups": int(degradation_frame["supported_both"].sum()),
            "target_supported_coverage": float(
                degradation_frame.loc[
                    degradation_frame["supported_both"], "target_sample_share"
                ].sum()
            ),
            "reference_supported_coverage": float(
                degradation_frame.loc[
                    degradation_frame["supported_both"], "reference_sample_share"
                ].sum()
            ),
            "max_target_group_share": float(degradation_frame["target_sample_share"].max()),
        }
        return performance_frame, degradation_frame, stability_row

    def _build_worst_group_summary(self) -> dict[str, Any]:
        analyzable = self.degradation_table_.loc[self.degradation_table_["supported_both"]].copy()
        if analyzable.empty:
            return {
                "worst_accuracy_group": "n/a",
                "worst_ece_group": "n/a",
                "highest_severity_group": "n/a",
            }

        worst_accuracy = analyzable.sort_values("accuracy_target", ascending=True).iloc[0]
        worst_ece = analyzable.sort_values("ece_target", ascending=False).iloc[0]
        highest_severity = analyzable.sort_values("severity_score", ascending=False).iloc[0]
        return {
            "worst_accuracy_group": f"{worst_accuracy['slice_name']} -> {worst_accuracy['group']}",
            "worst_accuracy_value": float(worst_accuracy["accuracy_target"]),
            "worst_ece_group": f"{worst_ece['slice_name']} -> {worst_ece['group']}",
            "worst_ece_value": float(worst_ece["ece_target"]),
            "highest_severity_group": (
                f"{highest_severity['slice_name']} -> {highest_severity['group']}"
            ),
            "highest_severity_value": float(highest_severity["severity_score"]),
        }

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "degradation_table_"):
            raise ValueError("SubgroupAnalyzer must be fitted before accessing results.")


def _feature_spec_name(feature: str | int | tuple[str | int, ...], frame: pd.DataFrame) -> str:
    if isinstance(feature, tuple):
        return " x ".join(_resolve_feature_name(frame, item) for item in feature)
    return _resolve_feature_name(frame, feature)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _warnings_from_frame(frame: pd.DataFrame, slice_name: str) -> list[str]:
    flagged = frame.loc[~frame["supported_both"]]
    if flagged.empty:
        return []
    return [
        (
            f"{slice_name} has {len(flagged)} subgroup row(s) below the default support "
            "thresholds; subgroup-level calibration and disparity estimates may be unstable."
        )
    ]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
