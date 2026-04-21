"""Conditional reliability auditing under deployment shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from ..metrics import confidence_conditioned_error
from ..reliability.analyzer import ReliabilityAnalyzer
from ..subgroup import SubgroupAnalyzer, group_by_feature, group_metrics
from ..utils.schema import align_tabular_inputs, extract_feature_names, infer_feature_types
from .discovery import SliceDiscoverer
from .results import AuditReport


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-12 or pd.isna(denominator):
        return float("nan")
    return float(numerator / denominator)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    feature_types = infer_feature_types(X)
    categorical_columns = [name for name, kind in feature_types.items() if kind == "categorical"]
    continuous_columns = [name for name, kind in feature_types.items() if kind == "continuous"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )
    if continuous_columns:
        transformers.append(("continuous", "passthrough", continuous_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _merge_group_tables(reference_frame: pd.DataFrame, target_frame: pd.DataFrame) -> pd.DataFrame:
    merged = reference_frame.merge(
        target_frame,
        on=["slice_name", "group"],
        how="outer",
        suffixes=("_reference", "_target"),
    )
    for prefix in ["reference", "target"]:
        merged[f"{prefix}_n_samples"] = merged[f"n_samples_{prefix}"].fillna(0.0)
        merged[f"{prefix}_sample_share"] = merged[f"sample_share_{prefix}"].fillna(0.0)
        merged[f"{prefix}_support_ok"] = merged[f"support_ok_{prefix}"].fillna(False)

    merged["supported_both"] = merged["reference_support_ok"] & merged["target_support_ok"]
    merged["delta_accuracy"] = merged["accuracy_target"] - merged["accuracy_reference"]
    merged["delta_error_rate"] = merged["error_rate_target"] - merged["error_rate_reference"]
    merged["delta_log_loss"] = merged["log_loss_target"] - merged["log_loss_reference"]
    merged["delta_ece"] = merged["ece_target"] - merged["ece_reference"]
    merged["exposure_gap"] = merged["target_sample_share"] - merged["reference_sample_share"]
    merged["exposure_ratio"] = merged.apply(
        lambda row: _safe_ratio(row["target_sample_share"], row["reference_sample_share"]),
        axis=1,
    )
    return merged


class ConditionalReliabilityAuditor:
    """Audit conditional reliability, subgroup disparities, and failure localization."""

    def __init__(
        self,
        *,
        n_bins: int = 10,
        subgroup_bins: int = 4,
        shift_severity_bins: int = 4,
        strategy: str = "uniform",
        subgroup_strategy: str = "quantile",
        min_group_size: int = 30,
        min_class_count: int = 8,
        max_categories: int = 8,
        max_intersection_groups: int = 20,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.subgroup_bins = subgroup_bins
        self.shift_severity_bins = shift_severity_bins
        self.strategy = strategy
        self.subgroup_strategy = subgroup_strategy
        self.min_group_size = min_group_size
        self.min_class_count = min_class_count
        self.max_categories = max_categories
        self.max_intersection_groups = max_intersection_groups
        self.random_state = random_state

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
        discover_slices: bool = True,
        slice_objective: str = "error",
    ) -> ConditionalReliabilityAuditor:
        """Fit an end-to-end conditional reliability audit."""

        X_ref_aligned, X_target_aligned = align_tabular_inputs(X_ref, X_target)
        ref_frame = _as_frame(X_ref_aligned)
        target_frame = _as_frame(X_target_aligned)

        aggregate = ReliabilityAnalyzer(n_bins=self.n_bins, strategy=self.strategy).fit(
            y_ref,
            p_ref,
            y_target,
            p_target,
            reference_weight=reference_weight,
            target_weight=target_weight,
        )
        self.aggregate_analyzer_ = aggregate

        subgroup_analyzer = SubgroupAnalyzer(
            n_bins=self.n_bins,
            strategy=self.strategy,
            subgroup_bins=self.subgroup_bins,
            subgroup_strategy=self.subgroup_strategy,
            max_categories=self.max_categories,
            max_intersection_groups=self.max_intersection_groups,
            min_group_size=self.min_group_size,
            min_class_count=self.min_class_count,
        ).fit(
            ref_frame,
            y_ref,
            p_ref,
            target_frame,
            y_target,
            p_target,
            subgroup_features=subgroup_features,
            intersectional_features=intersectional_features,
            reference_weight=reference_weight,
            target_weight=target_weight,
        )
        self.subgroup_analyzer_ = subgroup_analyzer
        self.subgroup_report_ = subgroup_analyzer.to_report()

        self.conditional_error_table_ = self._build_conditional_error_table(
            y_ref,
            p_ref,
            y_target,
            p_target,
            reference_weight=reference_weight,
            target_weight=target_weight,
        )
        self.shift_severity_table_ = self._build_shift_severity_table(
            ref_frame,
            y_ref,
            p_ref,
            target_frame,
            y_target,
            p_target,
            reference_weight=reference_weight,
            target_weight=target_weight,
        )
        self.disparity_summary_ = self._build_disparity_summary()
        self.aggregate_vs_subgroup_table_ = self._build_aggregate_vs_subgroup_table()
        self.heatmap_table_ = self._build_heatmap_table()

        if discover_slices:
            discoverer = SliceDiscoverer(
                max_depth=2,
                min_samples_leaf=self.min_group_size,
                n_bins=self.n_bins,
                random_state=self.random_state,
            ).fit(
                ref_frame,
                y_ref,
                p_ref,
                target_frame,
                y_target,
                p_target,
                objective=slice_objective,
                reference_weight=reference_weight,
                target_weight=target_weight,
            )
            self.slice_discoverer_ = discoverer
            self.discovered_slices_ = discoverer.summary()
        else:
            self.discovered_slices_ = pd.DataFrame()

        self.hidden_failure_flags_ = self._build_hidden_failure_flags()
        self.caveats_ = self._build_caveats()
        self.operational_implications_ = self._build_operational_implications()
        self.aggregate_summary_ = {
            "reference_accuracy": float(aggregate.reference_profile_.accuracy),
            "target_accuracy": float(aggregate.target_profile_.accuracy),
            "delta_accuracy": float(aggregate.degradation_summary_.delta_accuracy),
            "reference_ece": float(aggregate.reference_profile_.ece),
            "target_ece": float(aggregate.target_profile_.ece),
            "delta_ece": float(aggregate.degradation_summary_.delta_ece),
            "reference_log_loss": float(aggregate.reference_profile_.log_loss),
            "target_log_loss": float(aggregate.target_profile_.log_loss),
            "delta_log_loss": float(aggregate.degradation_summary_.delta_log_loss),
        }
        return self

    def conditional_error_table(self) -> pd.DataFrame:
        """Return confidence-conditioned error summaries."""

        self._check_is_fitted()
        return self.conditional_error_table_.copy()  # type: ignore[no-any-return]

    def shift_severity_table(self) -> pd.DataFrame:
        """Return performance by shift-severity slices."""

        self._check_is_fitted()
        return self.shift_severity_table_.copy()  # type: ignore[no-any-return]

    def disparity_summary(self) -> pd.DataFrame:
        """Return worst-versus-best subgroup disparity summaries."""

        self._check_is_fitted()
        return self.disparity_summary_.copy()  # type: ignore[no-any-return]

    def aggregate_vs_subgroup_summary(self) -> pd.DataFrame:
        """Return aggregate versus worst-group comparison rows."""

        self._check_is_fitted()
        return self.aggregate_vs_subgroup_table_.copy()  # type: ignore[no-any-return]

    def heatmap_table(self) -> pd.DataFrame:
        """Return long-format subgroup x metric audit data."""

        self._check_is_fitted()
        return self.heatmap_table_.copy()  # type: ignore[no-any-return]

    def discovered_slices(self) -> pd.DataFrame:
        """Return automatically discovered failure slices."""

        self._check_is_fitted()
        return self.discovered_slices_.copy()  # type: ignore[no-any-return]

    def to_report(self) -> AuditReport:
        """Return a structured audit report."""

        self._check_is_fitted()
        return AuditReport(
            aggregate_summary=self.aggregate_summary_,
            subgroup_report=self.subgroup_report_.to_dict(),
            conditional_error_table=_frame_records(self.conditional_error_table_),
            shift_severity_table=_frame_records(self.shift_severity_table_),
            disparity_summary=_frame_records(self.disparity_summary_),
            aggregate_vs_subgroup_table=_frame_records(self.aggregate_vs_subgroup_table_),
            heatmap_table=_frame_records(self.heatmap_table_),
            discovered_slices=_frame_records(self.discovered_slices_),
            hidden_failure_flags=self.hidden_failure_flags_,
            caveats=self.caveats_,
            operational_implications=self.operational_implications_,
        )

    def _build_conditional_error_table(
        self,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        y_target: np.ndarray,
        p_target: np.ndarray,
        *,
        reference_weight: np.ndarray | None,
        target_weight: np.ndarray | None,
    ) -> pd.DataFrame:
        reference_frame = confidence_conditioned_error(
            y_ref,
            p_ref,
            n_bins=self.n_bins,
            sample_weight=reference_weight,
        ).assign(dataset="reference")
        target_frame = confidence_conditioned_error(
            y_target,
            p_target,
            n_bins=self.n_bins,
            sample_weight=target_weight,
        ).assign(dataset="target")
        return pd.concat([reference_frame, target_frame], ignore_index=True)  # type: ignore[no-any-return]

    def _build_shift_severity_table(
        self,
        ref_frame: pd.DataFrame,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        target_frame: pd.DataFrame,
        y_target: np.ndarray,
        p_target: np.ndarray,
        *,
        reference_weight: np.ndarray | None,
        target_weight: np.ndarray | None,
    ) -> pd.DataFrame:
        ref_scores, target_scores = self._estimate_shift_scores(ref_frame, target_frame)
        ref_score_frame = pd.DataFrame({"shift_severity": ref_scores})
        target_score_frame = pd.DataFrame({"shift_severity": target_scores})
        combined_score_frame = pd.concat(
            [ref_score_frame, target_score_frame],
            axis=0,
            ignore_index=True,
        )
        reference_groups = group_by_feature(
            ref_score_frame,
            "shift_severity",
            reference=combined_score_frame,
            n_bins=self.shift_severity_bins,
            strategy="quantile",
        )
        target_groups = group_by_feature(
            target_score_frame,
            "shift_severity",
            reference=combined_score_frame,
            n_bins=self.shift_severity_bins,
            strategy="quantile",
        )

        reference_metrics = group_metrics(
            y_ref,
            p_ref,
            reference_groups,
            sample_weight=reference_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
            dataset_name="reference",
            slice_name="shift_severity",
            min_group_size=self.min_group_size,
            min_class_count=self.min_class_count,
        )
        target_metrics = group_metrics(
            y_target,
            p_target,
            target_groups,
            sample_weight=target_weight,
            n_bins=self.n_bins,
            strategy=self.strategy,
            dataset_name="target",
            slice_name="shift_severity",
            min_group_size=self.min_group_size,
            min_class_count=self.min_class_count,
        )
        return _merge_group_tables(reference_metrics, target_metrics)  # type: ignore[no-any-return]

    def _estimate_shift_scores(
        self,
        ref_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        combined = pd.concat([ref_frame, target_frame], axis=0, ignore_index=True)
        y_domain = np.concatenate(
            [np.zeros(len(ref_frame), dtype=int), np.ones(len(target_frame), dtype=int)]
        )
        preprocessor = _build_preprocessor(combined)
        X_encoded = preprocessor.fit_transform(combined)
        classifier = LogisticRegression(max_iter=3000, random_state=self.random_state)
        classifier.fit(X_encoded, y_domain)
        scores = classifier.predict_proba(X_encoded)[:, 1]
        return scores[: len(ref_frame)], scores[len(ref_frame) :]

    def _build_disparity_summary(self) -> pd.DataFrame:
        target_performance = self.subgroup_analyzer_.performance_table(dataset="target")
        target_degradation = self.subgroup_analyzer_.degradation_ranking()
        supported = target_performance.loc[target_performance["support_ok"]].copy()
        if supported.empty:
            return pd.DataFrame.from_records([])

        rows: list[dict[str, Any]] = []
        for metric, higher_is_worse in [
            ("accuracy", False),
            ("error_rate", True),
            ("ece", True),
            ("log_loss", True),
        ]:
            if higher_is_worse:
                worst = supported.sort_values(metric, ascending=False).iloc[0]
                best = supported.sort_values(metric, ascending=True).iloc[0]
                absolute_gap = float(worst[metric] - best[metric])
                risk_ratio = _safe_ratio(float(worst[metric]), float(best[metric]))
            else:
                worst = supported.sort_values(metric, ascending=True).iloc[0]
                best = supported.sort_values(metric, ascending=False).iloc[0]
                absolute_gap = float(best[metric] - worst[metric])
                risk_ratio = _safe_ratio(float(worst[metric]), float(best[metric]))

            match = target_degradation.loc[
                (target_degradation["slice_name"] == worst["slice_name"])
                & (target_degradation["group"] == worst["group"])
            ]
            worst_delta = (
                float(match[f"delta_{metric}"].iloc[0]) if not match.empty else float("nan")
            )
            rows.append(
                {
                    "metric": metric,
                    "worst_group": f"{worst['slice_name']} -> {worst['group']}",
                    "best_group": f"{best['slice_name']} -> {best['group']}",
                    "worst_target_value": float(worst[metric]),
                    "best_target_value": float(best[metric]),
                    "absolute_gap": absolute_gap,
                    "risk_ratio": risk_ratio,
                    "worst_group_delta": worst_delta,
                }
            )
        return pd.DataFrame.from_records(rows)  # type: ignore[no-any-return]

    def _build_aggregate_vs_subgroup_table(self) -> pd.DataFrame:
        target = self.aggregate_analyzer_.target_profile_
        degradation = self.aggregate_analyzer_.degradation_summary_
        aggregate_metrics = {
            "accuracy": (target.accuracy, degradation.delta_accuracy),
            "error_rate": (target.error_rate, degradation.delta_error_rate),
            "ece": (target.ece, degradation.delta_ece),
            "log_loss": (target.log_loss, degradation.delta_log_loss),
        }

        rows: list[dict[str, Any]] = []
        disparity_frame = self.disparity_summary_
        for metric, (aggregate_value, aggregate_delta) in aggregate_metrics.items():
            disparity = disparity_frame.loc[disparity_frame["metric"] == metric]
            if disparity.empty:
                continue
            record = disparity.iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "aggregate_target_value": float(aggregate_value),
                    "worst_group_target_value": float(record["worst_target_value"]),
                    "absolute_gap": float(abs(record["worst_target_value"] - aggregate_value)),
                    "aggregate_delta": float(aggregate_delta),
                    "worst_group_delta": float(record["worst_group_delta"]),
                    "worst_group": record["worst_group"],
                }
            )
        return pd.DataFrame.from_records(rows)  # type: ignore[no-any-return]

    def _build_heatmap_table(self) -> pd.DataFrame:
        degradation = self.subgroup_analyzer_.degradation_ranking()
        metric_columns = {
            "accuracy": ("accuracy_reference", "accuracy_target", "delta_accuracy"),
            "error_rate": ("error_rate_reference", "error_rate_target", "delta_error_rate"),
            "ece": ("ece_reference", "ece_target", "delta_ece"),
            "log_loss": ("log_loss_reference", "log_loss_target", "delta_log_loss"),
        }
        records: list[dict[str, Any]] = []
        for metric, (reference_col, target_col, delta_col) in metric_columns.items():
            subset = degradation.loc[
                :,
                [
                    "slice_name",
                    "group",
                    "supported_both",
                    "severity_score",
                    "target_sample_share",
                    reference_col,
                    target_col,
                    delta_col,
                ],
            ].copy()
            subset["metric"] = metric
            subset = subset.rename(
                columns={
                    reference_col: "reference_value",
                    target_col: "target_value",
                    delta_col: "delta_value",
                }
            )
            records.extend(subset.to_dict(orient="records"))
        return pd.DataFrame.from_records(records)  # type: ignore[no-any-return]

    def _build_hidden_failure_flags(self) -> dict[str, bool]:
        degradation = self.subgroup_analyzer_.degradation_ranking()
        supported = degradation.loc[degradation["supported_both"]].copy()
        aggregate = self.aggregate_analyzer_.degradation_summary_
        worst_delta_accuracy = (
            float(supported["delta_accuracy"].min()) if not supported.empty else 0.0
        )
        worst_delta_ece = float(supported["delta_ece"].max()) if not supported.empty else 0.0
        supported_coverage = float(
            self.subgroup_analyzer_.stability_diagnostics()["target_supported_coverage"].mean()
        )
        top_slice_failure_share = (
            float(self.discovered_slices_["target_failure_share"].max())
            if not self.discovered_slices_.empty
            else 0.0
        )
        top_slice_concentration = (
            float(self.discovered_slices_["failure_concentration"].max())
            if not self.discovered_slices_.empty
            else 0.0
        )
        return {
            "masked_accuracy_drop": aggregate.delta_accuracy > -0.03
            and worst_delta_accuracy < -0.08,
            "masked_calibration_drift": aggregate.delta_ece < 0.02 and worst_delta_ece > 0.05,
            "concentrated_failures": (
                top_slice_failure_share > 0.35 or top_slice_concentration > 1.75
            ),
            "limited_support_coverage": supported_coverage < 0.8,
        }

    def _build_caveats(self) -> list[str]:
        caveats = [
            (
                "Subgroup and slice diagnostics are descriptive deployment-risk audits, "
                "not fairness-compliance determinations."
            ),
            (
                "Shift-severity bins are based on an in-sample domain classifier and "
                "should be interpreted as relative exposure scores."
            ),
        ]
        caveats.extend(self.subgroup_report_.warnings)
        if hasattr(self, "slice_discoverer_"):
            caveats.extend(self.slice_discoverer_.caveats_)
        if self.hidden_failure_flags_.get("limited_support_coverage", False):
            caveats.append(
                "A meaningful share of target samples falls below the default subgroup "
                "support thresholds."
            )
        return _deduplicate_preserve_order(caveats)

    def _build_operational_implications(self) -> list[str]:
        implications: list[str] = []
        if self.hidden_failure_flags_.get("masked_accuracy_drop", False):
            implications.append(
                "Aggregate accuracy looks materially safer than at least one supported "
                "subgroup, so monitoring should include the worst-group slice."
            )
        if self.hidden_failure_flags_.get("masked_calibration_drift", False):
            implications.append(
                "Aggregate calibration understates subgroup calibration drift; "
                "thresholding or triage policies should be stress-tested on the "
                "affected slices."
            )
        if not self.discovered_slices_.empty:
            top_slice = self.discovered_slices_.iloc[0]
            implications.append(
                "The highest-concentration failure slice is "
                f"{top_slice['slice_label']} with rule `{top_slice['rule']}`."
            )
        if self.hidden_failure_flags_.get("limited_support_coverage", False):
            implications.append(
                "Several conclusions rest on incomplete subgroup support, so additional "
                "targeted data collection would improve audit coverage."
            )
        return implications

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "aggregate_summary_"):
            raise ValueError("ConditionalReliabilityAuditor must be fitted before use.")


ReliabilityAuditor = ConditionalReliabilityAuditor


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
