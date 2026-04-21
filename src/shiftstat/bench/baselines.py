"""Default baseline implementations for ShiftStat V5 benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from shiftstat.audit import ReliabilityAuditor
from shiftstat.bench.registry import BaselineRegistry
from shiftstat.reliability import evaluate_under_shift
from shiftstat.reliability.workflow import _resolve_binary_probabilities
from shiftstat.selective import evaluate_selective_under_shift


def default_baseline_registry() -> BaselineRegistry:
    """Return the default registry of benchmark baselines."""

    registry = BaselineRegistry()
    registry.register(
        "raw_model",
        _raw_model_baseline,
        description="Uncorrected reference-trained model evaluation on the target distribution.",
        category="reliability",
    )
    registry.register(
        "weighting_only",
        _weighting_only_baseline,
        description="Importance-weighted reliability evaluation without recalibration.",
        category="reliability",
    )
    registry.register(
        "recalibration_only",
        _recalibration_only_baseline,
        description=(
            "Reference-fit model followed by post-hoc calibration on held-out reference data."
        ),
        category="reliability",
    )
    registry.register(
        "weighting_and_recalibration",
        _weighting_and_recalibration_baseline,
        description="Importance weighting combined with post-hoc recalibration.",
        category="reliability",
    )
    registry.register(
        "subgroup_audit",
        _subgroup_audit_baseline,
        description=(
            "Subgroup-aware audit highlighting worst-group degradation and failure concentration."
        ),
        category="audit",
    )
    registry.register(
        "confidence_abstention",
        _confidence_abstention_baseline,
        description=(
            "Confidence-threshold abstention tuned for target coverage on "
            "reference validation data."
        ),
        category="selective",
    )
    registry.register(
        "weighted_confidence_abstention",
        _weighted_confidence_abstention_baseline,
        description="Confidence-threshold abstention with weighted threshold tuning under shift.",
        category="selective",
    )
    registry.register(
        "recalibrated_confidence_abstention",
        _recalibrated_confidence_abstention_baseline,
        description="Selective prediction with weighting-aware recalibration before abstention.",
        category="selective",
    )
    registry.register(
        "learned_risk_abstention",
        _learned_risk_abstention_baseline,
        description="Simple learned-risk rejector baseline using low-dimensional policy features.",
        category="selective",
    )
    return registry


def _raw_model_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _reliability_baseline(
        case,
        seed=seed,
        apply_importance_weighting=False,
        recalibration=None,
    )


def _weighting_only_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _reliability_baseline(
        case,
        seed=seed,
        apply_importance_weighting=True,
        recalibration=None,
    )


def _recalibration_only_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _reliability_baseline(
        case,
        seed=seed,
        apply_importance_weighting=False,
        recalibration=_recalibration_method(case),
    )


def _weighting_and_recalibration_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _reliability_baseline(
        case,
        seed=seed,
        apply_importance_weighting=True,
        recalibration=_recalibration_method(case),
    )


def _subgroup_audit_baseline(case: Any, seed: int) -> dict[str, Any]:
    estimator = clone(case.estimator)
    estimator.fit(case.X_ref, case.y_ref)
    p_ref = _resolve_binary_probabilities(estimator, case.X_ref)
    p_target = _resolve_binary_probabilities(estimator, case.X_target)

    subgroup_features = case.subgroup_features or _default_subgroup_features(case)
    intersectional_features = case.intersectional_features or []
    auditor = ReliabilityAuditor(
        min_group_size=int(case.metadata.get("min_group_size", 30)),
        random_state=seed,
    ).fit(
        case.X_ref,
        case.y_ref,
        p_ref,
        case.X_target,
        case.y_target,
        p_target,
        subgroup_features=subgroup_features,
        intersectional_features=intersectional_features,
    )
    report = auditor.to_report()
    comparison = auditor.aggregate_vs_subgroup_summary()
    accuracy_row = _row_or_empty(comparison, "accuracy")
    ece_row = _row_or_empty(comparison, "ece")
    top_slice = auditor.discovered_slices().head(1)
    top_slice_rule = str(top_slice["rule"].iloc[0]) if not top_slice.empty else "n/a"
    return {
        "target_accuracy": report.aggregate_summary["target_accuracy"],
        "delta_accuracy": report.aggregate_summary["delta_accuracy"],
        "target_ece": report.aggregate_summary["target_ece"],
        "delta_ece": report.aggregate_summary["delta_ece"],
        "delta_log_loss": report.aggregate_summary["delta_log_loss"],
        "worst_group_accuracy_gap": accuracy_row.get("absolute_gap", float("nan")),
        "worst_group_ece_gap": ece_row.get("absolute_gap", float("nan")),
        "worst_group_delta_accuracy": accuracy_row.get("worst_group_delta", float("nan")),
        "worst_group_delta_ece": ece_row.get("worst_group_delta", float("nan")),
        "masked_accuracy_drop": bool(report.hidden_failure_flags["masked_accuracy_drop"]),
        "masked_calibration_drift": bool(report.hidden_failure_flags["masked_calibration_drift"]),
        "concentrated_failures": bool(report.hidden_failure_flags["concentrated_failures"]),
        "top_discovered_slice": top_slice_rule,
    }


def _confidence_abstention_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _selective_baseline(
        case,
        seed=seed,
        policy_method="confidence",
        apply_importance_weighting=False,
        use_weighted_threshold_tuning=False,
        recalibration=None,
    )


def _weighted_confidence_abstention_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _selective_baseline(
        case,
        seed=seed,
        policy_method="confidence",
        apply_importance_weighting=True,
        use_weighted_threshold_tuning=True,
        recalibration=None,
    )


def _recalibrated_confidence_abstention_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _selective_baseline(
        case,
        seed=seed,
        policy_method="confidence",
        apply_importance_weighting=True,
        use_weighted_threshold_tuning=True,
        recalibration=_recalibration_method(case),
        selection_probability_source="recalibrated",
    )


def _learned_risk_abstention_baseline(case: Any, seed: int) -> dict[str, Any]:
    return _selective_baseline(
        case,
        seed=seed,
        policy_method="learned_risk",
        apply_importance_weighting=False,
        use_weighted_threshold_tuning=False,
        recalibration=None,
    )


def _reliability_baseline(
    case: Any,
    *,
    seed: int,
    apply_importance_weighting: bool,
    recalibration: str | None,
) -> dict[str, Any]:
    result = evaluate_under_shift(
        case.estimator,
        case.X_ref,
        case.y_ref,
        case.X_target,
        case.y_target,
        apply_importance_weighting=apply_importance_weighting,
        weighting_method=_weighting_method(case),
        recalibration=recalibration,
        categorical_features=case.categorical_features,
        random_state=seed,
    )
    record = result.summary_frame().iloc[0].to_dict()
    if result.weighting_summary is not None:
        record["effective_sample_size"] = result.weighting_summary["effective_sample_size"]
    return {
        str(key): _scalarize(value) for key, value in record.items() if str(key) != "estimator_name"
    }


def _selective_baseline(
    case: Any,
    *,
    seed: int,
    policy_method: str,
    apply_importance_weighting: bool,
    use_weighted_threshold_tuning: bool,
    recalibration: str | None,
    selection_probability_source: str = "raw",
) -> dict[str, Any]:
    result = evaluate_selective_under_shift(
        case.estimator,
        case.X_ref,
        case.y_ref,
        case.X_target,
        case.y_target,
        policy_method=policy_method,
        target_coverage=float(case.metadata.get("target_coverage", 0.8)),
        apply_importance_weighting=apply_importance_weighting,
        weighting_method=_weighting_method(case),
        use_weighted_threshold_tuning=use_weighted_threshold_tuning,
        compare_threshold_tuning=True,
        recalibration=recalibration,
        selection_probability_source=selection_probability_source,
        categorical_features=case.categorical_features,
        subgroup_features=case.subgroup_features,
        intersectional_features=case.intersectional_features,
        random_state=seed,
    )
    record = result.summary_frame().iloc[0].to_dict()
    disparity_frame = pd.DataFrame.from_records(result.subgroup_abstention_disparities)
    abstention_gap = float("nan")
    if not disparity_frame.empty:
        match = disparity_frame.loc[
            disparity_frame["metric"] == "target_abstention_rate",
            "absolute_gap",
        ]
        if not match.empty:
            abstention_gap = float(match.iloc[0])
    record["subgroup_abstention_gap"] = abstention_gap
    return {
        str(key): _scalarize(value) for key, value in record.items() if str(key) != "estimator_name"
    }


def _weighting_method(case: Any) -> str:
    return str(case.metadata.get("weighting_method", "domain_classifier"))


def _recalibration_method(case: Any) -> str:
    return str(case.metadata.get("recalibration_method", "temperature"))


def _row_or_empty(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    if frame.empty:
        return {}
    match = frame.loc[frame["metric"] == metric]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {str(key): _scalarize(value) for key, value in row.to_dict().items()}


def _default_subgroup_features(case: Any) -> list[str | int]:
    if getattr(case, "categorical_features", None):
        return list(case.categorical_features)
    if hasattr(case.X_ref, "columns"):
        return [str(column) for column in list(case.X_ref.columns)[: min(2, case.X_ref.shape[1])]]
    return [0, 1]


def _scalarize(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
