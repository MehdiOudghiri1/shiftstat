"""Workflow integration for selective prediction under shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from shiftstat.detect import ShiftDetector
from shiftstat.reliability.analyzer import ReliabilityAnalyzer
from shiftstat.reliability.workflow import _make_calibrator, _resolve_binary_probabilities
from shiftstat.reweight import ImportanceWeighter
from shiftstat.selective.policy import AbstentionPolicy
from shiftstat.selective.predictor import SelectivePredictor, abstention_summary_by_groups
from shiftstat.selective.results import SelectiveShiftEvaluationResult
from shiftstat.utils.random import random_state_to_int


def evaluate_selective_under_shift(
    estimator: Any,
    X_ref: Any,
    y_ref: np.ndarray,
    X_target: Any,
    y_target: np.ndarray,
    *,
    estimator_is_fitted: bool = False,
    policy: AbstentionPolicy | None = None,
    policy_method: str = "confidence",
    policy_threshold: float | None = None,
    target_coverage: float | None = None,
    target_risk: float | None = None,
    selection_fraction: float = 0.25,
    apply_importance_weighting: bool = False,
    weighting_method: str = "domain_classifier",
    use_weighted_threshold_tuning: bool = False,
    compare_threshold_tuning: bool = True,
    recalibration: str | None = None,
    selection_probability_source: str = "raw",
    categorical_features: list[str] | list[int] | None = None,
    subgroup_features: list[str] | list[int] | None = None,
    intersectional_features: list[tuple[str | int, ...]] | None = None,
    n_bins: int = 10,
    random_state: int | np.random.RandomState | None = None,
) -> SelectiveShiftEvaluationResult:
    """Evaluate selective prediction tradeoffs under reference-to-target shift."""

    if selection_probability_source not in {"raw", "recalibrated"}:
        raise ValueError("selection_probability_source must be 'raw' or 'recalibrated'.")

    seed = random_state_to_int(random_state)
    base_policy = (
        policy.copy()
        if policy is not None
        else AbstentionPolicy(
            method=policy_method,
            threshold=policy_threshold,
            random_state=random_state,
        )
    )

    requires_holdout = (
        recalibration is not None
        or target_coverage is not None
        or target_risk is not None
        or base_policy.method == "learned_risk"
    )
    if estimator_is_fitted:
        fitted_estimator = estimator
        X_eval_ref = X_ref
        y_eval_ref = np.asarray(y_ref)
    elif requires_holdout:
        X_train_ref, X_eval_ref, y_train_ref, y_eval_ref = train_test_split(
            X_ref,
            y_ref,
            test_size=selection_fraction,
            stratify=y_ref,
            random_state=seed,
        )
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(X_train_ref, y_train_ref)
    else:
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(X_ref, y_ref)
        X_eval_ref = X_ref
        y_eval_ref = np.asarray(y_ref)

    p_ref_raw = _resolve_binary_probabilities(fitted_estimator, X_eval_ref)
    p_target_raw = _resolve_binary_probabilities(fitted_estimator, X_target)

    detector = ShiftDetector(
        categorical_features=categorical_features,
        random_state=random_state,
    ).fit(X_eval_ref, X_target)

    reference_weight = None
    weighting_summary = None
    if apply_importance_weighting:
        weighter = ImportanceWeighter(
            method=weighting_method,
            categorical_features=categorical_features,
            random_state=random_state,
        ).fit(X_eval_ref, X_target)
        reference_weight = weighter.predict_weights(X_eval_ref)
        weighting_summary = weighter.summary()

    reliability = ReliabilityAnalyzer(n_bins=n_bins)
    reliability.fit(
        y_eval_ref,
        p_ref_raw,
        y_target,
        p_target_raw,
        reference_weight=reference_weight,
    )

    p_ref_calibrated = None
    p_target_calibrated = None
    recalibration_summary = None
    if recalibration is not None:
        calibrator = _make_calibrator(recalibration)
        calibrator.fit(p_ref_raw, y_eval_ref, sample_weight=reference_weight)
        p_ref_calibrated = calibrator.predict_proba(p_ref_raw)
        p_target_calibrated = calibrator.predict_proba(p_target_raw)
        recalibration_summary = {
            "method": recalibration,
            "fit_sample_size": int(len(y_eval_ref)),
        }

    selection_ref_prob = (
        p_ref_calibrated
        if selection_probability_source == "recalibrated" and p_ref_calibrated is not None
        else p_ref_raw
    )
    selection_target_prob = (
        p_target_calibrated
        if selection_probability_source == "recalibrated" and p_target_calibrated is not None
        else p_target_raw
    )
    tuning_weights = reference_weight if use_weighted_threshold_tuning else None

    predictor = SelectivePredictor(base_policy, n_bins=n_bins)
    predictor.fit(
        y_eval_ref,
        selection_ref_prob,
        X=X_eval_ref,
        sample_weight=tuning_weights,
        threshold=policy_threshold,
        target_coverage=target_coverage,
        target_risk=target_risk,
    )
    reference_selective_profile = predictor.evaluate(
        y_eval_ref,
        selection_ref_prob,
        X=X_eval_ref,
        sample_weight=reference_weight,
        name="reference",
    )
    target_selective_profile = predictor.evaluate(
        y_target,
        selection_target_prob,
        X=X_target,
        sample_weight=None,
        name="target",
    )
    target_curve = predictor.risk_coverage_curve(
        y_target,
        selection_target_prob,
        X=X_target,
        sample_weight=None,
        name="target",
    )

    subgroup_summary = _merge_subgroup_abstention_summaries(
        abstention_summary_by_groups(
            X_eval_ref,
            y_eval_ref,
            selection_ref_prob,
            predictor.policy,
            sample_weight=reference_weight,
            dataset_name="reference",
            subgroup_features=subgroup_features,
            intersectional_features=intersectional_features,
        ),
        abstention_summary_by_groups(
            X_target,
            y_target,
            selection_target_prob,
            predictor.policy,
            sample_weight=None,
            dataset_name="target",
            subgroup_features=subgroup_features,
            intersectional_features=intersectional_features,
        ),
    )
    subgroup_disparities = _subgroup_abstention_disparities(subgroup_summary)

    threshold_comparison = None
    if compare_threshold_tuning and reference_weight is not None and (
        target_coverage is not None or target_risk is not None or base_policy.method == "learned_risk"
    ):
        alternative_policy = base_policy.copy()
        alternative_predictor = SelectivePredictor(alternative_policy, n_bins=n_bins)
        alternative_predictor.fit(
            y_eval_ref,
            selection_ref_prob,
            X=X_eval_ref,
            sample_weight=None if use_weighted_threshold_tuning else reference_weight,
            threshold=policy_threshold,
            target_coverage=target_coverage,
            target_risk=target_risk,
        )
        threshold_comparison = [
            predictor.policy.summary(),
            alternative_predictor.policy.summary(),
        ]

    recalibrated_target_selective_profile = None
    if p_target_calibrated is not None:
        recalibrated_policy = base_policy.copy()
        recalibrated_predictor = SelectivePredictor(recalibrated_policy, n_bins=n_bins)
        recalibrated_predictor.fit(
            y_eval_ref,
            p_ref_calibrated if p_ref_calibrated is not None else p_ref_raw,
            X=X_eval_ref,
            sample_weight=tuning_weights,
            threshold=policy_threshold,
            target_coverage=target_coverage,
            target_risk=target_risk,
        )
        recalibrated_target_selective_profile = recalibrated_predictor.evaluate(
            y_target,
            p_target_calibrated,
            X=X_target,
            sample_weight=None,
            name="target_recalibrated",
        )

    policy_summary = predictor.policy.summary()
    policy_summary["probability_source"] = selection_probability_source
    return SelectiveShiftEvaluationResult(
        estimator_name=fitted_estimator.__class__.__name__,
        dataset_shift_overview=detector.dataset_summary_.to_dict(),
        policy_summary=policy_summary,
        reference_full_profile=reliability.reference_profile_,
        target_full_profile=reliability.target_profile_,
        reference_selective_profile=reference_selective_profile,
        target_selective_profile=target_selective_profile,
        target_risk_coverage_curve=target_curve,
        subgroup_abstention_summary=_frame_records(subgroup_summary),
        subgroup_abstention_disparities=_frame_records(subgroup_disparities),
        threshold_comparison=threshold_comparison,
        weighting_summary=weighting_summary,
        recalibration_summary=recalibration_summary,
        recalibrated_target_selective_profile=recalibrated_target_selective_profile,
    )


def _merge_subgroup_abstention_summaries(
    reference_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = reference_frame.merge(
        target_frame,
        on=["slice_name", "group"],
        how="outer",
        suffixes=("_reference", "_target"),
    )
    for prefix in ["reference", "target"]:
        merged[f"{prefix}_n_samples"] = merged[f"n_samples_{prefix}"].fillna(0.0)
        merged[f"{prefix}_coverage"] = merged[f"coverage_{prefix}"].fillna(0.0)
        merged[f"{prefix}_abstention_rate"] = merged[f"abstention_rate_{prefix}"].fillna(1.0)
        merged[f"{prefix}_selective_accuracy"] = merged[f"selective_accuracy_{prefix}"].fillna(float("nan"))
        merged[f"{prefix}_support_ok"] = merged[f"support_ok_{prefix}"].fillna(False)
    merged["supported_both"] = merged["support_ok_reference"] & merged["support_ok_target"]
    merged["delta_abstention_rate"] = (
        merged["target_abstention_rate"] - merged["reference_abstention_rate"]
    )
    merged["delta_coverage"] = merged["target_coverage"] - merged["reference_coverage"]
    merged["delta_selective_accuracy"] = (
        merged["target_selective_accuracy"] - merged["reference_selective_accuracy"]
    )
    return merged  # type: ignore[no-any-return]


def _subgroup_abstention_disparities(summary: pd.DataFrame) -> pd.DataFrame:
    supported = summary.loc[summary["supported_both"]].copy()
    if supported.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for metric, higher_is_worse in [
        ("target_abstention_rate", True),
        ("target_coverage", False),
        ("target_selective_accuracy", False),
        ("delta_abstention_rate", True),
    ]:
        if higher_is_worse:
            worst = supported.sort_values(metric, ascending=False).iloc[0]
            best = supported.sort_values(metric, ascending=True).iloc[0]
            gap = float(worst[metric] - best[metric])
        else:
            worst = supported.sort_values(metric, ascending=True).iloc[0]
            best = supported.sort_values(metric, ascending=False).iloc[0]
            gap = float(best[metric] - worst[metric])

        rows.append(
            {
                "metric": metric,
                "worst_group": f"{worst['slice_name']} -> {worst['group']}",
                "best_group": f"{best['slice_name']} -> {best['group']}",
                "worst_value": float(worst[metric]),
                "best_value": float(best[metric]),
                "absolute_gap": gap,
            }
        )
    return pd.DataFrame.from_records(rows)  # type: ignore[no-any-return]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
