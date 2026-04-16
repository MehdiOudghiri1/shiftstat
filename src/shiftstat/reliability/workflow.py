"""Model-facing evaluation workflows under distribution shift."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from shiftstat.calibration.calibrators import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
)
from shiftstat.calibration.evaluator import compare_calibration
from shiftstat.detect import ShiftDetector
from shiftstat.reliability.analyzer import ReliabilityAnalyzer
from shiftstat.reliability.results import ShiftEvaluationResult
from shiftstat.reweight import ImportanceWeighter
from shiftstat.utils.probabilities import extract_positive_class_probabilities
from shiftstat.utils.random import random_state_to_int


def _resolve_binary_probabilities(estimator: Any, X: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(
            extract_positive_class_probabilities(estimator.predict_proba(X)),
            dtype=float,
        )
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return np.asarray(1.0 / (1.0 + np.exp(-scores)), dtype=float)
    raise ValueError(
        "Estimator must expose predict_proba or decision_function for reliability analysis."
    )


def _make_calibrator(method: str) -> Any:
    if method == "temperature":
        return TemperatureScaler()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method in {"platt", "logistic"}:
        return PlattCalibrator()
    raise ValueError(f"Unsupported recalibration method: {method}.")


def evaluate_under_shift(
    estimator: Any,
    X_ref: Any,
    y_ref: np.ndarray,
    X_target: Any,
    y_target: np.ndarray,
    *,
    estimator_is_fitted: bool = False,
    apply_importance_weighting: bool = False,
    weighting_method: str = "domain_classifier",
    recalibration: str | None = None,
    calibration_fraction: float = 0.25,
    categorical_features: list[str] | list[int] | None = None,
    n_bins: int = 10,
    random_state: int | np.random.RandomState | None = None,
) -> ShiftEvaluationResult:
    """Evaluate a classifier on reference and target distributions under shift."""

    seed = random_state_to_int(random_state)
    if estimator_is_fitted:
        fitted_estimator = estimator
        X_eval_ref = X_ref
        y_eval_ref = np.asarray(y_ref)
    elif recalibration is None:
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(X_ref, y_ref)
        X_eval_ref = X_ref
        y_eval_ref = np.asarray(y_ref)
    else:
        X_train_ref, X_eval_ref, y_train_ref, y_eval_ref = train_test_split(
            X_ref,
            y_ref,
            test_size=calibration_fraction,
            stratify=y_ref,
            random_state=seed,
        )
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(X_train_ref, y_train_ref)

    p_ref = _resolve_binary_probabilities(fitted_estimator, X_eval_ref)
    p_target = _resolve_binary_probabilities(fitted_estimator, X_target)

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

    analyzer = ReliabilityAnalyzer(n_bins=n_bins)
    analyzer.fit(
        y_eval_ref,
        p_ref,
        y_target,
        p_target,
        reference_weight=reference_weight,
    )

    recalibrated_target_profile = None
    recalibration_summary = None
    calibration_comparison = analyzer.calibration_comparison_
    if recalibration is not None:
        calibrator = _make_calibrator(recalibration)
        calibrator.fit(p_ref, y_eval_ref, sample_weight=reference_weight)
        calibrated_target_probabilities = calibrator.predict_proba(p_target)
        recalibrated_target_profile = analyzer._build_profile(
            y_target,
            calibrated_target_probabilities,
            sample_weight=None,
            name="target_recalibrated",
        )
        calibration_comparison = compare_calibration(
            {
                "reference": analyzer._profile_calibration_result(analyzer.reference_profile_),
                "target": analyzer._profile_calibration_result(analyzer.target_profile_),
                "target_recalibrated": analyzer._profile_calibration_result(
                    recalibrated_target_profile
                ),
            }
        )
        recalibration_summary = {
            "method": recalibration,
            "fit_sample_size": int(len(y_eval_ref)),
        }

    return ShiftEvaluationResult(
        estimator_name=fitted_estimator.__class__.__name__,
        reference_profile=analyzer.reference_profile_,
        target_profile=analyzer.target_profile_,
        degradation_summary=analyzer.degradation_summary_,
        dataset_shift_overview=detector.dataset_summary_.to_dict(),
        weighted_reference_profile=analyzer.weighted_reference_profile_,
        recalibrated_target_profile=recalibrated_target_profile,
        weighting_summary=weighting_summary,
        recalibration_summary=recalibration_summary,
        calibration_comparison=_frame_records(calibration_comparison),
    )


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame-like comparison object to typed records."""

    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
