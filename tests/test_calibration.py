from __future__ import annotations

import numpy as np

from shiftstat.calibration import (
    CalibrationEvaluator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    compare_calibration,
)
from shiftstat.metrics import expected_calibration_error, weighted_ece


def test_calibration_metrics_detect_miscalibration() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.45, 0.45, 0.45, 0.55, 0.55, 0.55])
    assert expected_calibration_error(y_true, y_prob, n_bins=6) > 0.1


def test_weighted_ece_differs_from_unweighted() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    weights = np.array([10.0, 1.0, 1.0, 1.0])
    unweighted = expected_calibration_error(y_true, y_prob, n_bins=2)
    weighted = weighted_ece(y_true, y_prob, n_bins=2, sample_weight=weights)
    assert weighted != unweighted


def test_calibration_evaluator_and_compare() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.75, 0.25, 0.7, 0.3])
    evaluator = CalibrationEvaluator(n_bins=4)
    result = evaluator.evaluate(y_true, y_prob, name="baseline")
    comparison = compare_calibration({"baseline": result})
    assert result.to_frame().shape[1] >= 5
    assert comparison.iloc[0]["name"] == "baseline"


def test_recalibrators_produce_valid_probabilities() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.5])
    for calibrator in [TemperatureScaler(), IsotonicCalibrator(), PlattCalibrator()]:
        calibrated = calibrator.fit(y_prob, y_true).predict_proba(y_prob)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
