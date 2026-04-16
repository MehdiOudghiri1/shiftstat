"""Calibration assessment and recalibration under distribution shift."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CalibrationEvaluator",
    "CalibrationResult",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureScaler",
    "compare_calibration",
]


def __getattr__(name: str) -> Any:
    if name in {"CalibrationEvaluator", "compare_calibration"}:
        from shiftstat.calibration import evaluator

        return getattr(evaluator, name)
    if name in {"IsotonicCalibrator", "PlattCalibrator", "TemperatureScaler"}:
        from shiftstat.calibration import calibrators

        return getattr(calibrators, name)
    if name == "CalibrationResult":
        from shiftstat.calibration.results import CalibrationResult

        return CalibrationResult
    raise AttributeError(f"module 'shiftstat.calibration' has no attribute {name!r}")
