"""ShiftStat public package API."""

from shiftstat._version import __version__
from shiftstat.calibration import (
    CalibrationEvaluator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
)
from shiftstat.detect import ShiftDetector
from shiftstat.reliability import ReliabilityAnalyzer, evaluate_under_shift
from shiftstat.reweight import (
    ImportanceWeighter,
    compute_effective_sample_size,
    weighted_mean,
    weighted_risk,
)

__all__ = [
    "CalibrationEvaluator",
    "ImportanceWeighter",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "ReliabilityAnalyzer",
    "ShiftDetector",
    "TemperatureScaler",
    "__version__",
    "compute_effective_sample_size",
    "evaluate_under_shift",
    "weighted_mean",
    "weighted_risk",
]
