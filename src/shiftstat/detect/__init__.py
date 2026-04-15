"""Shift detection for tabular data."""

from shiftstat.detect.detector import ShiftDetector
from shiftstat.detect.results import ClassifierShiftResult, DatasetShiftSummary, FeatureShiftResult

__all__ = [
    "ClassifierShiftResult",
    "DatasetShiftSummary",
    "FeatureShiftResult",
    "ShiftDetector",
]

