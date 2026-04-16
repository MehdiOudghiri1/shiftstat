"""Shift detection for tabular data."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ClassifierShiftResult",
    "DatasetShiftSummary",
    "FeatureShiftResult",
    "ShiftDetector",
]


def __getattr__(name: str) -> Any:
    if name == "ShiftDetector":
        from shiftstat.detect.detector import ShiftDetector

        return ShiftDetector
    if name in {"ClassifierShiftResult", "DatasetShiftSummary", "FeatureShiftResult"}:
        from shiftstat.detect import results

        return getattr(results, name)
    raise AttributeError(f"module 'shiftstat.detect' has no attribute {name!r}")
