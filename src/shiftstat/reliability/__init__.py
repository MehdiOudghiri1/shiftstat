"""Reliability diagnostics and evaluation workflows under shift."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ReliabilityAnalyzer",
    "ReliabilityDegradationSummary",
    "ReliabilityProfile",
    "ReliabilityShiftReport",
    "ShiftEvaluationResult",
    "evaluate_under_shift",
]


def __getattr__(name: str) -> Any:
    if name == "ReliabilityAnalyzer":
        from shiftstat.reliability.analyzer import ReliabilityAnalyzer

        return ReliabilityAnalyzer
    if name == "evaluate_under_shift":
        from shiftstat.reliability.workflow import evaluate_under_shift

        return evaluate_under_shift
    if name in {
        "ReliabilityDegradationSummary",
        "ReliabilityProfile",
        "ReliabilityShiftReport",
        "ShiftEvaluationResult",
    }:
        from shiftstat.reliability import results

        return getattr(results, name)
    raise AttributeError(f"module 'shiftstat.reliability' has no attribute {name!r}")
