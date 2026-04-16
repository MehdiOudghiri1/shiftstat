"""Selective prediction, abstention, and decision-aware reliability."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AbstentionPolicy",
    "RiskCoverageCurve",
    "SelectiveDeploymentReport",
    "SelectivePredictor",
    "SelectiveProfile",
    "SelectiveShiftEvaluationResult",
    "evaluate_selective_under_shift",
]


def __getattr__(name: str) -> Any:
    if name == "AbstentionPolicy":
        from .policy import AbstentionPolicy

        return AbstentionPolicy
    if name == "SelectivePredictor":
        from .predictor import SelectivePredictor

        return SelectivePredictor
    if name == "evaluate_selective_under_shift":
        from .workflow import evaluate_selective_under_shift

        return evaluate_selective_under_shift
    if name in {
        "RiskCoverageCurve",
        "SelectiveDeploymentReport",
        "SelectiveProfile",
        "SelectiveShiftEvaluationResult",
    }:
        from . import results

        return getattr(results, name)
    raise AttributeError(f"module 'shiftstat.selective' has no attribute {name!r}")
