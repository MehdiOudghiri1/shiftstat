"""ShiftStat public package API."""

from __future__ import annotations

from typing import Any

from ._version import __version__

__all__ = [
    "AbstentionPolicy",
    "BaselineRegistry",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkScenario",
    "CalibrationEvaluator",
    "ConditionalReliabilityAuditor",
    "ImportanceWeighter",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "ReliabilityAuditor",
    "ReliabilityAnalyzer",
    "RiskCoverageCurve",
    "SliceDiscoverer",
    "SubgroupAnalyzer",
    "SelectivePredictor",
    "ShiftDetector",
    "TemperatureScaler",
    "__version__",
    "compute_effective_sample_size",
    "discover_failure_slices",
    "evaluate_under_shift",
    "evaluate_selective_under_shift",
    "group_by_feature",
    "group_metrics",
    "run_experiment",
    "weighted_mean",
    "weighted_risk",
]


def __getattr__(name: str) -> Any:
    if name in {"ConditionalReliabilityAuditor", "ReliabilityAuditor"}:
        from .audit import ConditionalReliabilityAuditor, ReliabilityAuditor

        if name == "ConditionalReliabilityAuditor":
            return ConditionalReliabilityAuditor
        return ReliabilityAuditor
    if name in {"SliceDiscoverer", "discover_failure_slices"}:
        from .audit import SliceDiscoverer, discover_failure_slices

        if name == "SliceDiscoverer":
            return SliceDiscoverer
        return discover_failure_slices
    if name in {"SubgroupAnalyzer", "group_by_feature", "group_metrics"}:
        from .subgroup import SubgroupAnalyzer, group_by_feature, group_metrics

        if name == "SubgroupAnalyzer":
            return SubgroupAnalyzer
        if name == "group_by_feature":
            return group_by_feature
        return group_metrics
    if name in {
        "AbstentionPolicy",
        "SelectivePredictor",
        "RiskCoverageCurve",
        "evaluate_selective_under_shift",
    }:
        from .selective import (
            AbstentionPolicy,
            RiskCoverageCurve,
            SelectivePredictor,
            evaluate_selective_under_shift,
        )

        mapping = {
            "AbstentionPolicy": AbstentionPolicy,
            "SelectivePredictor": SelectivePredictor,
            "RiskCoverageCurve": RiskCoverageCurve,
            "evaluate_selective_under_shift": evaluate_selective_under_shift,
        }
        return mapping[name]
    if name in {
        "BaselineRegistry",
        "BenchmarkResult",
        "BenchmarkRunner",
        "BenchmarkScenario",
    }:
        from .bench import (
            BaselineRegistry,
            BenchmarkResult,
            BenchmarkRunner,
            BenchmarkScenario,
        )

        mapping = {
            "BaselineRegistry": BaselineRegistry,
            "BenchmarkResult": BenchmarkResult,
            "BenchmarkRunner": BenchmarkRunner,
            "BenchmarkScenario": BenchmarkScenario,
        }
        return mapping[name]
    if name in {"CalibrationEvaluator", "IsotonicCalibrator", "PlattCalibrator", "TemperatureScaler"}:
        from .calibration import (
            CalibrationEvaluator,
            IsotonicCalibrator,
            PlattCalibrator,
            TemperatureScaler,
        )

        mapping = {
            "CalibrationEvaluator": CalibrationEvaluator,
            "IsotonicCalibrator": IsotonicCalibrator,
            "PlattCalibrator": PlattCalibrator,
            "TemperatureScaler": TemperatureScaler,
        }
        return mapping[name]
    if name in {"ShiftDetector"}:
        from .detect import ShiftDetector

        return ShiftDetector
    if name in {"ReliabilityAnalyzer", "evaluate_under_shift"}:
        from .reliability import ReliabilityAnalyzer, evaluate_under_shift

        if name == "ReliabilityAnalyzer":
            return ReliabilityAnalyzer
        return evaluate_under_shift
    if name in {
        "ImportanceWeighter",
        "compute_effective_sample_size",
        "weighted_mean",
        "weighted_risk",
    }:
        from .reweight import (
            ImportanceWeighter,
            compute_effective_sample_size,
            weighted_mean,
            weighted_risk,
        )

        mapping = {
            "ImportanceWeighter": ImportanceWeighter,
            "compute_effective_sample_size": compute_effective_sample_size,
            "weighted_mean": weighted_mean,
            "weighted_risk": weighted_risk,
        }
        return mapping[name]
    if name in {"run_experiment"}:
        from .experiments import run_experiment

        return run_experiment
    raise AttributeError(f"module 'shiftstat' has no attribute {name!r}")
