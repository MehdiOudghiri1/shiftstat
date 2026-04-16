"""Benchmarking platform for reproducible reliability-under-shift studies."""

from shiftstat.bench.baselines import default_baseline_registry
from shiftstat.bench.registry import (
    BaselineDefinition,
    BaselineRegistry,
    MetricDefinition,
    MetricRegistry,
    default_metric_registry,
)
from shiftstat.bench.results import BenchmarkResult
from shiftstat.bench.runner import BenchmarkRunner
from shiftstat.bench.scenarios import (
    BenchmarkCase,
    BenchmarkScenario,
    make_calibration_drift_scenario,
    make_covariate_shift_sweep_scenario,
    make_mixed_tabular_scenario,
    make_selective_shift_scenario,
    make_subgroup_degradation_scenario,
    scenario_from_config,
)

__all__ = [
    "BaselineDefinition",
    "BaselineRegistry",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkScenario",
    "MetricDefinition",
    "MetricRegistry",
    "default_baseline_registry",
    "default_metric_registry",
    "make_calibration_drift_scenario",
    "make_covariate_shift_sweep_scenario",
    "make_mixed_tabular_scenario",
    "make_selective_shift_scenario",
    "make_subgroup_degradation_scenario",
    "scenario_from_config",
]
