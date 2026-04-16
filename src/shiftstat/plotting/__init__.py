"""Publication-friendly plotting utilities."""

from .benchmark import plot_benchmark_metric_sweep
from .audit import (
    plot_aggregate_vs_subgroup,
    plot_discovered_slice_summary,
    plot_failure_concentration,
    plot_subgroup_degradation,
    plot_subgroup_metric_heatmap,
    plot_worst_group_comparison,
)
from .calibration import (
    plot_calibration_comparison,
    plot_confidence_histogram,
    plot_recalibration_comparison,
    plot_reliability_diagram,
    plot_weighted_unweighted_calibration,
)
from .detect import (
    plot_feature_drift,
    plot_shift_severity_heatmap,
    plot_source_discrimination_roc,
)
from .reliability import plot_confidence_error_curve
from .reweight import plot_effective_sample_size, plot_importance_weight_histogram
from .selective import (
    plot_abstention_distribution,
    plot_confidence_accept_reject_distribution,
    plot_coverage_vs_threshold,
    plot_risk_coverage_curve,
    plot_selective_reliability_diagram,
    plot_subgroup_abstention_comparison,
)

__all__ = [
    "plot_aggregate_vs_subgroup",
    "plot_abstention_distribution",
    "plot_benchmark_metric_sweep",
    "plot_calibration_comparison",
    "plot_confidence_accept_reject_distribution",
    "plot_confidence_error_curve",
    "plot_confidence_histogram",
    "plot_coverage_vs_threshold",
    "plot_discovered_slice_summary",
    "plot_effective_sample_size",
    "plot_failure_concentration",
    "plot_feature_drift",
    "plot_importance_weight_histogram",
    "plot_recalibration_comparison",
    "plot_reliability_diagram",
    "plot_risk_coverage_curve",
    "plot_selective_reliability_diagram",
    "plot_shift_severity_heatmap",
    "plot_source_discrimination_roc",
    "plot_subgroup_abstention_comparison",
    "plot_subgroup_degradation",
    "plot_subgroup_metric_heatmap",
    "plot_weighted_unweighted_calibration",
    "plot_worst_group_comparison",
]
