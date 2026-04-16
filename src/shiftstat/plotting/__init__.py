"""Publication-friendly plotting utilities."""

from shiftstat.plotting.calibration import (
    plot_calibration_comparison,
    plot_confidence_histogram,
    plot_recalibration_comparison,
    plot_reliability_diagram,
    plot_weighted_unweighted_calibration,
)
from shiftstat.plotting.detect import (
    plot_feature_drift,
    plot_shift_severity_heatmap,
    plot_source_discrimination_roc,
)
from shiftstat.plotting.reliability import plot_confidence_error_curve
from shiftstat.plotting.reweight import plot_effective_sample_size, plot_importance_weight_histogram

__all__ = [
    "plot_calibration_comparison",
    "plot_confidence_error_curve",
    "plot_confidence_histogram",
    "plot_effective_sample_size",
    "plot_feature_drift",
    "plot_importance_weight_histogram",
    "plot_recalibration_comparison",
    "plot_reliability_diagram",
    "plot_shift_severity_heatmap",
    "plot_source_discrimination_roc",
    "plot_weighted_unweighted_calibration",
]
