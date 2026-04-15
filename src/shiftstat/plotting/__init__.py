"""Publication-friendly plotting utilities."""

from shiftstat.plotting.detect import (
    plot_feature_drift,
    plot_shift_severity_heatmap,
    plot_source_discrimination_roc,
)
from shiftstat.plotting.reweight import plot_effective_sample_size, plot_importance_weight_histogram

__all__ = [
    "plot_effective_sample_size",
    "plot_feature_drift",
    "plot_importance_weight_histogram",
    "plot_shift_severity_heatmap",
    "plot_source_discrimination_roc",
]

