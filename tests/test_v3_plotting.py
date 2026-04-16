from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from shiftstat.audit import ReliabilityAuditor
from shiftstat.datasets import make_hidden_subgroup_shift_classification
from shiftstat.plotting import (
    plot_aggregate_vs_subgroup,
    plot_discovered_slice_summary,
    plot_failure_concentration,
    plot_subgroup_degradation,
    plot_subgroup_metric_heatmap,
    plot_worst_group_comparison,
)


def test_v3_plotting_smoke() -> None:
    data = make_hidden_subgroup_shift_classification(random_state=72)
    auditor = ReliabilityAuditor(min_group_size=20, random_state=72).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
    )
    report = auditor.to_report()

    assert plot_subgroup_degradation(auditor.subgroup_analyzer_) is not None
    assert plot_worst_group_comparison(report) is not None
    assert plot_subgroup_metric_heatmap(report) is not None
    assert plot_discovered_slice_summary(report) is not None
    assert plot_failure_concentration(report) is not None
    assert plot_aggregate_vs_subgroup(report) is not None
