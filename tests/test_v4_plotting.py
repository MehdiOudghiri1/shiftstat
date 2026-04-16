from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.plotting import (
    plot_abstention_distribution,
    plot_confidence_accept_reject_distribution,
    plot_coverage_vs_threshold,
    plot_risk_coverage_curve,
    plot_selective_reliability_diagram,
    plot_subgroup_abstention_comparison,
)
from shiftstat.selective import evaluate_selective_under_shift


def test_v4_selective_plotting_smoke() -> None:
    data = make_covariate_shift_classification(random_state=122, shift_strength=1.0)
    result = evaluate_selective_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        policy_method="confidence",
        target_coverage=0.8,
        subgroup_features=["x0", "x1"],
        random_state=122,
    )
    report = result.to_report()

    assert plot_risk_coverage_curve(report) is not None
    assert plot_coverage_vs_threshold(report) is not None
    assert plot_abstention_distribution(result.target_selective_profile) is not None
    assert (
        plot_confidence_accept_reject_distribution(result.target_selective_profile) is not None
    )
    assert plot_subgroup_abstention_comparison(report) is not None
    assert plot_selective_reliability_diagram(result.target_selective_profile) is not None
