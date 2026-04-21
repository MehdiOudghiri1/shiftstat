from __future__ import annotations

from shiftstat.audit import ReliabilityAuditor, SliceDiscoverer
from shiftstat.datasets import make_hidden_subgroup_shift_classification


def test_slice_discovery_is_reproducible() -> None:
    data = make_hidden_subgroup_shift_classification(
        pattern="minority_subgroup_degradation",
        random_state=70,
    )
    summary_one = (
        SliceDiscoverer(random_state=70, min_samples_leaf=20)
        .fit(
            data.X_ref,
            data.y_ref,
            data.reference_predictions,
            data.X_target,
            data.y_target,
            data.target_predictions,
        )
        .summary()
    )
    summary_two = (
        SliceDiscoverer(random_state=70, min_samples_leaf=20)
        .fit(
            data.X_ref,
            data.y_ref,
            data.reference_predictions,
            data.X_target,
            data.y_target,
            data.target_predictions,
        )
        .summary()
    )

    assert summary_one["rule"].tolist() == summary_two["rule"].tolist()
    assert (
        summary_one["target_failure_share"].tolist() == summary_two["target_failure_share"].tolist()
    )


def test_reliability_auditor_report_structure() -> None:
    data = make_hidden_subgroup_shift_classification(
        pattern="operational_calibration_drift",
        random_state=71,
    )
    auditor = ReliabilityAuditor(min_group_size=20, random_state=71).fit(
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

    assert {"metric", "worst_group", "absolute_gap"}.issubset(auditor.disparity_summary().columns)
    assert {"slice_name", "metric", "delta_value"}.issubset(auditor.heatmap_table().columns)
    assert "Deployment reliability audit" in report.to_markdown()
    assert "hidden_failure_flags" in report.to_dict()
    assert len(report.discovered_slices) >= 1
