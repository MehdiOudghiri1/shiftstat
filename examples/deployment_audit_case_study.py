"""Example: end-to-end deployment reliability audit."""

from __future__ import annotations

from shiftstat.audit import ReliabilityAuditor
from shiftstat.datasets import make_hidden_subgroup_shift_classification


def run_example(random_state: int = 44) -> dict[str, object]:
    """Run a full V3 deployment audit workflow."""

    data = make_hidden_subgroup_shift_classification(
        pattern="masked_subgroup_shift",
        random_state=random_state,
    )
    auditor = ReliabilityAuditor(min_group_size=25, random_state=random_state).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
        subgroup_features=["region", "channel", "score", "load"],
        intersectional_features=[("region", "channel")],
    )
    report = auditor.to_report()
    return {
        "report_markdown": report.to_markdown(),
        "hidden_failure_flags": report.hidden_failure_flags,
        "discovered_slice_count": int(len(report.discovered_slices)),
        "top_operational_implication": (
            report.operational_implications[0] if report.operational_implications else ""
        ),
    }


if __name__ == "__main__":
    print(run_example())
