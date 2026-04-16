"""Example: aggregate calibration can hide subgroup calibration drift."""

from __future__ import annotations

from shiftstat.audit import ReliabilityAuditor
from shiftstat.datasets import make_hidden_subgroup_shift_classification


def run_example(random_state: int = 43) -> dict[str, object]:
    """Compare aggregate and worst-group calibration conclusions."""

    data = make_hidden_subgroup_shift_classification(
        pattern="operational_calibration_drift",
        random_state=random_state,
    )
    auditor = ReliabilityAuditor(min_group_size=25, random_state=random_state).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
    )
    comparison = auditor.aggregate_vs_subgroup_summary()
    ece_row = comparison.loc[comparison["metric"] == "ece"].iloc[0]
    return {
        "aggregate_ece": float(auditor.to_report().aggregate_summary["target_ece"]),
        "worst_group_ece": float(ece_row["worst_group_target_value"]),
        "ece_gap": float(ece_row["absolute_gap"]),
        "hidden_failure_flags": auditor.to_report().hidden_failure_flags,
    }


if __name__ == "__main__":
    print(run_example())
