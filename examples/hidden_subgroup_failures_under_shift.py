"""Example: hidden subgroup failures under deployment shift."""

from __future__ import annotations

from shiftstat.datasets import make_hidden_subgroup_shift_classification
from shiftstat.subgroup import SubgroupAnalyzer


def run_example(random_state: int = 41) -> dict[str, object]:
    """Detect concentrated subgroup degradation beneath aggregate stability."""

    data = make_hidden_subgroup_shift_classification(
        pattern="masked_subgroup_shift",
        random_state=random_state,
    )
    analyzer = SubgroupAnalyzer(min_group_size=25).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
    )
    top_row = analyzer.degradation_ranking().iloc[0]
    return {
        "worst_slice": f"{top_row['slice_name']} -> {top_row['group']}",
        "severity_score": float(top_row["severity_score"]),
        "delta_error_rate": float(top_row["delta_error_rate"]),
        "report_markdown": analyzer.to_report().to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
