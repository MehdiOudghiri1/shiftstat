"""Reliability benchmark focused on subgroup failures."""

from __future__ import annotations

from shiftstat.bench import BenchmarkRunner, make_subgroup_degradation_scenario


def run_example() -> dict[str, object]:
    """Run a compact subgroup-failure benchmark example."""

    scenario = make_subgroup_degradation_scenario(
        name="example_subgroup_benchmark",
        patterns=["masked_subgroup_shift"],
        seeds=[11],
        n_samples_ref=220,
        n_samples_target=220,
        baseline_names=["raw_model", "subgroup_audit", "weighted_confidence_abstention"],
    )
    result = BenchmarkRunner().run(scenario)
    summary = result.aggregate_frame()
    return {
        "summary_rows": len(summary),
        "worst_group_gap_present": "worst_group_accuracy_gap" in summary.columns,
        "report_markdown": result.to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
