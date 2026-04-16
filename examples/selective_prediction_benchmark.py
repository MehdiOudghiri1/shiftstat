"""Selective prediction benchmark under shift."""

from __future__ import annotations

from shiftstat.bench import BenchmarkRunner, make_selective_shift_scenario


def run_example() -> dict[str, object]:
    """Run a compact selective-prediction benchmark example."""

    scenario = make_selective_shift_scenario(
        name="example_selective_benchmark",
        severities=[0.4, 1.0],
        seeds=[5],
        n_samples_ref=180,
        n_samples_target=180,
        baseline_names=[
            "confidence_abstention",
            "weighted_confidence_abstention",
            "recalibrated_confidence_abstention",
        ],
    )
    result = BenchmarkRunner().run(scenario)
    return {
        "publication_metrics": result.publication_metrics,
        "summary_rows": len(result.aggregate_frame()),
        "report_markdown": result.to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
