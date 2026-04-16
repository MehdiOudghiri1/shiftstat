"""Benchmark sweep over covariate shift severity."""

from __future__ import annotations

from shiftstat.bench import BenchmarkRunner, make_covariate_shift_sweep_scenario


def run_example() -> dict[str, object]:
    """Run a compact V5 benchmark sweep example."""

    scenario = make_covariate_shift_sweep_scenario(
        name="example_covariate_sweep",
        severities=[0.3, 0.9],
        seeds=[7],
        n_samples_ref=180,
        n_samples_target=180,
        baseline_names=["raw_model", "weighting_only", "confidence_abstention"],
    )
    result = BenchmarkRunner().run(scenario)
    return {
        "available_metrics": result.available_metrics(),
        "summary_rows": len(result.aggregate_frame()),
        "report_markdown": result.to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
