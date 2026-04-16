from __future__ import annotations

import math
from pathlib import Path

from shiftstat.bench import (
    BenchmarkRunner,
    make_covariate_shift_sweep_scenario,
    make_selective_shift_scenario,
)


def test_benchmark_runner_aggregation_matches_seed_average() -> None:
    scenario = make_covariate_shift_sweep_scenario(
        name="aggregation_test",
        severities=[0.3],
        seeds=[1, 2],
        n_samples_ref=140,
        n_samples_target=140,
        baseline_names=["raw_model"],
    )

    result = BenchmarkRunner().run(scenario)
    run_frame = result.run_frame()
    aggregate_frame = result.aggregate_frame()

    assert len(run_frame) == 2
    assert len(aggregate_frame) == 1
    expected = float(run_frame["delta_accuracy"].mean())
    observed = float(aggregate_frame["delta_accuracy"].iloc[0])
    assert math.isclose(observed, expected)
    assert aggregate_frame["n_runs"].iloc[0] == 2


def test_benchmark_result_exports_publication_artifacts(tmp_path: Path) -> None:
    scenario = make_selective_shift_scenario(
        name="artifact_test",
        severities=[0.4],
        seeds=[3],
        n_samples_ref=140,
        n_samples_target=140,
        baseline_names=["confidence_abstention"],
    )

    result = BenchmarkRunner().run(scenario)
    paths = result.export_artifacts(tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["runs_csv"].exists()
    assert paths["summary_csv"].exists()
    assert paths["manifest"].exists()
    assert paths["figures"]["target_risk_reduction"].exists()
    assert paths["latex_tables"]["target_risk_reduction"].exists()


def test_benchmark_schema_stability() -> None:
    scenario = make_covariate_shift_sweep_scenario(
        name="schema_test",
        severities=[0.2],
        seeds=[1],
        n_samples_ref=120,
        n_samples_target=120,
        baseline_names=["raw_model"],
    )

    result = BenchmarkRunner().run(scenario)
    payload = result.to_dict()

    assert set(payload) == {
        "aggregated_records",
        "baseline_names",
        "created_at_utc",
        "description",
        "family",
        "metric_schema",
        "publication_metrics",
        "run_records",
        "scenario_config",
        "scenario_name",
        "seeds",
        "x_axis_label",
    }
