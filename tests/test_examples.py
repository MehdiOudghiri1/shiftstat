from __future__ import annotations

from examples.abstention_under_shift import run_example as run_abstention_example
from examples.aggregate_vs_subgroup_calibration import run_example as run_aggregate_vs_subgroup_example
from examples.benchmark_covariate_sweep import run_example as run_benchmark_covariate_sweep_example
from examples.deployment_audit_case_study import run_example as run_deployment_audit_example
from examples.discover_failure_slices import run_example as run_slice_discovery_example
from examples.experiment_configuration_workflow import run_example as run_experiment_configuration_example
from examples.generate_paper_ready_artifacts import run_example as run_paper_artifact_example
from examples.selective_deployment_case_study import run_example as run_selective_case_study_example
from examples.selective_prediction_benchmark import run_example as run_selective_benchmark_example
from examples.reliability_benchmark_with_subgroup_failures import (
    run_example as run_subgroup_benchmark_example,
)
from examples.subgroup_rejection_analysis import run_example as run_subgroup_rejection_example
from examples.weighted_threshold_tuning_under_shift import run_example as run_weighted_threshold_example
from examples.calibration_degradation_under_shift import run_example as run_calibration_example
from examples.detect_feature_drift import run_example as run_detection_example
from examples.full_workflow import run_example as run_workflow_example
from examples.hidden_subgroup_failures_under_shift import run_example as run_hidden_failure_example
from examples.importance_weighting_evaluation import run_example as run_weighting_example
from examples.recalibration_under_shift import run_example as run_recalibration_example
from examples.reliability_report_workflow import run_example as run_reliability_workflow_example
from examples.weighted_calibration_evaluation import run_example as run_weighted_calibration_example


def test_examples_run() -> None:
    abstention_result = run_abstention_example()
    detection_result = run_detection_example()
    weighting_result = run_weighting_example()
    workflow_result = run_workflow_example()
    calibration_result = run_calibration_example()
    weighted_calibration_result = run_weighted_calibration_example()
    recalibration_result = run_recalibration_example()
    reliability_result = run_reliability_workflow_example()
    hidden_failure_result = run_hidden_failure_example()
    slice_discovery_result = run_slice_discovery_example()
    aggregate_vs_subgroup_result = run_aggregate_vs_subgroup_example()
    deployment_audit_result = run_deployment_audit_example()
    weighted_threshold_result = run_weighted_threshold_example()
    subgroup_rejection_result = run_subgroup_rejection_example()
    selective_case_study_result = run_selective_case_study_example()
    benchmark_covariate_sweep_result = run_benchmark_covariate_sweep_example()
    subgroup_benchmark_result = run_subgroup_benchmark_example()
    selective_benchmark_result = run_selective_benchmark_example()
    experiment_configuration_result = run_experiment_configuration_example()
    paper_artifact_result = run_paper_artifact_example()

    assert "coverage" in abstention_result
    assert "summary" in detection_result
    assert "effective_sample_size" in weighting_result
    assert workflow_result["shift_detected"] is True
    assert "comparison" in calibration_result
    assert "comparison" in weighted_calibration_result
    assert "summary" in recalibration_result
    assert "report_markdown" in reliability_result
    assert "worst_slice" in hidden_failure_result
    assert slice_discovery_result["n_slices"] >= 1
    assert "hidden_failure_flags" in aggregate_vs_subgroup_result
    assert deployment_audit_result["discovered_slice_count"] >= 1
    assert weighted_threshold_result["threshold_comparison"] is not None
    assert "top_gap_metric" in subgroup_rejection_result
    assert "report_markdown" in selective_case_study_result
    assert benchmark_covariate_sweep_result["summary_rows"] >= 1
    assert subgroup_benchmark_result["worst_group_gap_present"] is True
    assert "target_risk_reduction" in selective_benchmark_result["publication_metrics"]
    assert experiment_configuration_result["manifest_exists"] is True
    assert paper_artifact_result["manifest_exists"] is True
