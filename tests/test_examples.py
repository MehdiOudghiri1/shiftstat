from __future__ import annotations

from examples.calibration_degradation_under_shift import run_example as run_calibration_example
from examples.detect_feature_drift import run_example as run_detection_example
from examples.full_workflow import run_example as run_workflow_example
from examples.importance_weighting_evaluation import run_example as run_weighting_example
from examples.recalibration_under_shift import run_example as run_recalibration_example
from examples.reliability_report_workflow import run_example as run_reliability_workflow_example
from examples.weighted_calibration_evaluation import run_example as run_weighted_calibration_example


def test_examples_run() -> None:
    detection_result = run_detection_example()
    weighting_result = run_weighting_example()
    workflow_result = run_workflow_example()
    calibration_result = run_calibration_example()
    weighted_calibration_result = run_weighted_calibration_example()
    recalibration_result = run_recalibration_example()
    reliability_result = run_reliability_workflow_example()

    assert "summary" in detection_result
    assert "effective_sample_size" in weighting_result
    assert workflow_result["shift_detected"] is True
    assert "comparison" in calibration_result
    assert "comparison" in weighted_calibration_result
    assert "summary" in recalibration_result
    assert "report_markdown" in reliability_result
