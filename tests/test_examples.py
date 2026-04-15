from __future__ import annotations

from examples.detect_feature_drift import run_example as run_detection_example
from examples.full_workflow import run_example as run_workflow_example
from examples.importance_weighting_evaluation import run_example as run_weighting_example


def test_examples_run() -> None:
    detection_result = run_detection_example()
    weighting_result = run_weighting_example()
    workflow_result = run_workflow_example()

    assert "summary" in detection_result
    assert "effective_sample_size" in weighting_result
    assert workflow_result["shift_detected"] is True

