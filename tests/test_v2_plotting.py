from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression

from shiftstat.calibration import CalibrationEvaluator
from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.plotting import (
    plot_calibration_comparison,
    plot_confidence_error_curve,
    plot_confidence_histogram,
    plot_recalibration_comparison,
    plot_reliability_diagram,
)
from shiftstat.reliability import evaluate_under_shift


def test_v2_plotting_smoke() -> None:
    data = make_covariate_shift_classification(random_state=27, shift_strength=1.0)
    evaluator = CalibrationEvaluator(n_bins=6)
    reference = evaluator.evaluate(data.y_ref, data.reference_predictions, name="reference")
    target = evaluator.evaluate(data.y_target, data.target_predictions, name="target")
    result = evaluate_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        recalibration="temperature",
        random_state=27,
    )

    assert plot_reliability_diagram(reference) is not None
    assert plot_calibration_comparison(reference, target) is not None
    assert plot_confidence_histogram(data.target_predictions) is not None
    assert plot_confidence_error_curve(result.target_profile) is not None
    assert (
        plot_recalibration_comparison(
            result.target_profile,
            result.recalibrated_target_profile,
        )
        is not None
    )
