"""Classification calibration degradation under covariate shift."""

from __future__ import annotations

from shiftstat.calibration import CalibrationEvaluator, compare_calibration
from shiftstat.datasets import make_covariate_shift_classification


def run_example(random_state: int = 21) -> dict[str, object]:
    """Compare reference and target calibration on a shifted synthetic problem."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.0)
    evaluator = CalibrationEvaluator(n_bins=8)
    reference = evaluator.evaluate(data.y_ref, data.reference_predictions, name="reference")
    target = evaluator.evaluate(data.y_target, data.target_predictions, name="target")
    comparison = compare_calibration({"reference": reference, "target": target})
    return {
        "reference": reference.to_dict(),
        "target": target.to_dict(),
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = run_example()
    print(result["comparison"])

