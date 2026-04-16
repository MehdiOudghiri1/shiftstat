"""Weighted versus unweighted calibration under covariate shift."""

from __future__ import annotations

from shiftstat.calibration import CalibrationEvaluator, compare_calibration
from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reweight import ImportanceWeighter


def run_example(random_state: int = 22) -> dict[str, object]:
    """Compare unweighted and weighted reference calibration diagnostics."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.0)
    weighter = ImportanceWeighter(method="domain_classifier", random_state=random_state)
    weights = weighter.fit_predict(data.X_ref, data.X_target)

    evaluator = CalibrationEvaluator(n_bins=8)
    unweighted = evaluator.evaluate(data.y_ref, data.reference_predictions, name="reference")
    weighted = evaluator.evaluate(
        data.y_ref,
        data.reference_predictions,
        sample_weight=weights,
        name="reference_weighted",
    )
    comparison = compare_calibration(
        {"reference": unweighted, "reference_weighted": weighted}
    )
    return {
        "comparison": comparison,
        "weight_summary": weighter.summary(),
        "unweighted": unweighted.to_dict(),
        "weighted": weighted.to_dict(),
    }


if __name__ == "__main__":
    result = run_example()
    print(result["comparison"])

