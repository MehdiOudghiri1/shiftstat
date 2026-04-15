"""Full mini workflow: detect, reweight, and evaluate under shift."""

from __future__ import annotations

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import ImportanceWeighter, weighted_risk


def run_example(random_state: int = 19) -> dict[str, object]:
    """Run an end-to-end ShiftStat workflow."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.0)
    detector = ShiftDetector(random_state=random_state)
    detector.fit(data.X_ref, data.X_target)

    weighter = ImportanceWeighter(method="logistic", random_state=random_state)
    weights = weighter.fit_predict(data.X_ref, data.X_target)

    unweighted_log_loss = weighted_risk(
        data.y_ref,
        data.reference_predictions,
        loss="log_loss",
    )
    weighted_log_loss = weighted_risk(
        data.y_ref,
        data.reference_predictions,
        sample_weight=weights,
        loss="log_loss",
    )
    return {
        "top_features": detector.summary()["feature_name"].head(3).tolist(),
        "shift_detected": detector.dataset_summary_.overall_shift_detected,
        "classifier_auc": detector.classifier_result_.auc,
        "effective_sample_size": weighter.effective_sample_size_,
        "unweighted_log_loss": unweighted_log_loss,
        "weighted_log_loss": weighted_log_loss,
    }


if __name__ == "__main__":
    print(run_example())

