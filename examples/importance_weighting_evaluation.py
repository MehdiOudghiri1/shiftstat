"""Compare weighted and unweighted evaluation under covariate shift."""

from __future__ import annotations

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reweight import ImportanceWeighter, weighted_risk


def run_example(random_state: int = 11) -> dict[str, float]:
    """Run a compact importance-weighting example."""

    data = make_covariate_shift_classification(random_state=random_state)
    weighter = ImportanceWeighter(method="domain_classifier", random_state=random_state)
    weights = weighter.fit_predict(data.X_ref, data.X_target)

    unweighted = weighted_risk(data.y_ref, data.reference_predictions, loss="log_loss")
    weighted = weighted_risk(
        data.y_ref,
        data.reference_predictions,
        sample_weight=weights,
        loss="log_loss",
    )
    return {
        "unweighted_log_loss": float(unweighted),
        "weighted_log_loss": float(weighted),
        "effective_sample_size": float(weighter.effective_sample_size_),
    }


if __name__ == "__main__":
    print(run_example())

