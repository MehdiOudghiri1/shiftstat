from __future__ import annotations

import numpy as np
import pandas as pd

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reweight import ImportanceWeighter, weighted_mean, weighted_risk


def test_importance_weighter_produces_positive_normalized_weights() -> None:
    bundle = make_covariate_shift_classification(random_state=6)
    weighter = ImportanceWeighter(method="logistic", random_state=6)
    weights = weighter.fit_predict(bundle.X_ref, bundle.X_target)
    assert np.all(weights > 0)
    assert np.isclose(weights.mean(), 1.0, atol=1e-6)
    assert weighter.effective_sample_size_ <= len(weights)


def test_importance_weighter_supports_dataframe_and_report() -> None:
    bundle = make_covariate_shift_classification(random_state=8)
    weighter = ImportanceWeighter(method="domain_classifier", random_state=8).fit(
        bundle.X_ref,
        bundle.X_target,
    )
    report = weighter.to_report()
    assert "Importance weighting summary" in report.to_markdown()
    assert isinstance(report.to_frame(), pd.DataFrame)
    assert report.to_dict()["summary"]["method"] == "domain_classifier"


def test_weighted_mean_and_risk() -> None:
    values = np.array([1.0, 3.0, 5.0])
    weights = np.array([1.0, 1.0, 2.0])
    assert weighted_mean(values, weights) == 3.5

    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 0.0, 2.0])
    risk = weighted_risk(y_true, y_pred, sample_weight=weights, loss="mae")
    assert np.isclose(risk, 0.25)

