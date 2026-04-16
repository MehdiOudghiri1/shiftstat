from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import ImportanceWeighter


def test_detection_plot_smoke() -> None:
    bundle = make_covariate_shift_classification(random_state=9)
    detector = ShiftDetector(random_state=9).fit(bundle.X_ref, bundle.X_target)
    assert detector.plot("feature_drift") is not None
    assert detector.plot("severity_heatmap") is not None
    assert detector.plot("roc") is not None


def test_reweight_plot_smoke() -> None:
    bundle = make_covariate_shift_classification(random_state=10)
    weighter = ImportanceWeighter(random_state=10).fit(bundle.X_ref, bundle.X_target)
    assert weighter.plot("histogram") is not None
    assert weighter.plot("effective_sample_size") is not None
