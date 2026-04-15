from __future__ import annotations

import numpy as np
import pandas as pd

from shiftstat.datasets import make_covariate_shift_classification, make_mixed_type_shift
from shiftstat.detect import ShiftDetector


def test_shift_detector_detects_mixed_type_shift() -> None:
    bundle = make_mixed_type_shift(random_state=0)
    detector = ShiftDetector(
        categorical_features=["cat_0", "cat_1"],
        random_state=0,
    )
    detector.fit(bundle.X_ref, bundle.X_target)

    summary = detector.summary()
    assert not summary.empty
    assert detector.dataset_summary_.n_shifted_features >= 1
    assert summary.iloc[0]["severity_score"] >= summary.iloc[-1]["severity_score"]
    assert detector.classifier_result_.auc >= 0.5


def test_shift_detector_supports_numpy_arrays() -> None:
    bundle = make_covariate_shift_classification(random_state=3)
    detector = ShiftDetector(random_state=3)
    detector.fit(bundle.X_ref.to_numpy(), bundle.X_target.to_numpy())
    summary = detector.summary()
    assert summary["feature_name"].str.startswith("feature_").all()


def test_multiple_testing_correction_is_applied() -> None:
    bundle = make_covariate_shift_classification(random_state=2)
    detector = ShiftDetector(multiple_testing="bonferroni", random_state=2)
    detector.fit(bundle.X_ref, bundle.X_target)
    summary = detector.summary()
    assert np.all(summary["adjusted_p_value"] >= summary["p_value"])


def test_shift_detector_is_deterministic_with_fixed_seed() -> None:
    bundle = make_covariate_shift_classification(random_state=4)
    detector_a = ShiftDetector(random_state=5)
    detector_b = ShiftDetector(random_state=5)
    detector_a.fit(bundle.X_ref, bundle.X_target)
    detector_b.fit(bundle.X_ref, bundle.X_target)
    assert detector_a.classifier_result_.auc == detector_b.classifier_result_.auc


def test_detection_report_exports() -> None:
    bundle = make_covariate_shift_classification(random_state=1)
    detector = ShiftDetector(random_state=1).fit(bundle.X_ref, bundle.X_target)
    report = detector.to_report()
    assert "Shift summary" in report.to_markdown()
    assert isinstance(report.to_frame(), pd.DataFrame)
    assert "dataset_summary" in report.to_dict()

