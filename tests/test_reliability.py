from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reliability import ReliabilityAnalyzer, evaluate_under_shift


def test_reliability_analyzer_builds_profiles_and_report() -> None:
    data = make_covariate_shift_classification(random_state=25, shift_strength=1.0)
    analyzer = ReliabilityAnalyzer(n_bins=6).fit(
        data.y_ref,
        data.reference_predictions,
        data.y_target,
        data.target_predictions,
    )
    summary = analyzer.summary()
    report = analyzer.to_report()
    assert "delta_ece" in summary.columns
    assert "Reliability under shift" in report.to_markdown()
    assert "degradation_summary" in report.to_dict()


def test_evaluate_under_shift_returns_recalibrated_profile() -> None:
    data = make_covariate_shift_classification(random_state=26, shift_strength=1.1)
    result = evaluate_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        apply_importance_weighting=True,
        recalibration="temperature",
        random_state=26,
    )
    assert result.recalibrated_target_profile is not None
    assert result.weighting_summary is not None
    assert "delta_ece" in result.summary_frame().columns
    assert "### Effect of recalibration" in result.to_report().to_markdown()
