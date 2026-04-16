from __future__ import annotations

import pytest

from shiftstat.datasets import make_hidden_subgroup_shift_classification
from shiftstat.exceptions import SmallSampleWarning
from shiftstat.subgroup import SubgroupAnalyzer, group_by_feature, group_metrics


def test_group_by_feature_supports_numeric_and_categorical_slices() -> None:
    data = make_hidden_subgroup_shift_classification(random_state=60)

    numeric_groups = group_by_feature(data.X_target, "load", reference=data.X_ref, n_bins=3)
    categorical_groups = group_by_feature(data.X_target, "region", reference=data.X_ref)

    assert numeric_groups.str.contains("load", na=False).all()
    assert categorical_groups.str.startswith("region=", na=False).all()


def test_group_metrics_warns_on_small_groups() -> None:
    with pytest.warns(SmallSampleWarning):
        frame = group_metrics(
            y_true=[0, 1, 0, 1],
            y_prob=[0.2, 0.8, 0.7, 0.4],
            groups=["a", "a", "b", "b"],
            min_group_size=3,
            min_class_count=2,
        )

    assert {"group", "accuracy", "ece", "sample_size_flag"}.issubset(frame.columns)


def test_subgroup_analyzer_builds_ranked_report() -> None:
    data = make_hidden_subgroup_shift_classification(random_state=61)
    analyzer = SubgroupAnalyzer(min_group_size=20).fit(
        data.X_ref,
        data.y_ref,
        data.reference_predictions,
        data.X_target,
        data.y_target,
        data.target_predictions,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
    )

    ranking = analyzer.degradation_ranking()
    stability = analyzer.stability_diagnostics()
    report = analyzer.to_report()

    assert {"slice_name", "group", "delta_error_rate", "severity_score"}.issubset(ranking.columns)
    assert "region x channel" in stability["slice_name"].tolist()
    assert "Subgroup reliability analysis" in report.to_markdown()
    assert "degradation_table" in report.to_dict()
