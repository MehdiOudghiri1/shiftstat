from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.datasets import (
    make_covariate_shift_classification,
    make_hidden_subgroup_shift_classification,
)
from shiftstat.metrics import retained_coverage, risk_coverage_table, selective_accuracy, selective_summary
from shiftstat.selective import AbstentionPolicy, SelectivePredictor, evaluate_selective_under_shift


def _mixed_estimator() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["region", "channel"],
                        ),
                        ("continuous", "passthrough", ["score", "load", "signal"]),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )


def test_abstention_policy_thresholding_logic() -> None:
    probabilities = [0.9, 0.2, 0.55, 0.8]
    confidence_policy = AbstentionPolicy(method="confidence", threshold=0.8)
    margin_policy = AbstentionPolicy(method="margin", threshold=0.6)

    assert confidence_policy.accept_mask(probabilities).tolist() == [True, True, False, True]
    assert margin_policy.accept_mask(probabilities).tolist() == [True, True, False, True]


def test_selective_metrics_support_weighted_evaluation() -> None:
    y_true = [1, 0, 1]
    y_prob = [0.9, 0.8, 0.6]
    accepted = [True, False, True]
    weights = [1.0, 5.0, 1.0]

    summary = selective_summary(y_true, y_prob, accepted, sample_weight=weights)

    assert retained_coverage(accepted, sample_weight=weights) == 2.0 / 7.0
    assert selective_accuracy(y_true, y_prob, accepted, sample_weight=weights) == 1.0
    assert summary["selective_risk"] == 0.0


def test_risk_coverage_table_includes_full_coverage_point() -> None:
    y_true = [1, 0, 1, 0]
    y_prob = [0.95, 0.15, 0.7, 0.55]
    scores = [0.95, 0.85, 0.7, 0.55]

    frame = risk_coverage_table(y_true, y_prob, scores, max_points=5)

    assert frame["coverage"].iloc[0] == 1.0
    assert frame["selective_risk"].iloc[0] == 0.25
    assert frame["coverage"].is_monotonic_decreasing


def test_selective_predictor_and_shift_workflow_integration() -> None:
    data = make_covariate_shift_classification(random_state=120, shift_strength=1.1)
    result = evaluate_selective_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        policy_method="confidence",
        target_coverage=0.8,
        apply_importance_weighting=True,
        use_weighted_threshold_tuning=True,
        compare_threshold_tuning=True,
        recalibration="temperature",
        random_state=120,
    )

    assert result.threshold_comparison is not None
    assert result.target_selective_profile.coverage > 0.0
    assert "Selective deployment report" in result.to_report().to_markdown()


def test_subgroup_abstention_summary_is_exposed() -> None:
    data = make_hidden_subgroup_shift_classification(random_state=121)
    result = evaluate_selective_under_shift(
        _mixed_estimator(),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        policy_method="confidence",
        target_coverage=0.8,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
        random_state=121,
    )

    summary = result.to_report().subgroup_abstention_frame()
    disparity = result.to_report().subgroup_disparity_frame()

    assert {"slice_name", "group", "target_coverage", "target_abstention_rate"}.issubset(summary.columns)
    assert {"metric", "worst_group", "absolute_gap"}.issubset(disparity.columns)
