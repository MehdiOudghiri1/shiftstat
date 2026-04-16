"""Example: from shift diagnosis to selective deployment."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.datasets import make_hidden_subgroup_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.selective import evaluate_selective_under_shift


def _estimator() -> Pipeline:
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


def run_example(random_state: int = 104) -> dict[str, object]:
    """Run an end-to-end selective deployment workflow."""

    data = make_hidden_subgroup_shift_classification(
        pattern="masked_subgroup_shift",
        random_state=random_state,
    )
    detector = ShiftDetector(random_state=random_state).fit(data.X_ref, data.X_target)
    result = evaluate_selective_under_shift(
        _estimator(),
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
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
        random_state=random_state,
    )
    report = result.to_report()
    return {
        "shift_detected": bool(detector.dataset_summary_.overall_shift_detected),
        "coverage": float(result.target_selective_profile.coverage),
        "risk_reduction": float(result.target_selective_profile.risk_reduction),
        "report_markdown": report.to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
