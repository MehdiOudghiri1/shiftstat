"""Example: subgroup-specific rejection analysis."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.datasets import make_hidden_subgroup_shift_classification
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


def run_example(random_state: int = 103) -> dict[str, object]:
    """Measure how abstention behavior differs across deployment slices."""

    data = make_hidden_subgroup_shift_classification(
        pattern="minority_subgroup_degradation",
        random_state=random_state,
    )
    result = evaluate_selective_under_shift(
        _estimator(),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        policy_method="confidence",
        target_coverage=0.8,
        subgroup_features=["region", "channel", "load"],
        intersectional_features=[("region", "channel")],
        random_state=random_state,
    )
    top_gap = result.subgroup_abstention_disparities[0]
    return {
        "top_gap_metric": top_gap["metric"],
        "top_gap_size": float(top_gap["absolute_gap"]),
        "worst_group": str(top_gap["worst_group"]),
    }


if __name__ == "__main__":
    print(run_example())
