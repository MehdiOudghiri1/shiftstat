"""Example: weighted threshold tuning under covariate shift."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.selective import evaluate_selective_under_shift


def run_example(random_state: int = 102) -> dict[str, object]:
    """Compare weighted and unweighted threshold tuning."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.1)
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
        random_state=random_state,
    )
    return {
        "policy_summary": result.policy_summary,
        "threshold_comparison": result.threshold_comparison,
        "coverage": float(result.target_selective_profile.coverage),
    }


if __name__ == "__main__":
    print(run_example())
