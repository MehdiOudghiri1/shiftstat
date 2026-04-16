"""Example: selective abstention under distribution shift."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.selective import evaluate_selective_under_shift


def run_example(random_state: int = 101) -> dict[str, object]:
    """Evaluate a confidence-threshold policy under covariate shift."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.0)
    result = evaluate_selective_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        policy_method="confidence",
        target_coverage=0.8,
        random_state=random_state,
    )
    return {
        "coverage": float(result.target_selective_profile.coverage),
        "risk_reduction": float(result.target_selective_profile.risk_reduction),
        "report_markdown": result.to_report().to_markdown(),
    }


if __name__ == "__main__":
    print(run_example())
