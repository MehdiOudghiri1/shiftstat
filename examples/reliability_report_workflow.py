"""End-to-end reliability workflow under shift."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reliability import evaluate_under_shift


def run_example(random_state: int = 24) -> dict[str, object]:
    """Detect shift, reweight, recalibrate, and render a reliability report."""

    data = make_covariate_shift_classification(random_state=random_state, shift_strength=1.2)
    detector = ShiftDetector(random_state=random_state).fit(data.X_ref, data.X_target)
    result = evaluate_under_shift(
        LogisticRegression(max_iter=2000),
        data.X_ref,
        data.y_ref,
        data.X_target,
        data.y_target,
        apply_importance_weighting=True,
        recalibration="temperature",
        random_state=random_state,
    )
    report = result.to_report()
    return {
        "shift_summary": detector.dataset_summary_.to_dict(),
        "evaluation_summary": result.summary_frame(),
        "report_markdown": report.to_markdown(),
    }


if __name__ == "__main__":
    output = run_example()
    print(output["evaluation_summary"])
