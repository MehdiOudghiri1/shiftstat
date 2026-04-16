"""Detect feature drift between a reference and target dataset."""

from __future__ import annotations

from shiftstat.datasets import make_mixed_type_shift
from shiftstat.detect import ShiftDetector


def run_example(random_state: int = 7) -> dict[str, object]:
    """Run a compact drift-detection example."""

    data = make_mixed_type_shift(random_state=random_state)
    detector = ShiftDetector(
        categorical_features=["cat_0", "cat_1"],
        random_state=random_state,
    )
    detector.fit(data.X_ref, data.X_target)
    summary = detector.summary()
    report = detector.to_report()
    return {
        "summary": summary,
        "dataset_summary": detector.dataset_summary_.to_dict(),
        "report_markdown": report.to_markdown(),
    }


if __name__ == "__main__":
    result = run_example()
    print(result["summary"].head())
    print(result["report_markdown"])
