"""Recalibration under shift on a semi-real tabular dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from shiftstat.reliability import evaluate_under_shift


def _make_shifted_breast_cancer(
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    X_ref, X_target, y_ref, y_target = train_test_split(
        X,
        y,
        test_size=0.35,
        stratify=y,
        random_state=random_state,
    )
    shifted = X_target.copy()
    shifted.iloc[:, :5] = shifted.iloc[:, :5] * 1.15 + 0.25
    return X_ref, shifted, y_ref, y_target


def run_example(random_state: int = 23) -> dict[str, object]:
    """Compare target reliability before and after temperature scaling."""

    X_ref, X_target, y_ref, y_target = _make_shifted_breast_cancer(random_state)
    estimator = LogisticRegression(max_iter=4000, solver="liblinear")
    result = evaluate_under_shift(
        estimator,
        X_ref,
        y_ref.to_numpy(),
        X_target,
        y_target.to_numpy(),
        recalibration="temperature",
        random_state=random_state,
    )
    return {
        "summary": result.summary_frame(),
        "report_markdown": result.to_report().to_markdown(),
    }


if __name__ == "__main__":
    output = run_example()
    print(output["summary"])
