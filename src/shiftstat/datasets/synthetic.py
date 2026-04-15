"""Synthetic tabular datasets for shift experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shiftstat.utils.random import check_random_state


@dataclass(frozen=True)
class SyntheticShiftDataset:
    """Container for synthetic reference-target datasets."""

    X_ref: pd.DataFrame
    X_target: pd.DataFrame
    y_ref: np.ndarray
    y_target: np.ndarray
    reference_predictions: np.ndarray
    target_predictions: np.ndarray
    task: str


def _make_feature_frame(values: np.ndarray, *, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=feature_names)


def make_covariate_shift_classification(
    *,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    n_features: int = 6,
    shift_strength: float = 0.8,
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a binary classification problem with covariate shift."""

    rng = check_random_state(random_state)
    feature_names = [f"x{i}" for i in range(n_features)]
    X_ref = rng.normal(0.0, 1.0, size=(n_samples_ref, n_features))
    X_target = rng.normal(0.0, 1.0, size=(n_samples_target, n_features))
    X_target[:, : max(1, n_features // 2)] += shift_strength
    coefficients = rng.uniform(-1.2, 1.2, size=n_features)
    intercept = -0.1

    logits_ref = X_ref @ coefficients + intercept
    logits_target = X_target @ coefficients + intercept
    proba_ref = 1.0 / (1.0 + np.exp(-logits_ref))
    proba_target = 1.0 / (1.0 + np.exp(-logits_target))
    y_ref = rng.binomial(1, proba_ref)
    y_target = rng.binomial(1, proba_target)

    return SyntheticShiftDataset(
        X_ref=_make_feature_frame(X_ref, feature_names=feature_names),
        X_target=_make_feature_frame(X_target, feature_names=feature_names),
        y_ref=y_ref.astype(int),
        y_target=y_target.astype(int),
        reference_predictions=proba_ref.astype(float),
        target_predictions=proba_target.astype(float),
        task="classification",
    )


def make_covariate_shift_regression(
    *,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    n_features: int = 6,
    shift_strength: float = 1.0,
    noise: float = 0.4,
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a regression problem with shifted covariates."""

    rng = check_random_state(random_state)
    feature_names = [f"x{i}" for i in range(n_features)]
    X_ref = rng.normal(0.0, 1.0, size=(n_samples_ref, n_features))
    X_target = rng.normal(0.0, 1.0, size=(n_samples_target, n_features))
    X_target[:, ::2] += shift_strength
    coefficients = rng.uniform(-2.0, 2.0, size=n_features)

    pred_ref = X_ref @ coefficients
    pred_target = X_target @ coefficients
    y_ref = pred_ref + rng.normal(0.0, noise, size=n_samples_ref)
    y_target = pred_target + rng.normal(0.0, noise, size=n_samples_target)

    return SyntheticShiftDataset(
        X_ref=_make_feature_frame(X_ref, feature_names=feature_names),
        X_target=_make_feature_frame(X_target, feature_names=feature_names),
        y_ref=y_ref.astype(float),
        y_target=y_target.astype(float),
        reference_predictions=pred_ref.astype(float),
        target_predictions=pred_target.astype(float),
        task="regression",
    )


def make_mixed_type_shift(
    *,
    n_samples_ref: int = 400,
    n_samples_target: int = 400,
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a mixed continuous-categorical classification dataset with shift."""

    rng = check_random_state(random_state)
    ref_cont = rng.normal(size=(n_samples_ref, 3))
    target_cont = rng.normal(size=(n_samples_target, 3))
    target_cont[:, 0] += 1.0
    ref_cat = rng.choice(["a", "b", "c"], size=(n_samples_ref, 2), p=[0.6, 0.3, 0.1])
    target_cat = rng.choice(["a", "b", "c"], size=(n_samples_target, 2), p=[0.3, 0.5, 0.2])

    X_ref = pd.DataFrame(ref_cont, columns=["cont_0", "cont_1", "cont_2"])
    X_target = pd.DataFrame(target_cont, columns=["cont_0", "cont_1", "cont_2"])
    X_ref["cat_0"] = ref_cat[:, 0]
    X_ref["cat_1"] = ref_cat[:, 1]
    X_target["cat_0"] = target_cat[:, 0]
    X_target["cat_1"] = target_cat[:, 1]

    logits_ref = 0.8 * X_ref["cont_0"].to_numpy() - 0.6 * X_ref["cont_1"].to_numpy()
    logits_target = 0.8 * X_target["cont_0"].to_numpy() - 0.6 * X_target["cont_1"].to_numpy()
    logits_ref += (X_ref["cat_0"] == "b").astype(float).to_numpy()
    logits_target += (X_target["cat_0"] == "b").astype(float).to_numpy()

    proba_ref = 1.0 / (1.0 + np.exp(-logits_ref))
    proba_target = 1.0 / (1.0 + np.exp(-logits_target))
    y_ref = rng.binomial(1, proba_ref)
    y_target = rng.binomial(1, proba_target)

    return SyntheticShiftDataset(
        X_ref=X_ref,
        X_target=X_target,
        y_ref=y_ref.astype(int),
        y_target=y_target.astype(int),
        reference_predictions=proba_ref.astype(float),
        target_predictions=proba_target.astype(float),
        task="classification",
    )


def make_severity_controlled_shift(
    *,
    severity: float = 0.5,
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a classification dataset with a user-controlled shift severity."""

    return make_covariate_shift_classification(
        shift_strength=severity,
        random_state=random_state,
    )
