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


def make_configurable_shift_classification(
    *,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    n_numeric_features: int = 6,
    n_categorical_features: int = 2,
    n_categories: int = 4,
    shift_strength: float = 1.0,
    noise: float = 0.35,
    class_imbalance: float = 0.4,
    minority_rate: float = 0.2,
    shift_pattern: str = "mixed",
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a configurable mixed-type classification benchmark dataset.

    Parameters
    ----------
    shift_pattern:
        One of ``"covariate"``, ``"subgroup"``, ``"calibration"``, or ``"mixed"``.
        The patterns are intentionally simple and interpretable so they can serve
        as reproducible benchmark cases rather than domain-realistic simulators.
    """

    if not 0.0 < class_imbalance < 1.0:
        raise ValueError("class_imbalance must lie strictly between 0 and 1.")
    if n_numeric_features < 2:
        raise ValueError("n_numeric_features must be at least 2.")
    if n_categorical_features < 1:
        raise ValueError("n_categorical_features must be at least 1.")
    if n_categories < 2:
        raise ValueError("n_categories must be at least 2.")
    if shift_pattern not in {"covariate", "subgroup", "calibration", "mixed"}:
        raise ValueError(
            "shift_pattern must be one of 'covariate', 'subgroup', 'calibration', or 'mixed'."
        )

    rng = check_random_state(random_state)
    numeric_names = [f"x{i}" for i in range(n_numeric_features)]
    categorical_names = [f"cat_{i}" for i in range(n_categorical_features)]
    category_levels = [f"level_{i}" for i in range(n_categories)]

    X_ref_numeric = rng.normal(0.0, 1.0, size=(n_samples_ref, n_numeric_features))
    X_target_numeric = rng.normal(0.0, 1.0, size=(n_samples_target, n_numeric_features))
    X_target_numeric[:, : max(2, n_numeric_features // 2)] += shift_strength

    ref_categories = {
        name: rng.choice(
            category_levels,
            size=n_samples_ref,
            p=_category_probabilities(n_categories, concentration=1.0 + idx * 0.2),
        )
        for idx, name in enumerate(categorical_names)
    }
    target_categories = {
        name: rng.choice(
            category_levels,
            size=n_samples_target,
            p=_category_probabilities(
                n_categories,
                concentration=1.0 + idx * 0.2,
                tilt=shift_strength * 0.18,
            ),
        )
        for idx, name in enumerate(categorical_names)
    }

    ref_segment = rng.choice(
        ["majority", "minority"],
        size=n_samples_ref,
        p=[1.0 - minority_rate, minority_rate],
    )
    target_minority_rate = min(minority_rate + 0.05, 0.45)
    target_segment = rng.choice(
        ["majority", "minority"],
        size=n_samples_target,
        p=[1.0 - target_minority_rate, target_minority_rate],
    )

    X_ref = pd.DataFrame(X_ref_numeric, columns=numeric_names)
    X_target = pd.DataFrame(X_target_numeric, columns=numeric_names)
    for name in categorical_names:
        X_ref[name] = ref_categories[name]
        X_target[name] = target_categories[name]
    X_ref["segment"] = ref_segment
    X_target["segment"] = target_segment

    numeric_coefficients = rng.uniform(-1.2, 1.2, size=n_numeric_features)
    intercept = float(np.log(class_imbalance / (1.0 - class_imbalance)))
    category_bonus = {
        level: float(value)
        for level, value in zip(
            category_levels,
            np.linspace(-0.35, 0.35, num=n_categories),
            strict=True,
        )
    }

    ref_base_logit = _configurable_base_logit(
        X_ref,
        numeric_coefficients=numeric_coefficients,
        category_bonus=category_bonus,
        categorical_names=categorical_names,
        intercept=intercept,
    )
    target_base_logit = _configurable_base_logit(
        X_target,
        numeric_coefficients=numeric_coefficients,
        category_bonus=category_bonus,
        categorical_names=categorical_names,
        intercept=intercept,
    )

    ref_prediction_logit = ref_base_logit.copy()
    target_prediction_logit = target_base_logit.copy()
    target_truth_logit = target_base_logit.copy()

    minority_target = (X_target["segment"] == "minority").to_numpy(dtype=float)
    high_load_target = (X_target["x1"].to_numpy(dtype=float) > 0.4).astype(float)
    difficult_region = (X_target["x0"].to_numpy(dtype=float) < 0.0).astype(float)

    if shift_pattern in {"covariate", "mixed"}:
        target_truth_logit += 0.45 * shift_strength * X_target["x0"].to_numpy(dtype=float)
        target_truth_logit -= 0.25 * shift_strength * X_target["x1"].to_numpy(dtype=float)
    if shift_pattern in {"subgroup", "mixed"}:
        target_truth_logit += 1.05 * shift_strength * minority_target * high_load_target
        target_truth_logit += 0.35 * shift_strength * minority_target
    if shift_pattern in {"calibration", "mixed"}:
        target_prediction_logit += 0.55 * shift_strength * difficult_region
        target_prediction_logit += 0.25 * shift_strength * minority_target

    y_ref = rng.binomial(1, _sigmoid(ref_base_logit + rng.normal(0.0, noise, size=n_samples_ref)))
    y_target = rng.binomial(
        1,
        _sigmoid(target_truth_logit + rng.normal(0.0, noise, size=n_samples_target)),
    )

    return SyntheticShiftDataset(
        X_ref=X_ref,
        X_target=X_target,
        y_ref=y_ref.astype(int),
        y_target=y_target.astype(int),
        reference_predictions=_sigmoid(ref_prediction_logit).astype(float),
        target_predictions=_sigmoid(target_prediction_logit).astype(float),
        task="classification",
    )


def _category_probabilities(
    n_categories: int,
    *,
    concentration: float,
    tilt: float = 0.0,
) -> np.ndarray:
    base = np.linspace(1.0, concentration, num=n_categories)
    if tilt != 0.0:
        base += np.linspace(0.0, tilt * n_categories, num=n_categories)
    normalized = base / np.sum(base)
    return np.asarray(normalized, dtype=float)


def _configurable_base_logit(
    frame: pd.DataFrame,
    *,
    numeric_coefficients: np.ndarray,
    category_bonus: dict[str, float],
    categorical_names: list[str],
    intercept: float,
) -> np.ndarray:
    numeric_matrix = frame[[f"x{i}" for i in range(len(numeric_coefficients))]].to_numpy(
        dtype=float
    )
    logit = numeric_matrix @ numeric_coefficients + intercept
    for name in categorical_names:
        logit += frame[name].map(category_bonus).to_numpy(dtype=float)
    logit += 0.45 * (frame["segment"] == "minority").to_numpy(dtype=float)
    return np.asarray(logit, dtype=float)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)
