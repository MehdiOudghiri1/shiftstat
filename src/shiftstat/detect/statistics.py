"""Statistical routines used by the shift detection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.detect.results import ClassifierShiftResult
from shiftstat.typing import FeatureTypes, TabularLike
from shiftstat.utils.random import random_state_to_int
from shiftstat.utils.schema import extract_feature_names


@dataclass(frozen=True)
class TestStatisticResult:
    """Container for a feature-level test statistic."""

    test_name: str
    statistic: float
    p_value: float
    psi: float
    wasserstein_distance: float | None


def _to_dataframe(X: TabularLike) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X), columns=extract_feature_names(X))  # type: ignore[no-any-return]


def apply_multiple_testing_correction(
    p_values: list[float],
    *,
    method: str = "benjamini-hochberg",
) -> list[float]:
    """Adjust p-values using a lightweight multiple-testing correction."""

    if method == "none":
        return p_values

    values = np.asarray(p_values, dtype=float)
    if method == "bonferroni":
        return [float(item) for item in np.clip(values * len(values), 0.0, 1.0)]

    if method != "benjamini-hochberg":
        raise ValueError(f"Unsupported multiple testing correction method: {method}.")

    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    n_tests = len(ranked)
    running_min = 1.0
    for index in range(n_tests - 1, -1, -1):
        rank = index + 1
        candidate = ranked[index] * n_tests / rank
        running_min = min(running_min, candidate)
        adjusted[index] = running_min
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(item) for item in restored]


def compute_psi(
    reference: np.ndarray,
    new: np.ndarray,
    *,
    feature_type: str,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute a Population Stability Index style drift score."""

    ref = pd.Series(reference).dropna()
    tgt = pd.Series(new).dropna()
    if ref.empty or tgt.empty:
        return 0.0

    if feature_type == "continuous":
        try:
            _, bins = pd.qcut(ref, q=min(n_bins, ref.nunique()), retbins=True, duplicates="drop")
        except ValueError:
            bins = np.linspace(ref.min(), ref.max(), min(n_bins, ref.nunique()) + 1)
        if len(np.unique(bins)) < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref, bins=bins)
        tgt_counts, _ = np.histogram(tgt, bins=bins)
    else:
        categories = sorted(set(ref.astype(str)) | set(tgt.astype(str)))
        ref_counts = ref.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        tgt_counts = tgt.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()

    ref_dist = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    tgt_dist = np.clip(tgt_counts / max(tgt_counts.sum(), 1), epsilon, None)
    return float(np.sum((tgt_dist - ref_dist) * np.log(tgt_dist / ref_dist)))


def continuous_shift_test(
    reference: np.ndarray,
    new: np.ndarray,
    *,
    n_bins: int = 10,
) -> TestStatisticResult:
    """Compute continuous-feature shift diagnostics."""

    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    tgt = pd.Series(new).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(tgt) == 0:
        return TestStatisticResult("ks", 0.0, 1.0, 0.0, 0.0)

    ks_result = ks_2samp(ref, tgt, method="auto")
    psi = compute_psi(ref, tgt, feature_type="continuous", n_bins=n_bins)
    distance = float(wasserstein_distance(ref, tgt))
    return TestStatisticResult(
        test_name="kolmogorov_smirnov",
        statistic=float(ks_result.statistic),
        p_value=float(ks_result.pvalue),
        psi=psi,
        wasserstein_distance=distance,
    )


def categorical_shift_test(reference: np.ndarray, new: np.ndarray) -> TestStatisticResult:
    """Compute categorical-feature shift diagnostics."""

    ref = pd.Series(reference).dropna().astype(str)
    tgt = pd.Series(new).dropna().astype(str)
    if ref.empty or tgt.empty:
        return TestStatisticResult("chi_square", 0.0, 1.0, 0.0, None)

    categories = sorted(set(ref) | set(tgt))
    contingency = np.vstack(
        [
            ref.value_counts().reindex(categories, fill_value=0).to_numpy(),
            tgt.value_counts().reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    statistic, p_value, _, _ = chi2_contingency(contingency)
    psi = compute_psi(ref.to_numpy(), tgt.to_numpy(), feature_type="categorical")
    return TestStatisticResult(
        test_name="chi_square",
        statistic=float(statistic),
        p_value=float(p_value),
        psi=psi,
        wasserstein_distance=None,
    )


def compute_severity_score(
    *,
    statistic: float,
    adjusted_p_value: float,
    psi: float,
    wasserstein_distance: float | None,
) -> float:
    """Compute a bounded heuristic severity score for ranking features."""

    p_component = min(-np.log10(max(adjusted_p_value, 1e-12)) / 6.0, 1.5)
    psi_component = min(psi / 0.25, 1.5)
    statistic_component = min(abs(statistic), 1.5)
    parts = [p_component, psi_component, statistic_component]
    if wasserstein_distance is not None:
        parts.append(min(wasserstein_distance, 1.5))
    return float(np.mean(parts))


def classifier_two_sample_test(
    X_ref: TabularLike,
    X_new: TabularLike,
    *,
    feature_types: FeatureTypes,
    test_size: float = 0.3,
    random_state: int | np.random.RandomState | None = None,
) -> ClassifierShiftResult:
    """Fit a source-vs-target classifier and return holdout AUC diagnostics."""

    ref_df = _to_dataframe(X_ref)
    new_df = _to_dataframe(X_new)
    combined = pd.concat([ref_df, new_df], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(ref_df), dtype=int), np.ones(len(new_df), dtype=int)])

    categorical_columns = [name for name, kind in feature_types.items() if kind == "categorical"]
    continuous_columns = [name for name, kind in feature_types.items() if kind == "continuous"]

    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )
    if continuous_columns:
        transformers.append(("continuous", "passthrough", continuous_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )

    seed = random_state_to_int(random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        combined,
        y,
        stratify=y,
        test_size=test_size,
        random_state=seed,
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    fpr, tpr, threshold = roc_curve(y_test, probabilities)

    if auc < 0.55:
        interpretation = "Weak source-target separability."
    elif auc < 0.65:
        interpretation = "Mild source-target separability."
    elif auc < 0.8:
        interpretation = "Moderate source-target separability."
    else:
        interpretation = "Strong source-target separability."

    return ClassifierShiftResult(
        auc=auc,
        interpretation=interpretation,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=threshold.tolist(),
    )
