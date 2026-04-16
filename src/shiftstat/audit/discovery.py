"""Automatic discovery of interpretable failure slices."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from ..exceptions import SmallSampleWarning
from ..utils.probabilities import confidence_from_probabilities, labels_from_probabilities
from ..utils.schema import align_tabular_inputs, extract_feature_names, infer_feature_types
from ..subgroup.analyzer import group_metrics


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()  # type: ignore[no-any-return]
    array = np.asarray(X)
    return pd.DataFrame(array, columns=extract_feature_names(X))  # type: ignore[no-any-return]


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    feature_types = infer_feature_types(X)
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
    return (
        ColumnTransformer(transformers=transformers, remainder="drop"),
        categorical_columns,
    )


def _format_rule(
    feature_name: str,
    threshold: float,
    *,
    go_left: bool,
    categorical_columns: list[str],
) -> str:
    raw_name = feature_name.split("__", 1)[-1]
    for column in sorted(categorical_columns, key=len, reverse=True):
        prefix = f"{column}_"
        if raw_name.startswith(prefix) and abs(threshold - 0.5) < 1e-6:
            category = raw_name[len(prefix) :]
            comparator = "!=" if go_left else "=="
            return f"{column} {comparator} {category}"

    base_name = raw_name
    comparator = "<=" if go_left else ">"
    return f"{base_name} {comparator} {threshold:.3f}"


def _extract_leaf_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    *,
    categorical_columns: list[str],
) -> dict[int, str]:
    rules: dict[int, str] = {}
    tree_ = tree.tree_

    def walk(node: int, conditions: list[str]) -> None:
        is_leaf = tree_.children_left[node] == tree_.children_right[node]
        if is_leaf:
            rules[node] = " and ".join(conditions) if conditions else "all samples"
            return

        feature_index = int(tree_.feature[node])
        threshold = float(tree_.threshold[node])
        feature_name = feature_names[feature_index]
        left_rule = _format_rule(
            feature_name,
            threshold,
            go_left=True,
            categorical_columns=categorical_columns,
        )
        right_rule = _format_rule(
            feature_name,
            threshold,
            go_left=False,
            categorical_columns=categorical_columns,
        )
        walk(tree_.children_left[node], [*conditions, left_rule])
        walk(tree_.children_right[node], [*conditions, right_rule])

    walk(0, [])
    return rules


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


class SliceDiscoverer:
    """Discover interpretable slices with concentrated deployment failures."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        min_samples_leaf: int = 30,
        max_slices: int = 6,
        n_bins: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_slices = max_slices
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(
        self,
        X_ref: Any,
        y_ref: np.ndarray,
        p_ref: np.ndarray,
        X_target: Any,
        y_target: np.ndarray,
        p_target: np.ndarray,
        *,
        objective: str = "error",
        reference_weight: np.ndarray | None = None,
        target_weight: np.ndarray | None = None,
    ) -> SliceDiscoverer:
        """Fit a shallow decision tree to discover concentrated failures."""

        X_ref_aligned, X_target_aligned = align_tabular_inputs(X_ref, X_target)
        ref_frame = _as_frame(X_ref_aligned)
        target_frame = _as_frame(X_target_aligned)
        combined_frame = pd.concat([ref_frame, target_frame], axis=0, ignore_index=True)

        preprocessor, categorical_columns = _build_preprocessor(combined_frame)
        preprocessor.fit(combined_frame)
        ref_encoded = preprocessor.transform(ref_frame)
        target_encoded = preprocessor.transform(target_frame)
        feature_names = preprocessor.get_feature_names_out().tolist()

        target_failures = self._failure_indicator(y_target, p_target, objective=objective)
        if int(target_failures.sum()) < 5:
            warnings.warn(
                (
                    "Failure-slice discovery is operating with very few target failures. "
                    "Returned slices may be unstable and should be treated as hypotheses."
                ),
                SmallSampleWarning,
                stacklevel=2,
            )

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        tree.fit(target_encoded, target_failures)
        self.tree_ = tree
        self.preprocessor_ = preprocessor
        self.objective_ = objective

        leaf_rules = _extract_leaf_rules(
            tree,
            feature_names,
            categorical_columns=categorical_columns,
        )
        ref_leaf = tree.apply(ref_encoded)
        target_leaf = tree.apply(target_encoded)
        leaf_to_label = {
            leaf_id: f"slice_{index}"
            for index, leaf_id in enumerate(sorted(set(target_leaf.tolist())))
        }

        ref_groups = pd.Series([leaf_to_label[leaf_id] for leaf_id in ref_leaf])
        target_groups = pd.Series([leaf_to_label[leaf_id] for leaf_id in target_leaf])
        reference_metrics = group_metrics(
            y_ref,
            p_ref,
            ref_groups,
            sample_weight=reference_weight,
            n_bins=self.n_bins,
            dataset_name="reference",
            slice_name="discovered_slice",
            min_group_size=self.min_samples_leaf,
            warn_on_small_groups=False,
        )
        target_metrics = group_metrics(
            y_target,
            p_target,
            target_groups,
            sample_weight=target_weight,
            n_bins=self.n_bins,
            dataset_name="target",
            slice_name="discovered_slice",
            min_group_size=self.min_samples_leaf,
            warn_on_small_groups=False,
        )

        target_failure_total = int(np.sum(target_failures))
        target_confidence = confidence_from_probabilities(p_target)
        target_error = (labels_from_probabilities(p_target) != np.asarray(y_target, dtype=int)).astype(int)
        rows: list[dict[str, Any]] = []
        for leaf_id, slice_label in leaf_to_label.items():
            rule = leaf_rules.get(leaf_id, "all samples")
            reference_row = reference_metrics.loc[reference_metrics["group"] == slice_label]
            target_row = target_metrics.loc[target_metrics["group"] == slice_label]
            if reference_row.empty or target_row.empty:
                continue
            reference_record = reference_row.iloc[0]
            target_record = target_row.iloc[0]

            mask_target = target_groups == slice_label
            failure_count = int(np.sum(target_failures[mask_target.to_numpy()]))
            rows.append(
                {
                    "slice_label": slice_label,
                    "rule": rule,
                    "reference_n_samples": int(reference_record["n_samples"]),
                    "target_n_samples": int(target_record["n_samples"]),
                    "reference_accuracy": float(reference_record["accuracy"]),
                    "target_accuracy": float(target_record["accuracy"]),
                    "reference_error_rate": float(reference_record["error_rate"]),
                    "target_error_rate": float(target_record["error_rate"]),
                    "reference_ece": float(reference_record["ece"]),
                    "target_ece": float(target_record["ece"]),
                    "delta_accuracy": float(
                        target_record["accuracy"] - reference_record["accuracy"]
                    ),
                    "delta_error_rate": float(
                        target_record["error_rate"] - reference_record["error_rate"]
                    ),
                    "delta_ece": float(target_record["ece"] - reference_record["ece"]),
                    "target_sample_share": float(target_record["sample_share"]),
                    "target_failure_count": failure_count,
                    "target_failure_share": float(
                        failure_count / max(target_failure_total, 1)
                    ),
                    "failure_concentration": float(
                        _safe_ratio(
                            failure_count / max(target_failure_total, 1),
                            float(target_record["sample_share"]),
                        )
                    ),
                    "mean_target_confidence": float(
                        np.mean(target_confidence[mask_target.to_numpy()])
                    ),
                    "mean_target_error_indicator": float(
                        np.mean(target_error[mask_target.to_numpy()])
                    ),
                    "support_ok": bool(
                        bool(reference_record["support_ok"]) and bool(target_record["support_ok"])
                    ),
                }
            )

        self.summary_ = pd.DataFrame.from_records(rows).sort_values(
            ["failure_concentration", "delta_error_rate"],
            ascending=[False, False],
            ignore_index=True,
        )
        if not self.summary_.empty:
            self.summary_ = self.summary_.head(self.max_slices).reset_index(drop=True)
        self.caveats_ = [
            "Failure slices are descriptive partitions from a shallow decision tree; they are not causal explanations.",
            "Discovery is performed on the target sample and can be optimistic without external validation.",
        ]
        return self

    def summary(self) -> pd.DataFrame:
        """Return discovered slices ranked by failure concentration."""

        self._check_is_fitted()
        return self.summary_.copy()  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable representation."""

        self._check_is_fitted()
        return {
            "objective": self.objective_,
            "summary": self.summary_.to_dict(orient="records"),
            "caveats": self.caveats_,
        }

    def _failure_indicator(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        objective: str,
    ) -> np.ndarray:
        y_arr = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(y_prob, dtype=float)
        errors = (labels_from_probabilities(probabilities) != y_arr).astype(int)
        if objective == "error":
            return errors
        if objective == "high_confidence_error":
            confidence = confidence_from_probabilities(probabilities)
            return np.asarray((errors == 1) & (confidence >= 0.8), dtype=int)
        raise ValueError("objective must be 'error' or 'high_confidence_error'.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "summary_"):
            raise ValueError("SliceDiscoverer must be fitted before accessing results.")


def discover_failure_slices(
    X_ref: Any,
    y_ref: np.ndarray,
    p_ref: np.ndarray,
    X_target: Any,
    y_target: np.ndarray,
    p_target: np.ndarray,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper returning discovered failure slices as a DataFrame."""

    return SliceDiscoverer(**kwargs).fit(
        X_ref,
        y_ref,
        p_ref,
        X_target,
        y_target,
        p_target,
    ).summary()
