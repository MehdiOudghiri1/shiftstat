"""Abstention policies for selective prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shiftstat.metrics import risk_coverage_table
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
    labels_from_probabilities,
    predictive_entropy,
)
from shiftstat.utils.validation import ensure_1d, validate_same_length


@dataclass(frozen=True)
class _ThresholdSelection:
    threshold: float
    achieved_coverage: float
    achieved_selective_risk: float
    achieved_selective_accuracy: float
    objective: str
    constraint_satisfied: bool


class AbstentionPolicy:
    """Interpretable abstention policy with optional threshold tuning."""

    def __init__(
        self,
        *,
        method: str = "confidence",
        threshold: float | None = None,
        name: str | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.name = name or f"{method}_abstention"
        self.random_state = random_state

    def fit(
        self,
        y_true: np.ndarray | None,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        sample_weight: np.ndarray | None = None,
        threshold: float | None = None,
        target_coverage: float | None = None,
        target_risk: float | None = None,
        max_thresholds: int = 101,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> AbstentionPolicy:
        """Fit any learned selector and resolve the deployment threshold."""

        probabilities = extract_positive_class_probabilities(y_prob)
        y_true_arr = None if y_true is None else ensure_1d(y_true, name="y_true").astype(int)
        if y_true_arr is not None:
            validate_same_length(y_true_arr, probabilities)
        weights = None if sample_weight is None else ensure_1d(sample_weight, name="sample_weight").astype(float)
        if weights is not None:
            validate_same_length(probabilities, weights)

        if self.method == "learned_risk":
            if y_true_arr is None:
                raise ValueError("y_true is required to fit a learned abstention policy.")
            self.selector_ = LogisticRegression(max_iter=3000, random_state=self.random_state)
            self.selector_.fit(
                self._selection_features(probabilities),
                (labels_from_probabilities(probabilities) == y_true_arr).astype(int),
                sample_weight=weights,
            )

        scores = self.score(probabilities, X=X)
        explicit_threshold = threshold if threshold is not None else self.threshold
        if explicit_threshold is not None and (
            target_coverage is not None or target_risk is not None
        ):
            raise ValueError(
                "Use either a fixed threshold or a tuning target, not both at once."
            )
        if explicit_threshold is not None:
            accepted = scores >= float(explicit_threshold)
            if weights is None:
                achieved_coverage = float(np.mean(accepted))
            else:
                achieved_coverage = float(np.sum(weights[accepted]) / max(np.sum(weights), 1e-12))
            selection = _ThresholdSelection(
                threshold=float(explicit_threshold),
                achieved_coverage=achieved_coverage,
                achieved_selective_risk=float("nan") if y_true_arr is None else float(
                    risk_coverage_table(
                        y_true_arr,
                        probabilities,
                        scores,
                        thresholds=np.asarray([float(explicit_threshold)]),
                        sample_weight=weights,
                        n_bins=n_bins,
                        strategy=strategy,
                    )["selective_risk"].iloc[0]
                ),
                achieved_selective_accuracy=float("nan") if y_true_arr is None else float(
                    risk_coverage_table(
                        y_true_arr,
                        probabilities,
                        scores,
                        thresholds=np.asarray([float(explicit_threshold)]),
                        sample_weight=weights,
                        n_bins=n_bins,
                        strategy=strategy,
                    )["selective_accuracy"].iloc[0]
                ),
                objective="fixed_threshold",
                constraint_satisfied=True,
            )
        elif target_coverage is not None or target_risk is not None:
            if y_true_arr is None:
                raise ValueError("y_true is required for threshold tuning.")
            if target_coverage is not None and target_risk is not None:
                raise ValueError("Specify at most one of target_coverage and target_risk.")
            curve = risk_coverage_table(
                y_true_arr,
                probabilities,
                scores,
                sample_weight=weights,
                n_bins=n_bins,
                strategy=strategy,
                max_points=max_thresholds,
            )
            selection = (
                self._select_threshold_for_coverage(curve, float(target_coverage))
                if target_coverage is not None
                else self._select_threshold_for_risk(curve, float(target_risk))
            )
            self.threshold_curve_ = curve
        elif self.method == "learned_risk":
            selection = _ThresholdSelection(
                threshold=0.5,
                achieved_coverage=float(np.mean(scores >= 0.5)),
                achieved_selective_risk=float("nan"),
                achieved_selective_accuracy=float("nan"),
                objective="default_probability_cutoff",
                constraint_satisfied=True,
            )
        else:
            raise ValueError(
                "Provide an explicit threshold or a target_coverage/target_risk objective."
            )

        self.threshold_ = float(selection.threshold)
        self.tuning_summary_ = {
            "policy_name": self.name,
            "score_method": self.method,
            "threshold": float(selection.threshold),
            "objective": selection.objective,
            "target_coverage": target_coverage,
            "target_risk": target_risk,
            "achieved_coverage": selection.achieved_coverage,
            "achieved_selective_risk": selection.achieved_selective_risk,
            "achieved_selective_accuracy": selection.achieved_selective_accuracy,
            "weighted": sample_weight is not None,
            "constraint_satisfied": selection.constraint_satisfied,
        }
        return self

    def score(self, y_prob: np.ndarray, *, X: Any | None = None) -> np.ndarray:
        """Return selection scores where larger values indicate safer predictions."""

        probabilities = extract_positive_class_probabilities(y_prob)
        if self.method == "confidence":
            return confidence_from_probabilities(probabilities)
        if self.method == "entropy":
            entropy = predictive_entropy(probabilities)
            return np.asarray(1.0 - entropy / np.log(2.0), dtype=float)
        if self.method == "margin":
            return np.asarray(np.abs(2.0 * probabilities - 1.0), dtype=float)
        if self.method == "learned_risk":
            if not hasattr(self, "selector_"):
                raise ValueError("Learned abstention policy must be fitted before scoring.")
            return np.asarray(
                self.selector_.predict_proba(self._selection_features(probabilities))[:, 1],
                dtype=float,
            )
        raise ValueError(
            "method must be one of {'confidence', 'entropy', 'margin', 'learned_risk'}."
        )

    def accept_mask(
        self,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a boolean mask for accepted predictions."""

        score = self.score(y_prob, X=X)
        resolved_threshold = threshold
        if resolved_threshold is None:
            if not hasattr(self, "threshold_"):
                if self.threshold is None:
                    raise ValueError("No abstention threshold is available. Fit or set one first.")
                resolved_threshold = self.threshold
            else:
                resolved_threshold = self.threshold_
        return np.asarray(score >= float(resolved_threshold), dtype=bool)

    def predict(
        self,
        y_prob: np.ndarray,
        *,
        X: Any | None = None,
        reject_value: int = -1,
    ) -> np.ndarray:
        """Return hard predictions with rejected samples marked by `reject_value`."""

        predictions = labels_from_probabilities(y_prob)
        accepted = self.accept_mask(y_prob, X=X)
        output = predictions.copy()
        output[~accepted] = int(reject_value)
        return output

    def copy(self) -> AbstentionPolicy:
        """Return a shallow configuration copy."""

        return AbstentionPolicy(
            method=self.method,
            threshold=self.threshold,
            name=self.name,
            random_state=self.random_state,
        )

    def summary(self) -> dict[str, Any]:
        """Return the fitted policy summary."""

        if not hasattr(self, "tuning_summary_"):
            return {
                "policy_name": self.name,
                "score_method": self.method,
                "threshold": self.threshold,
                "objective": None,
            }
        return dict(self.tuning_summary_)

    def _selection_features(self, probabilities: np.ndarray) -> np.ndarray:
        confidence = confidence_from_probabilities(probabilities)
        margin = np.abs(2.0 * probabilities - 1.0)
        entropy = predictive_entropy(probabilities)
        return np.column_stack([probabilities, confidence, margin, entropy])

    def _select_threshold_for_coverage(
        self,
        curve: pd.DataFrame,
        target_coverage: float,
    ) -> _ThresholdSelection:
        ranked = curve.assign(
            coverage_gap=np.abs(curve["coverage"] - target_coverage),
        ).sort_values(
            ["coverage_gap", "selective_risk", "threshold"],
            ascending=[True, True, True],
        )
        chosen = ranked.iloc[0]
        return _ThresholdSelection(
            threshold=float(chosen["threshold"]),
            achieved_coverage=float(chosen["coverage"]),
            achieved_selective_risk=float(chosen["selective_risk"]),
            achieved_selective_accuracy=float(chosen["selective_accuracy"]),
            objective="target_coverage",
            constraint_satisfied=True,
        )

    def _select_threshold_for_risk(
        self,
        curve: pd.DataFrame,
        target_risk: float,
    ) -> _ThresholdSelection:
        feasible = curve.loc[curve["selective_risk"] <= target_risk].sort_values(
            ["coverage", "selective_risk", "threshold"],
            ascending=[False, True, True],
        )
        if not feasible.empty:
            chosen = feasible.iloc[0]
            return _ThresholdSelection(
                threshold=float(chosen["threshold"]),
                achieved_coverage=float(chosen["coverage"]),
                achieved_selective_risk=float(chosen["selective_risk"]),
                achieved_selective_accuracy=float(chosen["selective_accuracy"]),
                objective="target_risk",
                constraint_satisfied=True,
            )

        fallback = curve.sort_values(
            ["selective_risk", "coverage", "threshold"],
            ascending=[True, False, True],
        ).iloc[0]
        return _ThresholdSelection(
            threshold=float(fallback["threshold"]),
            achieved_coverage=float(fallback["coverage"]),
            achieved_selective_risk=float(fallback["selective_risk"]),
            achieved_selective_accuracy=float(fallback["selective_accuracy"]),
            objective="target_risk",
            constraint_satisfied=False,
        )
