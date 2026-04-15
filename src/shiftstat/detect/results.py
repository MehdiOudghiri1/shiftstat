"""Result containers for shift detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureShiftResult:
    """Feature-level shift diagnostics."""

    feature_name: str
    feature_type: str
    test_name: str
    statistic: float
    p_value: float
    adjusted_p_value: float
    reject_null: bool
    psi: float
    wasserstein_distance: float | None
    severity_score: float
    reference_size: int
    new_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class ClassifierShiftResult:
    """Classifier two-sample test output."""

    auc: float
    interpretation: str
    fpr: list[float]
    tpr: list[float]
    threshold: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class DatasetShiftSummary:
    """Dataset-level summary of a fitted shift detector."""

    n_features: int
    n_shifted_features: int
    shifted_fraction: float
    mean_psi: float
    max_severity: float
    min_adjusted_p_value: float
    classifier_auc: float
    overall_shift_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)

