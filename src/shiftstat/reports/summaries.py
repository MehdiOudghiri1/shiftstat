"""Scientific summary objects for markdown, dict, and tabular export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from shiftstat.detect.detector import ShiftDetector
    from shiftstat.reweight.importance import ImportanceWeighter


@dataclass(frozen=True)
class DetectionReport:
    """Reporting wrapper around a fitted `ShiftDetector`."""

    dataset_summary: dict[str, Any]
    per_feature: pd.DataFrame
    classifier_summary: dict[str, Any]

    @classmethod
    def from_detector(cls, detector: ShiftDetector) -> DetectionReport:
        """Construct a report from a fitted detector."""

        return cls(
            dataset_summary=detector.dataset_summary_.to_dict(),
            per_feature=detector.summary(),
            classifier_summary=detector.classifier_result_.to_dict(),
        )

    def to_markdown(self) -> str:
        """Render a concise markdown summary."""

        shifted = self.dataset_summary["n_shifted_features"]
        total = self.dataset_summary["n_features"]
        auc = self.dataset_summary["classifier_auc"]
        top_feature = (
            self.per_feature.iloc[0]["feature_name"] if not self.per_feature.empty else "n/a"
        )
        return (
            f"## Shift summary\n\n"
            f"- Shifted features: {shifted} / {total}\n"
            f"- Mean PSI: {self.dataset_summary['mean_psi']:.3f}\n"
            f"- Classifier AUC: {auc:.3f}\n"
            f"- Top shifted feature: {top_feature}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary summary."""

        return {
            "dataset_summary": self.dataset_summary,
            "classifier_summary": self.classifier_summary,
            "per_feature": self.per_feature.to_dict(orient="records"),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the per-feature report table."""

        return self.per_feature.copy()


@dataclass(frozen=True)
class ReweightingReport:
    """Reporting wrapper around a fitted `ImportanceWeighter`."""

    summary_dict: dict[str, Any]
    weights: pd.Series

    @classmethod
    def from_weighter(cls, weighter: ImportanceWeighter) -> ReweightingReport:
        """Construct a report from a fitted importance weighter."""

        return cls(
            summary_dict=weighter.summary(),
            weights=pd.Series(weighter.reference_weights_, name="importance_weight"),
        )

    def to_markdown(self) -> str:
        """Render a concise markdown summary."""

        return (
            f"## Importance weighting summary\n\n"
            f"- Method: {self.summary_dict['method']}\n"
            f"- Mean weight: {self.summary_dict['mean_weight']:.3f}\n"
            f"- Effective sample size: {self.summary_dict['effective_sample_size']:.1f}\n"
            f"- Source AUC: {self.summary_dict['source_auc']:.3f}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary summary."""

        return {
            "summary": self.summary_dict,
            "weights": self.weights.tolist(),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return weights as a DataFrame."""

        return self.weights.to_frame()
