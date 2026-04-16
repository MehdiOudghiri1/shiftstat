"""Result containers for calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration diagnostics for a set of binary predictions."""

    name: str
    n_samples: int
    weighted: bool
    ece: float
    mce: float
    brier_score: float
    negative_log_likelihood: float
    calibration_intercept: float
    calibration_slope: float
    mean_confidence: float
    mean_outcome: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    bin_summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "weighted": self.weighted,
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "negative_log_likelihood": self.negative_log_likelihood,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "mean_confidence": self.mean_confidence,
            "mean_outcome": self.mean_outcome,
            "brier_reliability": self.brier_reliability,
            "brier_resolution": self.brier_resolution,
            "brier_uncertainty": self.brier_uncertainty,
            "bin_summary": self.bin_summary,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return bin-wise calibration diagnostics as a DataFrame."""

        return pd.DataFrame.from_records(self.bin_summary)

