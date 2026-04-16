"""Calibration and reliability plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from shiftstat.calibration.results import CalibrationResult
from shiftstat.plotting._base import finalize_figure
from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
)

if TYPE_CHECKING:
    from shiftstat.reliability.results import ReliabilityProfile


def _curve_frame(result: CalibrationResult | ReliabilityProfile) -> pd.DataFrame:
    if isinstance(result, CalibrationResult):
        return result.to_frame()
    return result.calibration_frame()


def plot_reliability_diagram(
    result: CalibrationResult | ReliabilityProfile,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a single reliability diagram."""

    curve = _curve_frame(result)
    axis = ax or plt.subplots(figsize=(5, 5))[1]
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axis.plot(
        curve["mean_confidence"],
        curve["empirical_probability"],
        marker="o",
        color="#005f73",
        label=label or getattr(result, "name", "dataset"),
    )
    if label is not None or hasattr(result, "name"):
        axis.legend(loc="upper left")
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Empirical frequency")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    return finalize_figure(axis, title="Reliability diagram", save_path=save_path, show=show)


def plot_calibration_comparison(
    reference: CalibrationResult | ReliabilityProfile,
    target: CalibrationResult | ReliabilityProfile,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot side-by-side reference and target calibration curves."""

    axis = ax or plt.subplots(figsize=(5, 5))[1]
    plot_reliability_diagram(reference, ax=axis, label=getattr(reference, "name", "reference"))
    target_curve = _curve_frame(target)
    axis.plot(
        target_curve["mean_confidence"],
        target_curve["empirical_probability"],
        marker="s",
        color="#9b2226",
        label=getattr(target, "name", "target"),
    )
    axis.legend(loc="upper left")
    return finalize_figure(
        axis,
        title="Reference vs target calibration",
        save_path=save_path,
        show=show,
    )


def plot_weighted_unweighted_calibration(
    unweighted: CalibrationResult | ReliabilityProfile,
    weighted: CalibrationResult | ReliabilityProfile,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Overlay weighted and unweighted calibration curves."""

    axis = ax or plt.subplots(figsize=(5, 5))[1]
    plot_reliability_diagram(unweighted, ax=axis, label=getattr(unweighted, "name", "unweighted"))
    weighted_curve = _curve_frame(weighted)
    axis.plot(
        weighted_curve["mean_confidence"],
        weighted_curve["empirical_probability"],
        marker="^",
        color="#ca6702",
        label=getattr(weighted, "name", "weighted"),
    )
    axis.legend(loc="upper left")
    return finalize_figure(
        axis,
        title="Weighted vs unweighted calibration",
        save_path=save_path,
        show=show,
    )


def plot_recalibration_comparison(
    before: CalibrationResult | ReliabilityProfile,
    after: CalibrationResult | ReliabilityProfile,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Overlay pre- and post-recalibration reliability diagrams."""

    axis = ax or plt.subplots(figsize=(5, 5))[1]
    plot_reliability_diagram(before, ax=axis, label=getattr(before, "name", "before"))
    after_curve = _curve_frame(after)
    axis.plot(
        after_curve["mean_confidence"],
        after_curve["empirical_probability"],
        marker="D",
        color="#2a9d8f",
        label=getattr(after, "name", "after"),
    )
    axis.legend(loc="upper left")
    return finalize_figure(axis, title="Before/after recalibration", save_path=save_path, show=show)


def plot_confidence_histogram(
    probabilities: np.ndarray,
    *,
    bins: int = 20,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a histogram of predictive confidence."""

    axis = ax or plt.subplots(figsize=(6, 4))[1]
    confidence = confidence_from_probabilities(extract_positive_class_probabilities(probabilities))
    axis.hist(confidence, bins=bins, color="#94d2bd", edgecolor="white")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Frequency")
    return finalize_figure(axis, title="Confidence histogram", save_path=save_path, show=show)
