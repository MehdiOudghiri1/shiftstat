"""Plots for selective prediction and abstention analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from shiftstat.plotting._base import finalize_figure
from shiftstat.plotting.calibration import plot_reliability_diagram


def _as_frame(data: Any, method_name: str | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()  # type: ignore[no-any-return]
    if method_name is not None and hasattr(data, method_name):
        return getattr(data, method_name)()  # type: ignore[no-any-return]
    if hasattr(data, "to_frame"):
        return data.to_frame()  # type: ignore[no-any-return]
    raise TypeError("Unsupported plotting input.")


def plot_risk_coverage_curve(
    data: Any,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot selective risk against retained coverage."""

    frame = _as_frame(data, "risk_coverage_frame")
    axis = ax or plt.subplots(figsize=(6, 4))[1]
    axis.plot(frame["coverage"], frame["selective_risk"], marker="o", color="#005f73")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    return finalize_figure(axis, title="Risk-coverage curve", save_path=save_path, show=show)


def plot_coverage_vs_threshold(
    data: Any,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot retained coverage as a function of threshold."""

    frame = _as_frame(data, "risk_coverage_frame")
    axis = ax or plt.subplots(figsize=(6, 4))[1]
    axis.plot(frame["threshold"], frame["coverage"], marker="o", color="#bb3e03")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Coverage")
    return finalize_figure(axis, title="Coverage vs threshold", save_path=save_path, show=show)


def plot_abstention_distribution(
    profile: Any,
    *,
    which: str = "score",
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot accepted versus rejected score or confidence distributions."""

    frame = (
        profile.score_distribution_frame()
        if which == "score"
        else profile.confidence_distribution_frame()
    )
    axis = ax or plt.subplots(figsize=(6, 4))[1]
    for status, group in frame.groupby("status"):
        centers = 0.5 * (group["lower"] + group["upper"])
        axis.plot(centers, group["share"], marker="o", label=status)
    axis.set_xlabel("Selection score" if which == "score" else "Confidence")
    axis.set_ylabel("Share")
    axis.legend(loc="best")
    title = (
        "Abstention score distribution" if which == "score" else "Accepted vs rejected confidence"
    )
    return finalize_figure(axis, title=title, save_path=save_path, show=show)


def plot_subgroup_abstention_comparison(
    data: Any,
    *,
    metric: str = "target_abstention_rate",
    top_k: int = 10,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot subgroup abstention or coverage comparisons."""

    frame = _as_frame(data, "subgroup_abstention_frame").head(top_k)
    labels = [f"{row.slice_name}\n{row.group}" for row in frame.itertuples()]
    axis = ax or plt.subplots(figsize=(max(7, top_k), 4))[1]
    axis.bar(labels, frame[metric], color="#ca6702")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=45)
    return finalize_figure(
        axis,
        title="Subgroup abstention comparison",
        save_path=save_path,
        show=show,
    )


def plot_confidence_accept_reject_distribution(
    profile: Any,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Alias for the confidence distribution view."""

    return plot_abstention_distribution(
        profile,
        which="confidence",
        ax=ax,
        show=show,
        save_path=save_path,
    )


def plot_selective_reliability_diagram(
    profile: Any,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a reliability diagram on the accepted subset when available."""

    if profile.accepted_reliability_profile is None:
        raise ValueError("Selective profile does not contain an accepted-set reliability profile.")
    return plot_reliability_diagram(
        profile.accepted_reliability_profile,
        ax=ax,
        show=show,
        save_path=save_path,
    )
