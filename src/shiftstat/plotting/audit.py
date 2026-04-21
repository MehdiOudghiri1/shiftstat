"""Plots for subgroup reliability auditing and failure localization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ._base import finalize_figure


def _as_frame(data: Any, method_name: str | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()  # type: ignore[no-any-return]
    if method_name is not None and hasattr(data, method_name):
        return getattr(data, method_name)()  # type: ignore[no-any-return]
    if hasattr(data, "to_frame"):
        return data.to_frame()  # type: ignore[no-any-return]
    raise TypeError("Unsupported plotting input.")


def plot_subgroup_degradation(
    data: Any,
    *,
    metric: str = "delta_error_rate",
    top_k: int = 10,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot subgroup degradation bars ranked by severity."""

    frame = _as_frame(data, "degradation_ranking").head(top_k)
    labels = [f"{row.slice_name}\n{row.group}" for row in frame.itertuples()]
    axis = ax or plt.subplots(figsize=(max(7, top_k), 4))[1]
    axis.bar(labels, frame[metric], color="#bb3e03")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.tick_params(axis="x", rotation=45)
    return finalize_figure(
        axis,
        title="Subgroup degradation",
        save_path=save_path,
        show=show,
    )


def plot_worst_group_comparison(
    data: Any,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot aggregate versus worst-group comparisons for key metrics."""

    frame = _as_frame(data, "aggregate_vs_subgroup_frame")
    axis = ax or plt.subplots(figsize=(7, 4))[1]
    x = np.arange(len(frame))
    width = 0.35
    axis.bar(x - width / 2, frame["aggregate_target_value"], width=width, label="aggregate")
    axis.bar(
        x + width / 2,
        frame["worst_group_target_value"],
        width=width,
        label="worst group",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(frame["metric"], rotation=20)
    axis.set_ylabel("Metric value")
    axis.legend(loc="best")
    return finalize_figure(
        axis,
        title="Aggregate vs worst-group metrics",
        save_path=save_path,
        show=show,
    )


def plot_subgroup_metric_heatmap(
    data: Any,
    *,
    value_column: str = "delta_value",
    top_k: int = 15,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a subgroup x metric heatmap."""

    frame = _as_frame(data, "heatmap_frame")
    pivot_frame = frame.copy()
    pivot_frame["label"] = pivot_frame["slice_name"] + "\n" + pivot_frame["group"]
    pivot = (
        pivot_frame.sort_values("severity_score", ascending=False)
        .head(top_k)
        .pivot(index="label", columns="metric", values=value_column)
        .fillna(0.0)
    )
    axis = ax or plt.subplots(figsize=(7, max(4, 0.4 * len(pivot))))[1]
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    plt.colorbar(image, ax=axis, shrink=0.8)
    return finalize_figure(
        axis,
        title="Subgroup metric heatmap",
        save_path=save_path,
        show=show,
    )


def plot_discovered_slice_summary(
    data: Any,
    *,
    metric: str = "delta_error_rate",
    top_k: int = 6,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot discovered failure slices and their degradation."""

    frame = _as_frame(data, "discovered_slice_frame").head(top_k)
    axis = ax or plt.subplots(figsize=(7, 4))[1]
    axis.bar(frame["slice_label"], frame[metric], color="#9b2226")
    axis.set_xlabel("Slice")
    axis.set_ylabel(metric.replace("_", " ").title())
    return finalize_figure(
        axis,
        title="Discovered failure slices",
        save_path=save_path,
        show=show,
    )


def plot_failure_concentration(
    data: Any,
    *,
    top_k: int = 8,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot cumulative failure share against cumulative sample share."""

    frame = (
        _as_frame(data, "discovered_slice_frame")
        .sort_values(
            "target_failure_share",
            ascending=False,
        )
        .head(top_k)
    )
    cumulative_failure = frame["target_failure_share"].cumsum()
    cumulative_sample = frame["target_sample_share"].cumsum()
    axis = ax or plt.subplots(figsize=(6, 4))[1]
    axis.plot(cumulative_sample, cumulative_failure, marker="o", color="#005f73")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axis.set_xlabel("Cumulative sample share")
    axis.set_ylabel("Cumulative failure share")
    return finalize_figure(
        axis,
        title="Failure concentration",
        save_path=save_path,
        show=show,
    )


def plot_aggregate_vs_subgroup(
    data: Any,
    *,
    metric: str = "absolute_gap",
    top_k: int = 10,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot aggregate-versus-subgroup gap sizes."""

    frame = _as_frame(data, "aggregate_vs_subgroup_frame").head(top_k)
    axis = ax or plt.subplots(figsize=(7, 4))[1]
    axis.bar(frame["metric"], frame[metric], color="#ca6702")
    axis.set_ylabel(metric.replace("_", " ").title())
    return finalize_figure(
        axis,
        title="Aggregate vs subgroup gaps",
        save_path=save_path,
        show=show,
    )
