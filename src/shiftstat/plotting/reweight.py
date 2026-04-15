"""Plots for importance-weight diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from shiftstat.plotting._base import finalize_figure


def plot_importance_weight_histogram(
    weights: np.ndarray,
    *,
    bins: int = 30,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a histogram of estimated importance weights."""

    axis = ax or plt.subplots(figsize=(7, 4))[1]
    axis.hist(weights, bins=bins, color="#0a9396", edgecolor="white")
    axis.set_xlabel("Importance weight")
    axis.set_ylabel("Frequency")
    return finalize_figure(
        axis,
        title="Importance weight distribution",
        save_path=save_path,
        show=show,
    )


def plot_effective_sample_size(
    *,
    original_sample_size: int,
    effective_sample_size: float,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a compact diagnostic comparing nominal and effective sample size."""

    axis = ax or plt.subplots(figsize=(5, 4))[1]
    labels = ["Nominal n", "Effective n"]
    values = [float(original_sample_size), float(effective_sample_size)]
    axis.bar(labels, values, color=["#94d2bd", "#ee9b00"])
    axis.set_ylabel("Sample size")
    return finalize_figure(axis, title="Effective sample size", save_path=save_path, show=show)
