"""Plots for shift detection outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from shiftstat.detect.results import ClassifierShiftResult
from shiftstat.plotting._base import finalize_figure


def plot_feature_drift(
    summary: pd.DataFrame,
    *,
    top_n: int = 15,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot feature severity scores as a horizontal bar chart."""

    axis = ax or plt.subplots(figsize=(8, 4))[1]
    display = summary.nlargest(top_n, "severity_score").sort_values("severity_score")
    axis.barh(display["feature_name"], display["severity_score"], color="#335c67")
    axis.set_xlabel("Severity score")
    axis.set_ylabel("Feature")
    return finalize_figure(axis, title="Feature drift severity", save_path=save_path, show=show)


def plot_shift_severity_heatmap(
    summary: pd.DataFrame,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot a compact heatmap of per-feature drift metrics."""

    axis = ax or plt.subplots(figsize=(8, 4))[1]
    matrix = summary[["severity_score", "psi", "adjusted_p_value"]].copy()
    matrix["adjusted_p_value"] = -np.log10(np.clip(matrix["adjusted_p_value"], 1e-12, 1.0))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    axis.set_yticks(range(len(summary)))
    axis.set_yticklabels(summary["feature_name"])
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(["severity", "psi", "-log10(adj p)"])
    axis.figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    return finalize_figure(axis, title="Shift severity heatmap", save_path=save_path, show=show)


def plot_source_discrimination_roc(
    result: ClassifierShiftResult,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot ROC diagnostics for a classifier two-sample test."""

    axis = ax or plt.subplots(figsize=(5, 5))[1]
    axis.plot(result.fpr, result.tpr, label=f"AUC = {result.auc:.3f}", color="#9e2a2b")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    return finalize_figure(axis, title="Source discrimination ROC", save_path=save_path, show=show)

