"""Plots for confidence-conditioned reliability diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from shiftstat.plotting._base import finalize_figure

if TYPE_CHECKING:
    from shiftstat.reliability.results import ReliabilityProfile


def plot_confidence_error_curve(
    profile: ReliabilityProfile,
    *,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Plot error rate against mean confidence."""

    curve = profile.confidence_frame()
    axis = ax or plt.subplots(figsize=(6, 4))[1]
    axis.plot(curve["mean_confidence"], curve["error_rate"], marker="o", color="#9b2226")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Error rate")
    return finalize_figure(
        axis,
        title="Confidence-conditioned error",
        save_path=save_path,
        show=show,
    )
