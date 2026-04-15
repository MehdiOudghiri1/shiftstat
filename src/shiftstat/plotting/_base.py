"""Shared plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def finalize_figure(
    ax: Axes,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Axes:
    """Apply common figure finalization steps."""

    figure = cast(Figure, ax.get_figure())
    if title is not None:
        ax.set_title(title)
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax
