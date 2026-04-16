"""Benchmark plotting helpers for publication-oriented summary figures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark_metric_sweep(
    summary_frame: pd.DataFrame,
    *,
    metric: str,
    x_axis_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.5, 4.25),
) -> tuple[Any, Any]:
    """Plot one aggregated benchmark metric across cases and baselines."""

    if metric not in summary_frame.columns:
        raise ValueError(f"Metric {metric!r} is not present in the benchmark summary frame.")

    figure, axis = plt.subplots(figsize=figsize)
    plot_frame = summary_frame.copy()
    if "case_value" in plot_frame.columns and pd.api.types.is_numeric_dtype(
        plot_frame["case_value"]
    ):
        for baseline, group in plot_frame.groupby("baseline"):
            ordered = group.sort_values("case_value")
            axis.plot(
                ordered["case_value"],
                ordered[metric],
                marker="o",
                label=str(baseline),
            )
            std_column = f"{metric}_std"
            if std_column in ordered.columns:
                lower = ordered[metric] - ordered[std_column].fillna(0.0)
                upper = ordered[metric] + ordered[std_column].fillna(0.0)
                axis.fill_between(
                    ordered["case_value"],
                    lower,
                    upper,
                    alpha=0.12,
                )
        axis.set_xlabel(x_axis_label or "Case value")
    else:
        pivot = plot_frame.pivot_table(
            index="case_label",
            columns="baseline",
            values=metric,
            aggfunc="mean",
        )
        pivot.plot(kind="bar", ax=axis, rot=20)
        axis.set_xlabel(x_axis_label or "Case")

    axis.set_ylabel(metric.replace("_", " "))
    axis.set_title(title or metric.replace("_", " ").title())
    axis.legend(loc="best", title="Baseline")
    figure.tight_layout()
    return figure, axis
