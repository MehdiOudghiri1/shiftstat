"""Render the certified-audit figure for the preprint."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results" / "certified_audit"
SUMMARY_CSV = RESULTS / "certified_audit_summary.csv"
DECISIONS_CSV = RESULTS / "certified_audit_decisions.csv"
CELLS_CSV = RESULTS / "certified_audit_cells.csv"
OUTPUT = ROOT / "figures" / "certified_audit_main.png"

METHOD_LABELS = {
    "naive_max": "Naive worst cell",
    "global_ess_gate": "Global ESS gate",
    "local_ess_gate": "Local ESS gate",
    "simultaneous_ci": "Simultaneous CI",
}

COLORS = {
    "naive_max": "#263238",
    "global_ess_gate": "#2F6B9A",
    "local_ess_gate": "#A95C20",
    "simultaneous_ci": "#2E7D59",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D2C4", linewidth=0.8, alpha=0.55)
    ax.tick_params(labelsize=9, color="#4B4B4B")


def _plot_method_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    scenario: str,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    sub = frame.loc[frame["scenario"] == scenario]
    for method, method_frame in sub.groupby("method"):
        method_frame = method_frame.sort_values("shift_strength")
        ax.plot(
            method_frame["shift_strength"],
            method_frame[metric],
            marker="o",
            markersize=4.8,
            linewidth=2.1,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Shift severity", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.03, 1.03)
    _style_axis(ax)


def _plot_selected_support(ax: plt.Axes, decisions: pd.DataFrame) -> None:
    null = decisions.loc[
        (decisions["scenario"] == "null") & (decisions["method"] == "naive_max")
    ].copy()
    positions: list[int] = []
    labels: list[str] = []
    data: list[pd.Series] = []
    colors: list[str] = []
    for position, shift in enumerate(sorted(null["shift_strength"].unique()), start=1):
        values = null.loc[null["shift_strength"] == shift, "selected_local_ess"].dropna()
        if values.empty:
            continue
        positions.append(position)
        labels.append(f"{shift:g}")
        data.append(values)
        colors.append(COLORS["naive_max"])

    if not data:
        ax.text(
            0.5,
            0.5,
            "No selected cells available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#4B4B4B",
        )
        ax.set_axis_off()
        return

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#191919", "linewidth": 1.2},
        boxprops={"linewidth": 0.9, "color": "#4B4B4B"},
        whiskerprops={"linewidth": 0.9, "color": "#4B4B4B"},
        capprops={"linewidth": 0.9, "color": "#4B4B4B"},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
    ax.axhline(
        20.0,
        color=COLORS["local_ess_gate"],
        linestyle="--",
        linewidth=1.7,
        label="Local ESS gate",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Selected cell local ESS", fontsize=10)
    ax.set_xlabel("Shift severity", fontsize=10)
    ax.set_title(
        "C. The naive alarm often selects weak cells", loc="left", fontsize=11, fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style_axis(ax)


def _plot_decision_geometry(ax: plt.Axes, cells: pd.DataFrame) -> None:
    sub = cells.loc[
        (cells["scenario"] == "null") & (cells["shift_strength"].isin([0.8, 1.2]))
    ].copy()
    if len(sub) > 2500:
        sub = sub.sample(2500, random_state=7)

    ax.scatter(
        sub["local_ess"],
        sub["abs_gap"],
        s=12,
        alpha=0.18,
        color="#425466",
        linewidths=0,
        label="Null subgroup-bin estimates",
    )
    x_grid = np.geomspace(
        max(1.0, float(sub["local_ess"].min())), float(sub["local_ess"].max()), 250
    )
    tolerance = 0.20
    z_value = 3.0
    ax.plot(
        x_grid,
        np.full_like(x_grid, tolerance),
        color="#B3261E",
        linestyle="--",
        linewidth=1.8,
        label="Naive tolerance",
    )
    ax.plot(
        x_grid,
        tolerance + z_value * 0.5 / np.sqrt(x_grid),
        color="#2E7D59",
        linewidth=2.2,
        label="Certification frontier",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, min(1.0, max(0.55, float(sub["abs_gap"].quantile(0.995)) + 0.05)))
    ax.set_xlabel("Local ESS of subgroup-bin cell", fontsize=10)
    ax.set_ylabel("Estimated absolute gap", fontsize=10)
    ax.set_title(
        "D. Certification needs effect size and overlap", loc="left", fontsize=11, fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style_axis(ax)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV, keep_default_na=False)
    decisions = pd.read_csv(DECISIONS_CSV, keep_default_na=False)
    cells = pd.read_csv(CELLS_CSV, keep_default_na=False)
    for frame in [summary, decisions, cells]:
        for column in frame.columns:
            if column not in {
                "scenario",
                "method",
                "group",
                "bin",
                "selected_group",
                "selected_bin",
            }:
                frame[column] = pd.to_numeric(frame[column], errors="ignore")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#FCFAF5",
            "figure.facecolor": "#FCFAF5",
            "savefig.facecolor": "#FCFAF5",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    _plot_method_lines(
        axes[0, 0],
        summary,
        scenario="null",
        metric="alarm_rate",
        ylabel="False-alarm rate",
        title="A. Under the calibrated null, point audits over-alarm",
    )
    _plot_method_lines(
        axes[0, 1],
        summary,
        scenario="planted",
        metric="alarm_rate",
        ylabel="Detection power",
        title="B. Certified alarms trade power for validity",
    )
    _plot_selected_support(axes[1, 0], decisions)
    _plot_decision_geometry(axes[1, 1], cells)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.suptitle(
        "Worst-group alarms need simultaneous inference, not only reweighting",
        y=1.06,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=260, bbox_inches="tight")


if __name__ == "__main__":
    main()
