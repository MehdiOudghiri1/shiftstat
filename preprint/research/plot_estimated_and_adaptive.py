"""Render estimated-weight and adaptive-search summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESTIMATED = ROOT / "research" / "results" / "estimated_weights" / "estimated_weight_summary.csv"
ADAPTIVE = ROOT / "research" / "results" / "adaptive_search" / "adaptive_search_summary.csv"
OUTPUT = ROOT / "figures" / "estimated_weight_and_search.png"

WEIGHT_LABELS = {
    "oracle": "Oracle",
    "logistic_plugin": "Plugin logistic",
    "logistic_crossfit": "Cross-fit logistic",
}

METHOD_LABELS = {
    "naive_max": "Naive",
    "label_ci": "Label CI",
    "nuisance_ci": "Label+nuisance CI",
    "same_data_search": "Same-data search",
    "split_search_naive": "Split search, naive test",
    "split_search_certified": "Split search, certified",
    "full_family_simultaneous": "Full-family simultaneous",
}

COLORS = {
    "oracle": "#263238",
    "logistic_plugin": "#2F6B9A",
    "logistic_crossfit": "#A95C20",
    "naive_max": "#263238",
    "label_ci": "#2E7D59",
    "nuisance_ci": "#A95C20",
    "same_data_search": "#263238",
    "split_search_naive": "#2F6B9A",
    "split_search_certified": "#2E7D59",
    "full_family_simultaneous": "#7B6D8D",
}


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D2C4", alpha=0.58)
    ax.tick_params(labelsize=9)


def _grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    hue_column: str,
    hue_values: list[str],
    title: str,
    ylabel: str,
) -> None:
    categories = sorted(frame[category_column].unique())
    x = np.arange(len(categories), dtype=float)
    width = 0.75 / len(hue_values)
    for offset, hue in enumerate(hue_values):
        values = []
        for category in categories:
            match = frame.loc[
                (frame[category_column] == category) & (frame[hue_column] == hue),
                value_column,
            ]
            values.append(float(match.iloc[0]) if not match.empty else np.nan)
        ax.bar(
            x + (offset - (len(hue_values) - 1) / 2) * width,
            values,
            width=width,
            color=COLORS[hue],
            alpha=0.82,
            label=METHOD_LABELS.get(hue, WEIGHT_LABELS.get(hue, hue)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{category:g}" for category in categories])
    ax.set_xlabel("Shift severity", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    _style(ax)


def main() -> None:
    estimated = pd.read_csv(ESTIMATED, keep_default_na=False)
    adaptive = pd.read_csv(ADAPTIVE, keep_default_na=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#FCFAF5",
            "figure.facecolor": "#FCFAF5",
            "savefig.facecolor": "#FCFAF5",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    weight_quality = (
        estimated.loc[(estimated["scenario"] == "null") & (estimated["method"] == "label_ci")]
        .drop_duplicates(["weight_mode", "shift_strength"])
        .copy()
    )
    _grouped_bars(
        axes[0, 0],
        weight_quality,
        category_column="shift_strength",
        value_column="median_weight_rmse",
        hue_column="weight_mode",
        hue_values=["oracle", "logistic_plugin", "logistic_crossfit"],
        title="A. Ratio error",
        ylabel="Median weight RMSE",
    )

    null_estimated = estimated.loc[
        (estimated["scenario"] == "null")
        & (estimated["weight_mode"] == "logistic_crossfit")
        & (estimated["method"].isin(["naive_max", "label_ci", "nuisance_ci"]))
    ].copy()
    _grouped_bars(
        axes[0, 1],
        null_estimated,
        category_column="shift_strength",
        value_column="alarm_rate",
        hue_column="method",
        hue_values=["naive_max", "label_ci", "nuisance_ci"],
        title="B. Weight uncertainty",
        ylabel="Null false-alarm rate",
    )
    axes[0, 1].set_ylim(0.0, 1.05)

    null_adaptive = adaptive.loc[
        (adaptive["scenario"] == "null")
        & (
            adaptive["method"].isin(
                [
                    "same_data_search",
                    "split_search_naive",
                    "split_search_certified",
                    "full_family_simultaneous",
                ]
            )
        )
    ].copy()
    _grouped_bars(
        axes[1, 0],
        null_adaptive,
        category_column="shift_strength",
        value_column="alarm_rate",
        hue_column="method",
        hue_values=[
            "same_data_search",
            "split_search_naive",
            "split_search_certified",
            "full_family_simultaneous",
        ],
        title="C. Split search",
        ylabel="Null false-alarm rate",
    )
    axes[1, 0].set_ylim(0.0, 1.05)

    planted_adaptive = adaptive.loc[
        (adaptive["scenario"] == "planted")
        & (
            adaptive["method"].isin(
                [
                    "same_data_search",
                    "split_search_naive",
                    "split_search_certified",
                    "full_family_simultaneous",
                ]
            )
        )
    ].copy()
    _grouped_bars(
        axes[1, 1],
        planted_adaptive,
        category_column="shift_strength",
        value_column="alarm_rate",
        hue_column="method",
        hue_values=[
            "same_data_search",
            "split_search_naive",
            "split_search_certified",
            "full_family_simultaneous",
        ],
        title="D. Search power",
        ylabel="Planted-failure detection rate",
    )
    axes[1, 1].set_ylim(0.0, 1.05)

    for ax in axes.ravel():
        ax.legend(frameon=False, fontsize=7.4, loc="upper right")

    fig.suptitle(
        "Estimated weights and search",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=260, bbox_inches="tight")


if __name__ == "__main__":
    main()
