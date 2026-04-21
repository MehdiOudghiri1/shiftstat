"""Render the main local-ESS figure for the preprint."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORACLE_CSV = ROOT / "research" / "results" / "local_ess_oracle" / "local_ess_summary.csv"
LOGISTIC_CSV = ROOT / "research" / "results" / "local_ess_logistic" / "local_ess_summary.csv"
OUTPUT = ROOT / "figures" / "local_ess_main.png"


def main() -> None:
    oracle = pd.read_csv(ORACLE_CSV)
    logistic = pd.read_csv(LOGISTIC_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    ax.plot(
        oracle["shift_strength"],
        oracle["median_global_ess"],
        marker="o",
        linewidth=2.2,
        color="#0B6E4F",
        label="Median global ESS",
    )
    ax.plot(
        oracle["shift_strength"],
        oracle["median_q10_local_ess"],
        marker="s",
        linewidth=2.2,
        color="#C84C09",
        label="Median lower-tail local ESS",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Shift severity")
    ax.set_ylabel("Effective sample size (log scale)")
    ax.set_title("Global vs local weighted support")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(
        oracle["shift_strength"],
        oracle["false_alarm_rate_005"],
        marker="o",
        linewidth=2.2,
        color="#7B2CBF",
        label="Oracle weights",
    )
    ax.plot(
        logistic["shift_strength"],
        logistic["false_alarm_rate_005"],
        marker="D",
        linewidth=2.0,
        color="#D00000",
        label="Logistic weights",
    )
    ax.set_xlabel("Shift severity")
    ax.set_ylabel(r"False-alarm rate for $\widehat{\mathrm{WCE}} > 0.05$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("False subgroup alarms under pure covariate shift")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
