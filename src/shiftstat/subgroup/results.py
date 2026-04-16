"""Result containers for subgroup-aware reliability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SubgroupReport:
    """Markdown and structured report for subgroup reliability diagnostics."""

    aggregate_summary: dict[str, Any]
    performance_table: list[dict[str, Any]]
    calibration_table: list[dict[str, Any]]
    shift_exposure_table: list[dict[str, Any]]
    degradation_table: list[dict[str, Any]]
    stability_table: list[dict[str, Any]]
    worst_group_summary: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "aggregate_summary": self.aggregate_summary,
            "performance_table": self.performance_table,
            "calibration_table": self.calibration_table,
            "shift_exposure_table": self.shift_exposure_table,
            "degradation_table": self.degradation_table,
            "stability_table": self.stability_table,
            "worst_group_summary": self.worst_group_summary,
            "warnings": self.warnings,
        }

    def to_markdown(self, *, top_k: int = 5) -> str:
        """Render a concise markdown summary."""

        lines = ["## Subgroup reliability analysis", ""]
        lines.extend(
            [
                f"- Grouping specifications audited: {self.aggregate_summary['n_groupings']}",
                f"- Degradation rows analyzed: {self.aggregate_summary['n_degradation_rows']}",
                (
                    "- Target sample coverage above support thresholds: "
                    f"{self.aggregate_summary['supported_target_coverage']:.3f}"
                ),
                (
                    "- Flagged subgroup rows with support caveats: "
                    f"{self.aggregate_summary['flagged_group_rows']}"
                ),
                "",
                "### Worst-group summary",
                "",
                (
                    "- Worst target accuracy slice: "
                    f"{self.worst_group_summary.get('worst_accuracy_group', 'n/a')}"
                ),
                (
                    "- Highest target calibration error slice: "
                    f"{self.worst_group_summary.get('worst_ece_group', 'n/a')}"
                ),
                (
                    "- Most severe degradation slice: "
                    f"{self.worst_group_summary.get('highest_severity_group', 'n/a')}"
                ),
                "",
                "### Highest-severity subgroup degradations",
                "",
            ]
        )

        degradation_frame = self.degradation_frame().head(top_k)
        if degradation_frame.empty:
            lines.append("- No subgroup degradation rows available.")
        else:
            for _, row in degradation_frame.iterrows():
                lines.append(
                    (
                        f"- {row['slice_name']} -> {row['group']}: "
                        f"delta error {row['delta_error_rate']:.3f}, "
                        f"delta ECE {row['delta_ece']:.3f}, "
                        f"severity {row['severity_score']:.3f}"
                    )
                )

        if self.warnings:
            lines.extend(["", "### Caveats", ""])
            for warning in self.warnings:
                lines.append(f"- {warning}")
        return "\n".join(lines)

    def performance_frame(self) -> pd.DataFrame:
        """Return subgroup performance metrics as a DataFrame."""

        return pd.DataFrame.from_records(self.performance_table)  # type: ignore[no-any-return]

    def calibration_frame(self) -> pd.DataFrame:
        """Return subgroup calibration metrics as a DataFrame."""

        return pd.DataFrame.from_records(self.calibration_table)  # type: ignore[no-any-return]

    def shift_exposure_frame(self) -> pd.DataFrame:
        """Return subgroup exposure shifts as a DataFrame."""

        return pd.DataFrame.from_records(self.shift_exposure_table)  # type: ignore[no-any-return]

    def degradation_frame(self) -> pd.DataFrame:
        """Return subgroup degradation metrics as a DataFrame."""

        return pd.DataFrame.from_records(self.degradation_table)  # type: ignore[no-any-return]

    def stability_frame(self) -> pd.DataFrame:
        """Return subgroup stability diagnostics as a DataFrame."""

        return pd.DataFrame.from_records(self.stability_table)  # type: ignore[no-any-return]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return exportable tabular views."""

        return {
            "performance": self.performance_frame(),
            "calibration": self.calibration_frame(),
            "shift_exposure": self.shift_exposure_frame(),
            "degradation": self.degradation_frame(),
            "stability": self.stability_frame(),
            "worst_group_summary": pd.DataFrame.from_records([self.worst_group_summary]),
        }
