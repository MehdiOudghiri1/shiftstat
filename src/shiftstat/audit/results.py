"""Result containers for conditional reliability auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AuditReport:
    """Markdown and machine-readable report for deployment reliability auditing."""

    aggregate_summary: dict[str, Any]
    subgroup_report: dict[str, Any]
    conditional_error_table: list[dict[str, Any]]
    shift_severity_table: list[dict[str, Any]]
    disparity_summary: list[dict[str, Any]]
    aggregate_vs_subgroup_table: list[dict[str, Any]]
    heatmap_table: list[dict[str, Any]]
    discovered_slices: list[dict[str, Any]]
    hidden_failure_flags: dict[str, bool]
    caveats: list[str]
    operational_implications: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "aggregate_summary": self.aggregate_summary,
            "subgroup_report": self.subgroup_report,
            "conditional_error_table": self.conditional_error_table,
            "shift_severity_table": self.shift_severity_table,
            "disparity_summary": self.disparity_summary,
            "aggregate_vs_subgroup_table": self.aggregate_vs_subgroup_table,
            "heatmap_table": self.heatmap_table,
            "discovered_slices": self.discovered_slices,
            "hidden_failure_flags": self.hidden_failure_flags,
            "caveats": self.caveats,
            "operational_implications": self.operational_implications,
        }

    def to_markdown(self, *, top_k: int = 5) -> str:
        """Render a serious markdown audit report."""

        lines = [
            "## Deployment reliability audit",
            "",
            "### Aggregate reliability summary",
            "",
            f"- Reference accuracy: {self.aggregate_summary['reference_accuracy']:.3f}",
            f"- Target accuracy: {self.aggregate_summary['target_accuracy']:.3f}",
            f"- Delta accuracy: {self.aggregate_summary['delta_accuracy']:.3f}",
            f"- Reference ECE: {self.aggregate_summary['reference_ece']:.3f}",
            f"- Target ECE: {self.aggregate_summary['target_ece']:.3f}",
            f"- Delta ECE: {self.aggregate_summary['delta_ece']:.3f}",
            "",
            "### Subgroup degradation analysis",
            "",
        ]

        degradation = pd.DataFrame.from_records(
            self.subgroup_report.get("degradation_table", [])
        ).head(top_k)
        if degradation.empty:
            lines.append("- No supported subgroup degradations were identified.")
        else:
            for _, row in degradation.iterrows():
                lines.append(
                    (
                        f"- {row['slice_name']} -> {row['group']}: "
                        f"delta error {row['delta_error_rate']:.3f}, "
                        f"delta ECE {row['delta_ece']:.3f}, "
                        f"severity {row['severity_score']:.3f}"
                    )
                )

        lines.extend(["", "### Discovered failure slices", ""])
        slices = self.discovered_slice_frame().head(top_k)
        if slices.empty:
            lines.append("- No failure slices met the current discovery criteria.")
        else:
            for _, row in slices.iterrows():
                lines.append(
                    (
                        f"- {row['slice_label']}: error gap {row['delta_error_rate']:.3f}, "
                        f"failure share {row['target_failure_share']:.3f}, "
                        f"rule `{row['rule']}`"
                    )
                )

        lines.extend(["", "### Statistical caveats", ""])
        if self.caveats:
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
        else:
            lines.append("- No additional caveats recorded.")

        lines.extend(["", "### Operational implications", ""])
        if self.operational_implications:
            for implication in self.operational_implications:
                lines.append(f"- {implication}")
        else:
            lines.append("- No operational implications were generated.")
        return "\n".join(lines)

    def conditional_error_frame(self) -> pd.DataFrame:
        """Return confidence-conditioned error summaries."""

        return pd.DataFrame.from_records(self.conditional_error_table)  # type: ignore[no-any-return]

    def shift_severity_frame(self) -> pd.DataFrame:
        """Return performance metrics by shift-severity bin."""

        return pd.DataFrame.from_records(self.shift_severity_table)  # type: ignore[no-any-return]

    def disparity_frame(self) -> pd.DataFrame:
        """Return disparity summaries between supported subgroups."""

        return pd.DataFrame.from_records(self.disparity_summary)  # type: ignore[no-any-return]

    def aggregate_vs_subgroup_frame(self) -> pd.DataFrame:
        """Return aggregate versus worst-group comparison rows."""

        return pd.DataFrame.from_records(self.aggregate_vs_subgroup_table)  # type: ignore[no-any-return]

    def heatmap_frame(self) -> pd.DataFrame:
        """Return long-format subgroup x metric heatmap data."""

        return pd.DataFrame.from_records(self.heatmap_table)  # type: ignore[no-any-return]

    def discovered_slice_frame(self) -> pd.DataFrame:
        """Return discovered failure slices."""

        return pd.DataFrame.from_records(self.discovered_slices)  # type: ignore[no-any-return]

    def to_tables(self) -> dict[str, pd.DataFrame]:
        """Return exportable DataFrame tables."""

        return {
            "conditional_error": self.conditional_error_frame(),
            "shift_severity": self.shift_severity_frame(),
            "disparity": self.disparity_frame(),
            "aggregate_vs_subgroup": self.aggregate_vs_subgroup_frame(),
            "heatmap": self.heatmap_frame(),
            "discovered_slices": self.discovered_slice_frame(),
        }
