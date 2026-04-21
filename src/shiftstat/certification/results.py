"""Result containers for certified worst-group audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd


class AuditDecision(str, Enum):
    """Decision assigned to an audited subgroup-bin cell."""

    CERTIFIED_FAILURE = "certified_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_DETECTED_FAILURE = "no_detected_failure"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class CertifiedCellResult:
    """Decision record for one subgroup-bin cell."""

    group: str
    bin_lower: float
    bin_upper: float
    source_count: int
    target_count: int | None
    estimated_target_mass: float
    weighted_gap: float
    abs_weighted_gap: float
    local_ess: float
    label_radius: float
    rho_sensitivity: float
    gamma_population: float
    total_radius: float
    label_certified_excess: float
    certified_excess: float
    naive_alarm: bool
    label_certified_alarm: bool
    certified_alarm: bool
    decision: AuditDecision
    reason: str
    target_gap: float | None = None
    abs_target_gap: float | None = None

    @property
    def bin_label(self) -> str:
        """Return a compact score-bin label."""

        return f"[{self.bin_lower:.2f}, {self.bin_upper:.2f}]"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row."""

        row = asdict(self)
        row["decision"] = self.decision.value
        row["bin"] = self.bin_label
        return row


@dataclass(frozen=True)
class CertifiedAuditReport:
    """Machine-readable and human-readable certified audit report."""

    cells: tuple[CertifiedCellResult, ...]
    tolerance: float
    alpha: float
    n_cells_tested: int
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        """Return all audited cells as a DataFrame sorted by apparent gap."""

        frame = pd.DataFrame.from_records([cell.to_dict() for cell in self.cells])
        if frame.empty:
            return frame
        return frame.sort_values(
            ["abs_weighted_gap", "certified_excess"],
            ascending=[False, False],
            ignore_index=True,
        )

    def top_cells(self, k: int = 10) -> pd.DataFrame:
        """Return the top apparent cells by absolute weighted gap."""

        return self.to_frame().head(k)

    def certified_failures(self) -> pd.DataFrame:
        """Return cells with positive target-certified excess."""

        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame.loc[frame["decision"] == AuditDecision.CERTIFIED_FAILURE.value].reset_index(
            drop=True
        )

    def insufficient_evidence(self) -> pd.DataFrame:
        """Return naive alarms that are not certified."""

        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame.loc[
            frame["decision"] == AuditDecision.INSUFFICIENT_EVIDENCE.value
        ].reset_index(drop=True)

    def out_of_scope(self) -> pd.DataFrame:
        """Return cells excluded from certification by support or mass filters."""

        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame.loc[frame["decision"] == AuditDecision.OUT_OF_SCOPE.value].reset_index(
            drop=True
        )

    def summary(self) -> dict[str, Any]:
        """Return compact audit-level counts."""

        frame = self.to_frame()
        if frame.empty:
            return {
                "n_cells": 0,
                "n_naive_alarms": 0,
                "n_certified_failures": 0,
                "n_insufficient_evidence": 0,
                "n_out_of_scope": 0,
            }
        return {
            "n_cells": int(len(frame)),
            "n_cells_tested": int(self.n_cells_tested),
            "n_naive_alarms": int(frame["naive_alarm"].sum()),
            "n_label_certified_alarms": int(frame["label_certified_alarm"].sum()),
            "n_certified_failures": int(frame["certified_alarm"].sum()),
            "n_insufficient_evidence": int(
                (frame["decision"] == AuditDecision.INSUFFICIENT_EVIDENCE.value).sum()
            ),
            "n_out_of_scope": int((frame["decision"] == AuditDecision.OUT_OF_SCOPE.value).sum()),
            "min_local_ess": float(frame["local_ess"].min()),
            "median_local_ess": float(frame["local_ess"].median()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "summary": self.summary(),
            "tolerance": self.tolerance,
            "alpha": self.alpha,
            "n_cells_tested": self.n_cells_tested,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def to_markdown(self, *, top_k: int = 8) -> str:
        """Render an operational audit summary."""

        summary = self.summary()
        lines = [
            "## Certified worst-group audit",
            "",
            "### Decision summary",
            "",
            f"- Tolerance: {self.tolerance:.3f}",
            f"- Error budget alpha: {self.alpha:.3f}",
            f"- Audited cells: {summary['n_cells']}",
            f"- Naive alarms: {summary['n_naive_alarms']}",
            f"- Certified failures: {summary['n_certified_failures']}",
            f"- Insufficient-evidence alarms: {summary['n_insufficient_evidence']}",
            "",
            "### Top apparent cells",
            "",
        ]
        top = self.top_cells(top_k)
        if top.empty:
            lines.append("- No cells were evaluated.")
        else:
            for _, row in top.iterrows():
                lines.append(
                    f"- {row['group']} {row['bin']}: gap={row['abs_weighted_gap']:.3f}, "
                    f"local ESS={row['local_ess']:.2f}, radius={row['total_radius']:.3f}, "
                    f"decision={row['decision']}"
                )
        if self.warnings:
            lines.extend(["", "### Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.assumptions:
            lines.extend(["", "### Assumptions", ""])
            lines.extend(f"- {assumption}" for assumption in self.assumptions)
        return "\n".join(lines)
