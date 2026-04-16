"""Structured results for config-driven experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shiftstat.bench.results import BenchmarkResult


@dataclass(frozen=True)
class ExperimentResult:
    """Result of one config-driven experiment run."""

    experiment_name: str
    config_path: str
    output_dir: str
    figure_format: str
    benchmark_results: list[BenchmarkResult]
    scenario_artifacts: list[dict[str, Any]]
    started_at_utc: str
    completed_at_utc: str
    shiftstat_version: str
    manifest_path: str | None = None
    summary_csv_path: str | None = None
    markdown_path: str | None = None
    log_path: str | None = None

    def summary_frame(self) -> pd.DataFrame:
        """Return the combined aggregate frame across benchmark scenarios."""

        frames = []
        for result in self.benchmark_results:
            frame = result.aggregate_frame().copy()
            frame["experiment_name"] = self.experiment_name
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable experiment manifest."""

        return {
            "experiment_name": self.experiment_name,
            "config_path": self.config_path,
            "output_dir": self.output_dir,
            "figure_format": self.figure_format,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "shiftstat_version": self.shiftstat_version,
            "scenario_artifacts": self.scenario_artifacts,
            "manifest_path": self.manifest_path,
            "summary_csv_path": self.summary_csv_path,
            "markdown_path": self.markdown_path,
            "log_path": self.log_path,
            "benchmarks": [result.to_dict() for result in self.benchmark_results],
        }

    def to_markdown(self) -> str:
        """Render a concise markdown experiment summary."""

        lines = [
            "## Experiment summary",
            "",
            f"- Experiment: {self.experiment_name}",
            f"- Config: `{self.config_path}`",
            f"- Output directory: `{self.output_dir}`",
            f"- Scenarios: {len(self.benchmark_results)}",
            "",
        ]
        frame = self.summary_frame()
        if frame.empty:
            lines.append("- No benchmark outputs were generated.")
            return "\n".join(lines)

        for result in self.benchmark_results:
            lines.extend(
                [
                    f"### {result.scenario_name}",
                    "",
                    result.to_markdown(),
                    "",
                ]
            )
        return "\n".join(lines)

    def rerun_command(self) -> str:
        """Return a reproducible rerun command."""

        return f"python -m shiftstat.experiments \"{self.config_path}\""

    def artifact_root(self) -> Path:
        """Return the artifact root as a Path object."""

        return Path(self.output_dir)
