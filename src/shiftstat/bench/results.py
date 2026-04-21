"""Structured benchmark results and publication-oriented exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from shiftstat.plotting.benchmark import plot_benchmark_metric_sweep
from shiftstat.utils.artifacts import artifact_record, portable_path


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured output of a repeated-seed benchmark scenario."""

    scenario_name: str
    family: str
    description: str
    seeds: list[int]
    baseline_names: list[str]
    publication_metrics: list[str]
    scenario_config: dict[str, Any]
    metric_schema: dict[str, dict[str, Any]]
    run_records: list[dict[str, Any]]
    aggregated_records: list[dict[str, Any]]
    created_at_utc: str
    x_axis_label: str | None = None

    def run_frame(self) -> pd.DataFrame:
        """Return the per-seed benchmark records."""

        return pd.DataFrame.from_records(self.run_records)  # type: ignore[no-any-return]

    def aggregate_frame(self) -> pd.DataFrame:
        """Return seed-aggregated benchmark summaries."""

        return pd.DataFrame.from_records(self.aggregated_records)  # type: ignore[no-any-return]

    def available_metrics(self) -> list[str]:
        """Return metrics that are present in the aggregated result table."""

        frame = self.aggregate_frame()
        return [
            metric
            for metric in self.metric_schema
            if metric in frame.columns and not frame[metric].isna().all()
        ]

    def pivot_table(self, metric: str) -> pd.DataFrame:
        """Return a baseline-by-case pivot table for one metric."""

        frame = self.aggregate_frame()
        if metric not in frame.columns:
            return pd.DataFrame()
        pivot = frame.pivot_table(
            index="baseline",
            columns="case_label",
            values=metric,
            aggfunc="mean",
        ).sort_index(axis=0)
        return pivot  # type: ignore[no-any-return]

    def to_markdown(self, *, top_k: int = 3) -> str:
        """Render a concise benchmark markdown summary."""

        lines = [
            "## Benchmark summary",
            "",
            f"- Scenario: {self.scenario_name}",
            f"- Family: {self.family}",
            f"- Seeds: {', '.join(str(seed) for seed in self.seeds)}",
            f"- Baselines: {', '.join(self.baseline_names)}",
            "",
        ]
        frame = self.aggregate_frame()
        if frame.empty:
            lines.append("- No benchmark rows were produced.")
            return "\n".join(lines)

        lines.extend(["### Publication metrics", ""])
        for metric in self.publication_metrics[:top_k]:
            if metric not in frame.columns:
                continue
            metric_label = self.metric_schema.get(metric, {}).get("label", metric)
            higher_is_better = bool(
                self.metric_schema.get(metric, {}).get("higher_is_better", False)
            )
            ranked = frame.sort_values(metric, ascending=not higher_is_better).head(1)
            if ranked.empty:
                continue
            row = ranked.iloc[0]
            lines.append(
                f"- {metric_label}: best baseline `{row['baseline']}` on "
                f"`{row['case_label']}` with mean {row[metric]:.3f}"
            )
        return "\n".join(lines)

    def to_latex(self, *, metric: str | None = None) -> str:
        """Render a LaTeX table for one publication metric."""

        metric_name = metric or next(iter(self.publication_metrics), "")
        pivot = self.pivot_table(metric_name)
        if pivot.empty:
            return ""
        return pivot.round(3).to_latex()

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable benchmark payload."""

        return {
            "scenario_name": self.scenario_name,
            "family": self.family,
            "description": self.description,
            "seeds": self.seeds,
            "baseline_names": self.baseline_names,
            "publication_metrics": self.publication_metrics,
            "scenario_config": self.scenario_config,
            "metric_schema": self.metric_schema,
            "run_records": [_jsonable(record) for record in self.run_records],
            "aggregated_records": [_jsonable(record) for record in self.aggregated_records],
            "created_at_utc": self.created_at_utc,
            "x_axis_label": self.x_axis_label,
        }

    def export_artifacts(
        self,
        output_dir: str | Path,
        *,
        figure_format: str = "png",
    ) -> dict[str, Any]:
        """Persist benchmark tables, report text, figures, and provenance metadata."""

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        tables_dir = directory / "tables"
        figures_dir = directory / "figures"
        tables_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)

        run_frame = self.run_frame()
        aggregate_frame = self.aggregate_frame()
        payload = self.to_dict()

        json_path = directory / f"{self.scenario_name}_benchmark.json"
        markdown_path = directory / f"{self.scenario_name}_benchmark.md"
        runs_csv_path = directory / f"{self.scenario_name}_runs.csv"
        summary_csv_path = directory / f"{self.scenario_name}_summary.csv"
        manifest_path = directory / f"{self.scenario_name}_artifacts.json"

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        run_frame.to_csv(runs_csv_path, index=False)
        aggregate_frame.to_csv(summary_csv_path, index=False)

        latex_tables: dict[str, Path] = {}
        figure_paths: dict[str, Path] = {}
        for metric in self.publication_metrics:
            if metric not in aggregate_frame.columns or aggregate_frame[metric].isna().all():
                continue
            latex = self.to_latex(metric=metric)
            if latex:
                latex_path = tables_dir / f"{self.scenario_name}_{metric}.tex"
                latex_path.write_text(latex, encoding="utf-8")
                latex_tables[metric] = latex_path

            figure, _ = plot_benchmark_metric_sweep(
                aggregate_frame,
                metric=metric,
                x_axis_label=self.x_axis_label,
                title=f"{self.scenario_name}: {metric}",
            )
            figure_path = figures_dir / f"{self.scenario_name}_{metric}.{figure_format}"
            figure.savefig(figure_path, dpi=220, bbox_inches="tight")
            plt.close(figure)
            figure_paths[metric] = figure_path

        primary_files = {
            "json": json_path,
            "markdown": markdown_path,
            "runs_csv": runs_csv_path,
            "summary_csv": summary_csv_path,
        }

        artifact_manifest = {
            "scenario_name": self.scenario_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_root": portable_path(directory, relative_to=directory),
            "files": {
                key: portable_path(path, relative_to=directory)
                for key, path in primary_files.items()
            },
            "latex_tables": {
                key: portable_path(path, relative_to=directory)
                for key, path in latex_tables.items()
            },
            "figures": {
                key: portable_path(path, relative_to=directory)
                for key, path in figure_paths.items()
            },
            "checksums": {
                **{
                    key: artifact_record(path, relative_to=directory)
                    for key, path in primary_files.items()
                },
                **{
                    f"latex_tables.{key}": artifact_record(path, relative_to=directory)
                    for key, path in latex_tables.items()
                },
                **{
                    f"figures.{key}": artifact_record(path, relative_to=directory)
                    for key, path in figure_paths.items()
                },
            },
        }
        manifest_path.write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
        return {
            "json": json_path,
            "markdown": markdown_path,
            "runs_csv": runs_csv_path,
            "summary_csv": summary_csv_path,
            "manifest": manifest_path,
            "latex_tables": latex_tables,
            "figures": figure_paths,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
