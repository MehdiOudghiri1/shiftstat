"""Benchmark execution over cases, baselines, and seeds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd

from shiftstat.bench.baselines import default_baseline_registry
from shiftstat.bench.registry import BaselineRegistry, MetricRegistry, default_metric_registry
from shiftstat.bench.results import BenchmarkResult
from shiftstat.bench.scenarios import BenchmarkScenario


class BenchmarkRunner:
    """Run repeatable benchmark scenarios across multiple baselines and seeds."""

    def __init__(
        self,
        *,
        baseline_registry: BaselineRegistry | None = None,
        metric_registry: MetricRegistry | None = None,
    ) -> None:
        self.baseline_registry = baseline_registry or default_baseline_registry()
        self.metric_registry = metric_registry or default_metric_registry()

    def run(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Execute one benchmark scenario."""

        records: list[dict[str, Any]] = []
        for seed in scenario.seeds:
            for case_definition in scenario.case_definitions:
                case = scenario.build_case(case_definition, seed)
                for baseline_name in scenario.baseline_names:
                    baseline = self.baseline_registry.get(baseline_name)
                    output = baseline.runner(case, seed)
                    record = {
                        "scenario_name": scenario.name,
                        "scenario_family": scenario.family,
                        "case_id": case.case_id,
                        "case_label": case.case_label,
                        "case_value": case.case_value,
                        "baseline": baseline.name,
                        "baseline_category": baseline.category,
                        "seed": int(seed),
                    }
                    record.update(case.parameters)
                    record.update(output)
                    records.append(record)

        aggregated_records = self._aggregate(records)
        return BenchmarkResult(
            scenario_name=scenario.name,
            family=scenario.family,
            description=scenario.description,
            seeds=scenario.seeds,
            baseline_names=scenario.baseline_names,
            publication_metrics=scenario.publication_metrics,
            scenario_config=scenario.to_dict(),
            metric_schema=self.metric_registry.to_dict(),
            run_records=records,
            aggregated_records=aggregated_records,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            x_axis_label=scenario.x_axis_label,
        )

    def _aggregate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        frame = pd.DataFrame.from_records(records)
        group_columns = [
            "scenario_name",
            "scenario_family",
            "case_id",
            "case_label",
            "case_value",
            "baseline",
            "baseline_category",
        ]
        metric_names = [
            metric for metric in self.metric_registry.names() if metric in frame.columns
        ]
        if not metric_names:
            return cast(list[dict[str, Any]], frame.to_dict(orient="records"))

        aggregated = (
            frame.groupby(group_columns, dropna=False)[metric_names]
            .agg(["mean", "std"])
            .reset_index()
        )
        flattened_columns: list[str] = []
        for column in aggregated.columns:
            if isinstance(column, tuple):
                base, suffix = column
                flattened_columns.append(str(base) if suffix == "" else f"{base}_{suffix}")
            else:
                flattened_columns.append(str(column))
        aggregated.columns = flattened_columns

        rename_map = {
            f"{metric}_mean": metric
            for metric in metric_names
            if f"{metric}_mean" in aggregated.columns
        }
        aggregated = aggregated.rename(columns=rename_map)
        aggregated["n_runs"] = (
            frame.groupby(group_columns, dropna=False)["seed"].nunique().reset_index(drop=True)
        )
        return cast(list[dict[str, Any]], aggregated.to_dict(orient="records"))
