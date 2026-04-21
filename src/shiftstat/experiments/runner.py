"""Execute config-driven benchmark experiments and persist artifacts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shiftstat._version import __version__
from shiftstat.bench import BenchmarkRunner, scenario_from_config
from shiftstat.experiments.config import load_experiment_config
from shiftstat.experiments.results import ExperimentResult
from shiftstat.utils.artifacts import file_digest, portable_path


def run_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    figure_format: str | None = None,
) -> ExperimentResult:
    """Run one experiment manifest and persist publication-ready artifacts."""

    config_file = Path(config_path).resolve()
    config = load_experiment_config(config_file)
    resolved_output_dir = Path(output_dir or config.output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    runner = BenchmarkRunner()
    benchmark_results = []
    scenario_artifacts = []
    log_lines = [
        f"[{started_at}] starting experiment {config.name}",
        f"config={config_file}",
        f"output_dir={resolved_output_dir}",
    ]

    for scenario_config in config.scenarios:
        scenario = scenario_from_config(scenario_config.to_dict())
        benchmark = runner.run(scenario)
        scenario_dir = resolved_output_dir / scenario.name
        artifacts = benchmark.export_artifacts(
            scenario_dir,
            figure_format=figure_format or config.figure_format,
        )
        benchmark_results.append(benchmark)
        scenario_artifacts.append(
            {
                "scenario_name": scenario.name,
                "scenario_dir": portable_path(scenario_dir, relative_to=resolved_output_dir),
                "artifact_files": {
                    key: _jsonable(value, relative_to=resolved_output_dir)
                    for key, value in artifacts.items()
                },
            }
        )
        log_lines.append(
            f"[{datetime.now(timezone.utc).isoformat()}] completed scenario {scenario.name}"
        )

    result = ExperimentResult(
        experiment_name=config.name,
        config_path=str(config_file),
        output_dir=str(resolved_output_dir),
        figure_format=figure_format or config.figure_format,
        benchmark_results=benchmark_results,
        scenario_artifacts=scenario_artifacts,
        started_at_utc=started_at,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        shiftstat_version=__version__,
        config_sha256=file_digest(config_file),
        environment=_environment_snapshot(),
    )
    summary_frame = result.summary_frame()
    summary_csv_path = resolved_output_dir / f"{config.name}_summary.csv"
    if not summary_frame.empty:
        summary_frame.to_csv(summary_csv_path, index=False)
    markdown_path = resolved_output_dir / f"{config.name}_summary.md"
    markdown_path.write_text(result.to_markdown(), encoding="utf-8")

    manifest_path = resolved_output_dir / f"{config.name}_manifest.json"
    config_copy_path = resolved_output_dir / f"{config.name}_config{config_file.suffix.lower()}"
    config_copy_path.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = resolved_output_dir / f"{config.name}.log"
    log_lines.append(f"rerun={result.rerun_command()}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    hydrated = ExperimentResult(
        experiment_name=result.experiment_name,
        config_path=result.config_path,
        output_dir=result.output_dir,
        figure_format=result.figure_format,
        benchmark_results=result.benchmark_results,
        scenario_artifacts=result.scenario_artifacts,
        started_at_utc=result.started_at_utc,
        completed_at_utc=result.completed_at_utc,
        shiftstat_version=result.shiftstat_version,
        config_sha256=result.config_sha256,
        environment=result.environment,
        manifest_path=str(manifest_path),
        summary_csv_path=str(summary_csv_path),
        markdown_path=str(markdown_path),
        log_path=str(log_path),
    )
    manifest_path.write_text(
        json.dumps(hydrated.to_dict(relative_to=resolved_output_dir), indent=2),
        encoding="utf-8",
    )
    return hydrated


def _environment_snapshot() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _jsonable(value: Any, *, relative_to: Path | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, relative_to=relative_to) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_jsonable(item, relative_to=relative_to) for item in value]
    if isinstance(value, Path):
        return portable_path(value, relative_to=relative_to)
    return value
