"""Run a V5 publication-oriented benchmark suite from a config manifest."""

from __future__ import annotations

from pathlib import Path

from shiftstat.experiments import run_experiment


def run_v5_benchmark_suite(config_path: str | Path) -> dict[str, object]:
    """Run a config-driven V5 benchmark suite and return a compact manifest."""

    result = run_experiment(config_path)
    return {
        "manifest_path": result.manifest_path,
        "summary_csv_path": result.summary_csv_path,
        "n_scenarios": len(result.benchmark_results),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "configs" / "v5_publication_suite.yaml"
    print(run_v5_benchmark_suite(config))
