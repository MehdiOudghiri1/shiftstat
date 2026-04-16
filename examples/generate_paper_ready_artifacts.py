"""Generate paper-ready figures and tables from a compact V5 benchmark."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from shiftstat.bench import BenchmarkRunner, make_covariate_shift_sweep_scenario


def run_example() -> dict[str, object]:
    """Export markdown, CSV, LaTeX, and figure artifacts for one benchmark."""

    scenario = make_covariate_shift_sweep_scenario(
        name="paper_ready_example",
        severities=[0.3, 0.8],
        seeds=[2],
        n_samples_ref=180,
        n_samples_target=180,
        baseline_names=["raw_model", "weighting_only", "confidence_abstention"],
    )
    result = BenchmarkRunner().run(scenario)
    with TemporaryDirectory() as directory:
        paths = result.export_artifacts(Path(directory))
        return {
            "latex_tables": sorted(paths["latex_tables"]),
            "figures": sorted(paths["figures"]),
            "manifest_exists": Path(paths["manifest"]).exists(),
        }


if __name__ == "__main__":
    print(run_example())
