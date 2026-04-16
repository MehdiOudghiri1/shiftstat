"""End-to-end experiment configuration workflow."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from shiftstat.experiments import run_experiment


def run_example() -> dict[str, object]:
    """Run a full config-driven experiment in a temporary artifact directory."""

    config = {
        "name": "example_config_workflow",
        "figure_format": "png",
        "scenario": {
            "preset": "covariate_shift_sweep",
            "baseline_names": ["raw_model", "confidence_abstention"],
            "seeds": [1],
            "parameters": {
                "severities": [0.4],
                "n_samples_ref": 160,
                "n_samples_target": 160,
            },
        },
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "experiment.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = run_experiment(config_path, output_dir=root / "outputs")
        return {
            "manifest_exists": Path(result.manifest_path or "").exists(),
            "summary_rows": len(result.summary_frame()),
            "rerun_command": result.rerun_command(),
        }


if __name__ == "__main__":
    print(run_example())
