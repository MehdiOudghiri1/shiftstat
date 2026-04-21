from __future__ import annotations

import json
from pathlib import Path

import pytest

from shiftstat.experiments.cli import main


@pytest.mark.integration
def test_experiment_cli_main_returns_success(tmp_path: Path) -> None:
    config = {
        "name": "direct_cli",
        "figure_format": "png",
        "scenario": {
            "preset": "covariate_shift_sweep",
            "baseline_names": ["raw_model"],
            "seeds": [1],
            "parameters": {
                "severities": [0.2],
                "n_samples_ref": 80,
                "n_samples_target": 80,
            },
        },
    }
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "outputs"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = main([str(config_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "direct_cli_manifest.json").exists()
