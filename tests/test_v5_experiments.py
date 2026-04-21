from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from shiftstat.experiments import load_experiment_config, run_experiment


@pytest.mark.integration
def test_experiment_config_parses_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: yaml_experiment",
                "output_dir: artifacts",
                "figure_format: png",
                "scenario:",
                "  preset: covariate_shift_sweep",
                "  parameters:",
                "    severities: [0.2]",
                "    n_samples_ref: 120",
                "    n_samples_target: 120",
                "  baseline_names: [raw_model]",
                "  seeds: [1]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.name == "yaml_experiment"
    assert config.scenarios[0].preset == "covariate_shift_sweep"
    assert config.scenarios[0].baseline_names == ["raw_model"]


@pytest.mark.benchmark
@pytest.mark.integration
def test_run_experiment_is_deterministic(tmp_path: Path) -> None:
    config = {
        "name": "deterministic_experiment",
        "figure_format": "png",
        "scenario": {
            "preset": "covariate_shift_sweep",
            "baseline_names": ["raw_model"],
            "seeds": [1, 2],
            "parameters": {
                "severities": [0.3],
                "n_samples_ref": 130,
                "n_samples_target": 130,
            },
        },
    }
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    first = run_experiment(config_path, output_dir=tmp_path / "run_a")
    second = run_experiment(config_path, output_dir=tmp_path / "run_b")

    assert first.summary_frame().equals(second.summary_frame())
    assert Path(first.manifest_path or "").exists()
    assert Path(second.markdown_path or "").exists()


@pytest.mark.integration
def test_experiment_cli_smoke(tmp_path: Path) -> None:
    config = {
        "name": "cli_experiment",
        "figure_format": "png",
        "scenario": {
            "preset": "selective_shift",
            "baseline_names": ["confidence_abstention"],
            "seeds": [1],
            "parameters": {
                "severities": [0.4],
                "n_samples_ref": 120,
                "n_samples_target": 120,
            },
        },
    }
    config_path = tmp_path / "cli_experiment.json"
    output_dir = tmp_path / "cli_outputs"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env = os.environ.copy()
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = (
        source_dir if not env.get("PYTHONPATH") else f"{source_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "shiftstat.experiments",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    manifest_path = Path(completed.stdout.strip())
    assert manifest_path.exists()
    assert (output_dir / "cli_experiment_summary.md").exists()


@pytest.mark.integration
def test_experiment_result_schema_stability(tmp_path: Path) -> None:
    config = {
        "name": "schema_experiment",
        "figure_format": "png",
        "scenario": {
            "preset": "covariate_shift_sweep",
            "baseline_names": ["raw_model"],
            "seeds": [1],
            "parameters": {
                "severities": [0.2],
                "n_samples_ref": 120,
                "n_samples_target": 120,
            },
        },
    }
    config_path = tmp_path / "schema.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = run_experiment(config_path, output_dir=tmp_path / "schema_outputs")
    payload = result.to_dict()

    assert set(payload) == {
        "benchmarks",
        "completed_at_utc",
        "config_path",
        "config_sha256",
        "environment",
        "experiment_name",
        "figure_format",
        "log_path",
        "manifest_path",
        "markdown_path",
        "output_dir",
        "scenario_artifacts",
        "shiftstat_version",
        "started_at_utc",
        "summary_csv_path",
    }
    assert payload["config_sha256"]
    assert "python" in payload["environment"]

    manifest = json.loads(Path(result.manifest_path or "").read_text(encoding="utf-8"))
    assert manifest["manifest_path"] == "schema_experiment_manifest.json"
    assert manifest["scenario_artifacts"][0]["scenario_dir"] == "covariate_shift_sweep"
