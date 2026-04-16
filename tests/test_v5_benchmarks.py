from __future__ import annotations

from pathlib import Path

from benchmarks.run_v5_benchmark_suite import run_v5_benchmark_suite


def test_v5_benchmark_suite_runner(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: v5_suite_test",
                f"output_dir: {str(tmp_path / 'outputs').replace(chr(92), '/')}",
                "figure_format: png",
                "scenario:",
                "  preset: covariate_shift_sweep",
                "  baseline_names: [raw_model, confidence_abstention]",
                "  seeds: [1]",
                "  parameters:",
                "    severities: [0.3]",
                "    n_samples_ref: 140",
                "    n_samples_target: 140",
            ]
        ),
        encoding="utf-8",
    )

    result = run_v5_benchmark_suite(config_path)

    assert result["n_scenarios"] == 1
    assert Path(str(result["manifest_path"])).exists()
    assert Path(str(result["summary_csv_path"])).exists()
