from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.run_selective_benchmark import (
    run_selective_benchmark,
    save_selective_benchmark,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_selective_benchmark_runner(tmp_path: Path) -> None:
    config = {
        "name": "test_selective_benchmark",
        "random_state": 123,
        "n_samples_ref": 160,
        "n_samples_target": 160,
        "severities": [0.4, 0.9],
        "fixed_threshold": 0.75,
        "target_coverage": 0.8,
        "recalibration_method": "temperature",
        "hidden_failure_patterns": ["masked_subgroup_shift"],
        "hidden_failure_samples": 160,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_selective_benchmark(config_path)
    paths = save_selective_benchmark(result, tmp_path / "outputs")

    assert len(result["severity_records"]) == 8
    assert len(result["hidden_failure_records"]) == 1
    assert paths["json"].exists()
    assert paths["severity_csv"].exists()
    assert paths["hidden_failure_csv"].exists()
    assert paths["plot"].exists()
