from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.run_reliability_benchmark import (
    run_reliability_benchmark,
    save_reliability_benchmark,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_reliability_benchmark_runner(tmp_path: Path) -> None:
    config = {
        "name": "test_benchmark",
        "random_state": 28,
        "n_samples_ref": 120,
        "n_samples_target": 120,
        "severities": [0.3, 0.8],
        "recalibration_method": "temperature",
        "weighting_method": "domain_classifier",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_reliability_benchmark(config_path)
    paths = save_reliability_benchmark(result, tmp_path / "outputs")

    assert len(result["records"]) == 8
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["plot"].exists()
