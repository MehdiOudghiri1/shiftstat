from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.run_v3_audit_benchmark import (
    run_v3_audit_benchmark,
    save_v3_audit_benchmark,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_v3_audit_benchmark_runner(tmp_path: Path) -> None:
    config = {
        "name": "test_v3_benchmark",
        "n_samples_ref": 180,
        "n_samples_target": 180,
        "min_group_size": 15,
        "patterns": [
            "masked_subgroup_shift",
            "minority_subgroup_degradation",
            "operational_calibration_drift",
        ],
        "random_states": [51],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_v3_audit_benchmark(config_path)
    paths = save_v3_audit_benchmark(result, tmp_path / "outputs")

    assert len(result["records"]) == 3
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["plot"].exists()
