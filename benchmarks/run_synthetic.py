"""Synthetic benchmark runner for future experiment suites."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import ImportanceWeighter, weighted_risk


def run_benchmark(config_path: str | Path) -> dict[str, object]:
    """Run a synthetic benchmark experiment from a JSON config."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    data = make_covariate_shift_classification(
        n_samples_ref=config["n_samples_ref"],
        n_samples_target=config["n_samples_target"],
        shift_strength=config["shift_strength"],
        random_state=config["random_state"],
    )
    detector = ShiftDetector(random_state=config["random_state"])
    detector.fit(data.X_ref, data.X_target)

    weighter = ImportanceWeighter(
        method=config["weighting_method"],
        random_state=config["random_state"],
    )
    weights = weighter.fit_predict(data.X_ref, data.X_target)

    result = {
        "experiment": config["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": detector.dataset_summary_.to_dict(),
        "top_features": detector.summary()["feature_name"].head(5).tolist(),
        "weight_summary": weighter.summary(),
        "weighted_log_loss": weighted_risk(
            data.y_ref,
            data.reference_predictions,
            sample_weight=weights,
            loss="log_loss",
        ),
    }
    return result


def save_result(result: dict[str, object], output_dir: str | Path) -> Path:
    """Persist a benchmark result as JSON."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{result['experiment']}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "configs" / "default_experiment.json"
    result = run_benchmark(config)
    path = save_result(result, root / "results")
    print(f"Saved benchmark result to {path}")
