"""Reliability-focused benchmark runner for ShiftStat V2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reliability import evaluate_under_shift


def run_reliability_benchmark(config_path: str | Path) -> dict[str, object]:
    """Run a severity sweep benchmark comparing weighting and recalibration strategies."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    strategies = [
        {"name": "no_correction", "apply_importance_weighting": False, "recalibration": None},
        {
            "name": "weighting_only",
            "apply_importance_weighting": True,
            "recalibration": None,
        },
        {
            "name": "recalibration_only",
            "apply_importance_weighting": False,
            "recalibration": config["recalibration_method"],
        },
        {
            "name": "weighting_and_recalibration",
            "apply_importance_weighting": True,
            "recalibration": config["recalibration_method"],
        },
    ]

    records: list[dict[str, object]] = []
    for severity in config["severities"]:
        data = make_covariate_shift_classification(
            n_samples_ref=config["n_samples_ref"],
            n_samples_target=config["n_samples_target"],
            shift_strength=severity,
            random_state=config["random_state"],
        )
        for strategy in strategies:
            result = evaluate_under_shift(
                LogisticRegression(max_iter=2000),
                data.X_ref,
                data.y_ref,
                data.X_target,
                data.y_target,
                apply_importance_weighting=strategy["apply_importance_weighting"],
                weighting_method=config["weighting_method"],
                recalibration=strategy["recalibration"],
                random_state=config["random_state"],
            )
            record = {
                "severity": float(severity),
                "strategy": strategy["name"],
                "reference_accuracy": result.reference_profile.accuracy,
                "target_accuracy": result.target_profile.accuracy,
                "reference_ece": result.reference_profile.ece,
                "target_ece": result.target_profile.ece,
                "delta_accuracy": result.degradation_summary.delta_accuracy,
                "delta_ece": result.degradation_summary.delta_ece,
                "delta_log_loss": result.degradation_summary.delta_log_loss,
            }
            if result.recalibrated_target_profile is not None:
                record["recalibrated_target_ece"] = result.recalibrated_target_profile.ece
                record["recalibrated_target_log_loss"] = result.recalibrated_target_profile.log_loss
            if result.weighting_summary is not None:
                record["effective_sample_size"] = result.weighting_summary["effective_sample_size"]
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    return {
        "experiment": config["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "records": records,
        "summary_table": frame,
    }


def save_reliability_benchmark(
    result: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist benchmark outputs as JSON, CSV, and a summary plot."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = result["summary_table"]
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("Benchmark result must include a pandas DataFrame under 'summary_table'.")

    json_path = directory / f"{result['experiment']}.json"
    csv_path = directory / f"{result['experiment']}_summary.csv"
    plot_path = directory / f"{result['experiment']}_delta_ece.png"

    with json_path.open("w", encoding="utf-8") as handle:
        payload = {
            "experiment": result["experiment"],
            "timestamp_utc": result["timestamp_utc"],
            "config": result["config"],
            "records": result["records"],
        }
        json.dump(payload, handle, indent=2)

    summary.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(7, 4))
    for strategy, group in summary.groupby("strategy"):
        axis.plot(group["severity"], group["delta_ece"], marker="o", label=strategy)
    axis.set_xlabel("Shift severity")
    axis.set_ylabel("Delta ECE")
    axis.set_title("Reliability degradation across correction strategies")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {"json": json_path, "csv": csv_path, "plot": plot_path}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "configs" / "reliability_sweep.json"
    result = run_reliability_benchmark(config)
    paths = save_reliability_benchmark(result, root / "results")
    print(paths)
