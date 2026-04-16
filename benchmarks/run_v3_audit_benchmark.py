"""Benchmark runner for subgroup-aware deployment auditing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from shiftstat.audit import ReliabilityAuditor
from shiftstat.datasets import make_hidden_subgroup_shift_classification


def run_v3_audit_benchmark(config_path: str | Path) -> dict[str, object]:
    """Run V3 benchmarks where aggregate metrics can hide localized failures."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    records: list[dict[str, object]] = []
    for pattern in config["patterns"]:
        for seed in config["random_states"]:
            data = make_hidden_subgroup_shift_classification(
                n_samples_ref=config["n_samples_ref"],
                n_samples_target=config["n_samples_target"],
                pattern=pattern,
                random_state=seed,
            )
            auditor = ReliabilityAuditor(
                min_group_size=config["min_group_size"],
                random_state=seed,
            ).fit(
                data.X_ref,
                data.y_ref,
                data.reference_predictions,
                data.X_target,
                data.y_target,
                data.target_predictions,
                subgroup_features=["region", "channel", "score", "load"],
                intersectional_features=[("region", "channel")],
            )
            report = auditor.to_report()
            comparison = auditor.aggregate_vs_subgroup_summary()
            accuracy_row = comparison.loc[comparison["metric"] == "accuracy"].iloc[0]
            ece_row = comparison.loc[comparison["metric"] == "ece"].iloc[0]
            top_slice = auditor.discovered_slices().iloc[0]
            records.append(
                {
                    "pattern": pattern,
                    "seed": int(seed),
                    "aggregate_delta_accuracy": report.aggregate_summary["delta_accuracy"],
                    "aggregate_delta_ece": report.aggregate_summary["delta_ece"],
                    "worst_group_delta_accuracy": accuracy_row["worst_group_delta"],
                    "worst_group_delta_ece": ece_row["worst_group_delta"],
                    "worst_group_accuracy_gap": accuracy_row["absolute_gap"],
                    "worst_group_ece_gap": ece_row["absolute_gap"],
                    "masked_accuracy_drop": report.hidden_failure_flags["masked_accuracy_drop"],
                    "masked_calibration_drift": report.hidden_failure_flags[
                        "masked_calibration_drift"
                    ],
                    "concentrated_failures": report.hidden_failure_flags["concentrated_failures"],
                    "top_discovered_slice": top_slice["rule"],
                }
            )

    summary = pd.DataFrame.from_records(records)
    return {
        "experiment": config["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "records": records,
        "summary_table": summary,
    }


def save_v3_audit_benchmark(
    result: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist V3 benchmark outputs as JSON, CSV, and a summary plot."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summary = result["summary_table"]
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("Benchmark result must include a pandas DataFrame under 'summary_table'.")

    json_path = directory / f"{result['experiment']}.json"
    csv_path = directory / f"{result['experiment']}_summary.csv"
    plot_path = directory / f"{result['experiment']}_worst_group_gap.png"

    with json_path.open("w", encoding="utf-8") as handle:
        payload = {
            "experiment": result["experiment"],
            "timestamp_utc": result["timestamp_utc"],
            "config": result["config"],
            "records": result["records"],
        }
        json.dump(payload, handle, indent=2)

    summary.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 4))
    grouped = (
        summary.groupby("pattern", as_index=False)[
            ["aggregate_delta_accuracy", "worst_group_delta_accuracy"]
        ]
        .mean()
    )
    x = range(len(grouped))
    width = 0.35
    axis.bar(
        [value - width / 2 for value in x],
        grouped["aggregate_delta_accuracy"],
        width=width,
        label="aggregate delta accuracy",
    )
    axis.bar(
        [value + width / 2 for value in x],
        grouped["worst_group_delta_accuracy"],
        width=width,
        label="worst-group delta accuracy",
    )
    axis.set_xticks(list(x))
    axis.set_xticklabels(grouped["pattern"], rotation=20)
    axis.set_ylabel("Delta accuracy")
    axis.set_title("Aggregate versus worst-group degradation")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {"json": json_path, "csv": csv_path, "plot": plot_path}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "configs" / "v3_audit_benchmark.json"
    result = run_v3_audit_benchmark(config)
    paths = save_v3_audit_benchmark(result, root / "results")
    print(paths)
