"""Benchmark selective prediction under distribution shift."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.datasets import (
    make_covariate_shift_classification,
    make_hidden_subgroup_shift_classification,
)
from shiftstat.selective import evaluate_selective_under_shift


def _mixed_estimator() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["region", "channel"],
                        ),
                        ("continuous", "passthrough", ["score", "load", "signal"]),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )


def run_selective_benchmark(config_path: str | Path) -> dict[str, object]:
    """Run V4 benchmarks for selective deployment strategies."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    severity_records: list[dict[str, object]] = []
    strategies = [
        {
            "name": "fixed_confidence",
            "policy_threshold": config["fixed_threshold"],
            "target_coverage": None,
            "apply_importance_weighting": False,
            "use_weighted_threshold_tuning": False,
            "recalibration": None,
        },
        {
            "name": "tuned_confidence",
            "policy_threshold": None,
            "target_coverage": config["target_coverage"],
            "apply_importance_weighting": False,
            "use_weighted_threshold_tuning": False,
            "recalibration": None,
        },
        {
            "name": "weighted_tuned_confidence",
            "policy_threshold": None,
            "target_coverage": config["target_coverage"],
            "apply_importance_weighting": True,
            "use_weighted_threshold_tuning": True,
            "recalibration": None,
        },
        {
            "name": "weighted_tuned_recalibrated",
            "policy_threshold": None,
            "target_coverage": config["target_coverage"],
            "apply_importance_weighting": True,
            "use_weighted_threshold_tuning": True,
            "recalibration": config["recalibration_method"],
        },
    ]

    for severity in config["severities"]:
        data = make_covariate_shift_classification(
            n_samples_ref=config["n_samples_ref"],
            n_samples_target=config["n_samples_target"],
            shift_strength=severity,
            random_state=config["random_state"],
        )
        for strategy in strategies:
            result = evaluate_selective_under_shift(
                LogisticRegression(max_iter=2000),
                data.X_ref,
                data.y_ref,
                data.X_target,
                data.y_target,
                policy_method="confidence",
                policy_threshold=strategy["policy_threshold"],
                target_coverage=strategy["target_coverage"],
                apply_importance_weighting=strategy["apply_importance_weighting"],
                use_weighted_threshold_tuning=strategy["use_weighted_threshold_tuning"],
                compare_threshold_tuning=True,
                recalibration=strategy["recalibration"],
                random_state=config["random_state"],
            )
            record = {
                "severity": float(severity),
                "strategy": strategy["name"],
                "target_coverage": result.target_selective_profile.coverage,
                "target_selective_risk": result.target_selective_profile.selective_risk,
                "target_risk_reduction": result.target_selective_profile.risk_reduction,
                "target_selective_ece": result.target_selective_profile.selective_ece,
                "target_ece_reduction": result.target_selective_profile.ece_reduction,
                "threshold": result.policy_summary["threshold"],
            }
            if result.recalibrated_target_selective_profile is not None:
                record["recalibrated_target_selective_ece"] = (
                    result.recalibrated_target_selective_profile.selective_ece
                )
                record["recalibrated_target_selective_log_loss"] = (
                    result.recalibrated_target_selective_profile.selective_log_loss
                )
            severity_records.append(record)

    hidden_failure_records: list[dict[str, object]] = []
    for pattern in config["hidden_failure_patterns"]:
        data = make_hidden_subgroup_shift_classification(
            n_samples_ref=config["hidden_failure_samples"],
            n_samples_target=config["hidden_failure_samples"],
            pattern=pattern,
            random_state=config["random_state"],
        )
        result = evaluate_selective_under_shift(
            _mixed_estimator(),
            data.X_ref,
            data.y_ref,
            data.X_target,
            data.y_target,
            policy_method="confidence",
            target_coverage=config["target_coverage"],
            apply_importance_weighting=True,
            use_weighted_threshold_tuning=True,
            subgroup_features=["region", "channel", "load"],
            intersectional_features=[("region", "channel")],
            random_state=config["random_state"],
        )
        disparity_frame = pd.DataFrame.from_records(result.subgroup_abstention_disparities)
        abstention_gap = disparity_frame.loc[
            disparity_frame["metric"] == "target_abstention_rate", "absolute_gap"
        ].iloc[0]
        hidden_failure_records.append(
            {
                "pattern": pattern,
                "target_coverage": result.target_selective_profile.coverage,
                "target_risk_reduction": result.target_selective_profile.risk_reduction,
                "target_selective_ece": result.target_selective_profile.selective_ece,
                "subgroup_abstention_gap": float(abstention_gap),
            }
        )

    return {
        "experiment": config["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "severity_records": severity_records,
        "hidden_failure_records": hidden_failure_records,
        "severity_summary_table": pd.DataFrame.from_records(severity_records),
        "hidden_failure_summary_table": pd.DataFrame.from_records(hidden_failure_records),
    }


def save_selective_benchmark(
    result: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist selective benchmark outputs as JSON, CSVs, and a summary plot."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    severity_summary = result["severity_summary_table"]
    hidden_failure_summary = result["hidden_failure_summary_table"]
    if not isinstance(severity_summary, pd.DataFrame) or not isinstance(
        hidden_failure_summary, pd.DataFrame
    ):
        raise TypeError("Benchmark result must include pandas DataFrames.")

    json_path = directory / f"{result['experiment']}.json"
    severity_csv = directory / f"{result['experiment']}_severity.csv"
    hidden_csv = directory / f"{result['experiment']}_hidden_failure.csv"
    plot_path = directory / f"{result['experiment']}_risk_reduction.png"

    with json_path.open("w", encoding="utf-8") as handle:
        payload = {
            "experiment": result["experiment"],
            "timestamp_utc": result["timestamp_utc"],
            "config": result["config"],
            "severity_records": result["severity_records"],
            "hidden_failure_records": result["hidden_failure_records"],
        }
        json.dump(payload, handle, indent=2)

    severity_summary.to_csv(severity_csv, index=False)
    hidden_failure_summary.to_csv(hidden_csv, index=False)

    figure, axis = plt.subplots(figsize=(8, 4))
    for strategy, group in severity_summary.groupby("strategy"):
        axis.plot(group["severity"], group["target_risk_reduction"], marker="o", label=strategy)
    axis.set_xlabel("Shift severity")
    axis.set_ylabel("Risk reduction")
    axis.set_title("Selective risk reduction across shift severity")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {
        "json": json_path,
        "severity_csv": severity_csv,
        "hidden_failure_csv": hidden_csv,
        "plot": plot_path,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "configs" / "selective_benchmark.json"
    result = run_selective_benchmark(config)
    paths = save_selective_benchmark(result, root / "results")
    print(paths)
