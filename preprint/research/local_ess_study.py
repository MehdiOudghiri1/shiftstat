"""Research script for local effective sample size under covariate shift.

This script studies a simple but important question:

When worst-group calibration is estimated by importance weighting under
covariate shift, is the relevant notion of information the global effective
sample size or a subgroup-bin-local effective sample size?

The synthetic benchmark used here is intentionally clean. The target model is
perfectly calibrated by construction under pure covariate shift, so any
estimated subgroup calibration gap is sampling noise rather than a true target
failure. This makes it possible to study when subgroup audits create false
alarms due to overlap collapse.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from shiftstat.datasets.synthetic import make_covariate_shift_classification
from shiftstat.metrics.weighted import compute_effective_sample_size
from shiftstat.reweight.importance import ImportanceWeighter


def _oracle_covariate_shift_weights(
    frame: pd.DataFrame,
    *,
    shift_strength: float,
    n_features: int,
) -> np.ndarray:
    """Return the exact target-over-source density ratio for the benchmark."""

    matrix = frame[[f"x{i}" for i in range(n_features)]].to_numpy(dtype=float)
    shift_dimensions = max(1, n_features // 2)
    mean_shift = np.zeros(n_features, dtype=float)
    mean_shift[:shift_dimensions] = shift_strength
    # Source: N(0, I), target: N(mu, I), so q(x)/p(x)=exp(mu^T x - ||mu||^2 / 2).
    weights = np.exp(matrix @ mean_shift - 0.5 * float(np.sum(mean_shift**2)))
    return np.asarray(weights, dtype=float)


def _default_group_family(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build an interpretable family of tail and interaction subgroups."""

    groups: dict[str, np.ndarray] = {}
    for feature in ("x0", "x1", "x2"):
        values = frame[feature].to_numpy(dtype=float)
        for threshold in (0.5, 1.0, 1.5):
            groups[f"{feature}>{threshold:g}"] = values > threshold
        for threshold in (-0.5, 0.0):
            groups[f"{feature}<{threshold:g}"] = values < threshold

    for left, right in (("x0", "x1"), ("x0", "x2"), ("x1", "x2")):
        left_values = frame[left].to_numpy(dtype=float)
        right_values = frame[right].to_numpy(dtype=float)
        for left_threshold, right_threshold in ((1.0, 1.0), (1.5, 1.0), (1.5, 1.5)):
            name = f"{left}>{left_threshold:g} & {right}>{right_threshold:g}"
            groups[name] = (left_values > left_threshold) & (right_values > right_threshold)
    return groups


@dataclass(frozen=True)
class ReplicateSummary:
    weight_mode: str
    shift_strength: float
    seed: int
    global_ess: float
    q10_local_ess: float
    median_local_ess: float
    worst_group_noise: float
    false_alarm_005: bool
    false_alarm_010: bool


def _active_bin_mask(
    probabilities: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    if upper < 1.0:
        return (probabilities >= lower) & (probabilities < upper)
    return (probabilities >= lower) & (probabilities <= upper)


def _local_ess(weights: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return compute_effective_sample_size(weights[mask])


def _bin_group_diagnostics(
    *,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    shift_strength: float,
    seed: int,
    bins: np.ndarray,
    min_target_mass: float,
    weight_mode: str,
) -> tuple[pd.DataFrame, ReplicateSummary]:
    residual = y_true.astype(float) - probabilities.astype(float)
    groups = _default_group_family(frame)
    global_ess = compute_effective_sample_size(weights)
    rows: list[dict[str, float | int | str]] = []
    worst_group_noise = 0.0
    supported_local_ess: list[float] = []

    for group_name, group_mask in groups.items():
        group_sum = 0.0
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            bin_mask = _active_bin_mask(probabilities, lower=lower, upper=upper)
            active_mask = group_mask & bin_mask
            if not np.any(active_mask):
                continue

            indicator = active_mask.astype(float)
            theta_hat = float(np.mean(weights * residual * indicator))
            local_weights = weights[active_mask]
            bin_group_ess = compute_effective_sample_size(local_weights)
            target_mass_estimate = float(np.sum(local_weights) / len(weights))
            normalized_gap = float(np.average(residual[active_mask], weights=local_weights))

            if target_mass_estimate >= min_target_mass:
                supported_local_ess.append(bin_group_ess)

            rows.append(
                {
                    "weight_mode": weight_mode,
                    "shift_strength": shift_strength,
                    "seed": seed,
                    "group": group_name,
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "n_active": int(np.sum(active_mask)),
                    "global_ess": float(global_ess),
                    "bin_group_ess": float(bin_group_ess),
                    "target_mass_est": float(target_mass_estimate),
                    "abs_theta_hat": abs(theta_hat),
                    "abs_normalized_gap": abs(normalized_gap),
                }
            )
            group_sum += abs(theta_hat)
        worst_group_noise = max(worst_group_noise, group_sum)

    local_array = np.asarray(supported_local_ess, dtype=float)
    if local_array.size:
        q10_local_ess = float(np.quantile(local_array, 0.1))
        median_local_ess = float(np.median(local_array))
    else:
        q10_local_ess = float("nan")
        median_local_ess = float("nan")

    summary = ReplicateSummary(
        weight_mode=weight_mode,
        shift_strength=float(shift_strength),
        seed=int(seed),
        global_ess=float(global_ess),
        q10_local_ess=q10_local_ess,
        median_local_ess=median_local_ess,
        worst_group_noise=float(worst_group_noise),
        false_alarm_005=bool(worst_group_noise > 0.05),
        false_alarm_010=bool(worst_group_noise > 0.10),
    )
    return pd.DataFrame.from_records(rows), summary


def _summarize_replicates(
    bin_group_frame: pd.DataFrame,
    replicate_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | dict[str, dict[str, float]]]]:
    by_shift = (
        replicate_frame.groupby(["weight_mode", "shift_strength"], as_index=False)
        .agg(
            median_global_ess=("global_ess", "median"),
            median_q10_local_ess=("q10_local_ess", "median"),
            median_local_ess=("median_local_ess", "median"),
            median_worst_group_noise=("worst_group_noise", "median"),
            p90_worst_group_noise=("worst_group_noise", lambda values: float(np.quantile(values, 0.9))),
            false_alarm_rate_005=("false_alarm_005", "mean"),
            false_alarm_rate_010=("false_alarm_010", "mean"),
        )
        .sort_values(["weight_mode", "shift_strength"], ignore_index=True)
    )

    supported = bin_group_frame.loc[bin_group_frame["target_mass_est"] >= 0.02].copy()
    supported = supported.dropna(subset=["bin_group_ess", "global_ess", "abs_normalized_gap"])
    local_corr = float(
        np.corrcoef(np.log1p(supported["bin_group_ess"]), supported["abs_normalized_gap"])[0, 1]
    )
    global_corr = float(
        np.corrcoef(np.log1p(supported["global_ess"]), supported["abs_normalized_gap"])[0, 1]
    )
    worst_local_corr = float(
        np.corrcoef(np.log1p(replicate_frame["q10_local_ess"]), replicate_frame["worst_group_noise"])[0, 1]
    )
    worst_global_corr = float(
        np.corrcoef(np.log1p(replicate_frame["global_ess"]), replicate_frame["worst_group_noise"])[0, 1]
    )

    diagnostics = {
        "corr_log_local_ess_vs_abs_normalized_gap": local_corr,
        "corr_log_global_ess_vs_abs_normalized_gap": global_corr,
        "corr_log_q10_local_ess_vs_worst_group_noise": worst_local_corr,
        "corr_log_global_ess_vs_worst_group_noise": worst_global_corr,
        "within_shift_worst_group_noise_correlations": {
            f"{weight_mode}:{shift_strength:g}": {
                "corr_log_q10_local_ess": float(
                    np.corrcoef(
                        np.log1p(subframe["q10_local_ess"]),
                        subframe["worst_group_noise"],
                    )[0, 1]
                ),
                "corr_log_global_ess": float(
                    np.corrcoef(
                        np.log1p(subframe["global_ess"]),
                        subframe["worst_group_noise"],
                    )[0, 1]
                ),
            }
            for (weight_mode, shift_strength), subframe in replicate_frame.groupby(
                ["weight_mode", "shift_strength"]
            )
        },
    }
    return by_shift, diagnostics


def run_study(
    *,
    shifts: Iterable[float],
    seeds: Iterable[int],
    n_reference: int,
    n_target: int,
    n_features: int,
    n_bins: int,
    min_target_mass: float,
    weight_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | dict[str, dict[str, float]]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_group_frames: list[pd.DataFrame] = []
    summaries: list[ReplicateSummary] = []

    for shift_strength in shifts:
        for seed in seeds:
            dataset = make_covariate_shift_classification(
                n_samples_ref=n_reference,
                n_samples_target=n_target,
                n_features=n_features,
                shift_strength=float(shift_strength),
                random_state=int(seed),
            )
            if weight_mode == "oracle":
                weights = _oracle_covariate_shift_weights(
                    dataset.X_ref,
                    shift_strength=float(shift_strength),
                    n_features=n_features,
                )
                weights = weights / np.mean(weights)
            else:
                weights = ImportanceWeighter(
                    method=weight_mode,
                    random_state=int(seed),
                ).fit_predict(dataset.X_ref, dataset.X_target)
            bin_group_frame, replicate_summary = _bin_group_diagnostics(
                frame=dataset.X_ref,
                y_true=dataset.y_ref,
                probabilities=dataset.reference_predictions,
                weights=weights,
                shift_strength=float(shift_strength),
                seed=int(seed),
                bins=bins,
                min_target_mass=min_target_mass,
                weight_mode=weight_mode,
            )
            bin_group_frames.append(bin_group_frame)
            summaries.append(replicate_summary)

    bin_group_table = pd.concat(bin_group_frames, ignore_index=True)
    replicate_table = pd.DataFrame.from_records(asdict(summary) for summary in summaries)
    summary_table, diagnostics = _summarize_replicates(bin_group_table, replicate_table)
    return summary_table, bin_group_table, replicate_table, diagnostics


def _write_outputs(
    *,
    output_dir: Path,
    summary_table: pd.DataFrame,
    bin_group_table: pd.DataFrame,
    replicate_table: pd.DataFrame,
    diagnostics: dict[str, float | dict[str, dict[str, float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_dir / "local_ess_summary.csv", index=False)
    replicate_table.to_csv(output_dir / "local_ess_replicates.csv", index=False)
    bin_group_table.to_csv(output_dir / "local_ess_bin_groups.csv", index=False)
    with (output_dir / "local_ess_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reference", type=int, default=500)
    parser.add_argument("--n-target", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=7)
    parser.add_argument("--min-target-mass", type=float, default=0.01)
    parser.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=[0.4, 0.8, 1.2, 1.6],
    )
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprint/research/results/local_ess"),
    )
    parser.add_argument(
        "--weight-mode",
        choices=["oracle", "logistic", "domain_classifier"],
        default="oracle",
    )
    args = parser.parse_args()

    summary_table, bin_group_table, replicate_table, diagnostics = run_study(
        shifts=args.shifts,
        seeds=range(args.n_seeds),
        n_reference=args.n_reference,
        n_target=args.n_target,
        n_features=args.n_features,
        n_bins=args.n_bins,
        min_target_mass=args.min_target_mass,
        weight_mode=args.weight_mode,
    )
    _write_outputs(
        output_dir=args.output_dir,
        summary_table=summary_table,
        bin_group_table=bin_group_table,
        replicate_table=replicate_table,
        diagnostics=diagnostics,
    )

    print("Local ESS summary:")
    print(summary_table.to_string(index=False))
    print()
    print("Correlation diagnostics:")
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for inner_key, inner_value in value.items():
                print(f"    {inner_key}:")
                for metric_name, metric_value in inner_value.items():
                    print(f"      {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    print()
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
