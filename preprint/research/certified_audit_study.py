"""Monte Carlo study for certified worst-group auditing under covariate shift.

The local-ESS experiment shows that worst-group point estimates can look large
under a perfectly calibrated covariate-shift null. This script asks the next
statistical question: which audit rules convert those point estimates into
trustworthy alarms?

We compare four rules:

* naive_max: report an alarm when the largest subgroup-bin gap exceeds a
  practical tolerance.
* global_ess_gate: use the same rule only when the global ESS is above a
  reassuring threshold.
* local_ess_gate: only allow cells whose own local ESS is above a threshold.
* simultaneous_ci: require a Bonferroni simultaneous lower confidence bound to
  exceed the practical tolerance.

The null scenario uses the true conditional probability as the score, so every
alarm is false. The planted scenario keeps the same covariate-shift geometry but
adds a controlled subgroup residual, making it possible to measure detection
power and whether the selected alarm localizes the planted subgroup.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from local_ess_study import (
    _active_bin_mask,
    _default_group_family,
    _oracle_covariate_shift_weights,
)
from scipy.stats import norm

from shiftstat.datasets.synthetic import make_covariate_shift_classification
from shiftstat.metrics.weighted import compute_effective_sample_size

PLANTED_GROUP = "x0>0.5"


@dataclass(frozen=True)
class AuditCell:
    scenario: str
    shift_strength: float
    seed: int
    group: str
    bin: str
    n_active: int
    target_mass_est: float
    global_ess: float
    local_ess: float
    abs_gap: float
    se_bound: float
    ci_half_width: float
    is_planted_group: bool


@dataclass(frozen=True)
class AuditDecision:
    scenario: str
    method: str
    shift_strength: float
    seed: int
    alarm: bool
    abstained: bool
    selected_group: str
    selected_bin: str
    selected_abs_gap: float
    selected_local_ess: float
    selected_ci_half_width: float
    global_ess: float
    n_cells: int
    n_eligible_cells: int
    selected_is_planted_group: bool


def _plant_subgroup_failure(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    delta: float,
    seed: int,
) -> np.ndarray:
    """Generate labels with a positive residual on one interpretable subgroup."""

    groups = _default_group_family(frame)
    planted_mask = groups[PLANTED_GROUP]
    shifted_probabilities = probabilities.copy()
    shifted_probabilities[planted_mask] = np.clip(
        shifted_probabilities[planted_mask] + delta,
        1e-3,
        1.0 - 1e-3,
    )
    rng = np.random.RandomState(seed + 104_729)
    return rng.binomial(1, shifted_probabilities).astype(int)


def _cell_table(
    *,
    scenario: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    shift_strength: float,
    seed: int,
    bins: np.ndarray,
    min_target_mass: float,
    alpha: float,
) -> pd.DataFrame:
    """Build subgroup-bin estimates and simultaneous-CI ingredients."""

    residual = y_true.astype(float) - probabilities.astype(float)
    groups = _default_group_family(frame)
    global_ess = compute_effective_sample_size(weights)
    raw_rows: list[dict[str, object]] = []

    for group_name, group_mask in groups.items():
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            bin_mask = _active_bin_mask(probabilities, lower=float(lower), upper=float(upper))
            active_mask = group_mask & bin_mask
            if not np.any(active_mask):
                continue

            local_weights = weights[active_mask]
            denominator = float(np.sum(local_weights))
            target_mass_estimate = denominator / len(weights)
            if target_mass_estimate < min_target_mass:
                continue

            normalized_gap = float(np.average(residual[active_mask], weights=local_weights))
            local_ess = float(compute_effective_sample_size(local_weights))
            # Bernoulli residuals are bounded in [-1, 1] with variance at most 1/4.
            # This gives a transparent conservative cellwise standard-error scale.
            se_bound = 0.5 / np.sqrt(max(local_ess, 1e-12))
            raw_rows.append(
                {
                    "scenario": scenario,
                    "shift_strength": float(shift_strength),
                    "seed": int(seed),
                    "group": group_name,
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "n_active": int(np.sum(active_mask)),
                    "target_mass_est": float(target_mass_estimate),
                    "global_ess": float(global_ess),
                    "local_ess": local_ess,
                    "abs_gap": abs(normalized_gap),
                    "se_bound": float(se_bound),
                    "is_planted_group": bool(group_name == PLANTED_GROUP),
                }
            )

    frame_out = pd.DataFrame.from_records(raw_rows)
    if frame_out.empty:
        return frame_out

    n_cells = len(frame_out)
    z_value = float(norm.ppf(1.0 - alpha / (2.0 * max(n_cells, 1))))
    frame_out["ci_half_width"] = z_value * frame_out["se_bound"]
    return frame_out


def _pick_largest(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values("abs_gap", ascending=False).iloc[0]


def _decision_from_row(
    *,
    scenario: str,
    method: str,
    shift_strength: float,
    seed: int,
    row: pd.Series | None,
    alarm: bool,
    abstained: bool,
    global_ess: float,
    n_cells: int,
    n_eligible_cells: int,
) -> AuditDecision:
    if row is None:
        return AuditDecision(
            scenario=scenario,
            method=method,
            shift_strength=float(shift_strength),
            seed=int(seed),
            alarm=bool(alarm),
            abstained=bool(abstained),
            selected_group="none",
            selected_bin="none",
            selected_abs_gap=float("nan"),
            selected_local_ess=float("nan"),
            selected_ci_half_width=float("nan"),
            global_ess=float(global_ess),
            n_cells=int(n_cells),
            n_eligible_cells=int(n_eligible_cells),
            selected_is_planted_group=False,
        )

    return AuditDecision(
        scenario=scenario,
        method=method,
        shift_strength=float(shift_strength),
        seed=int(seed),
        alarm=bool(alarm),
        abstained=bool(abstained),
        selected_group=str(row["group"]),
        selected_bin=str(row["bin"]),
        selected_abs_gap=float(row["abs_gap"]),
        selected_local_ess=float(row["local_ess"]),
        selected_ci_half_width=float(row["ci_half_width"]),
        global_ess=float(global_ess),
        n_cells=int(n_cells),
        n_eligible_cells=int(n_eligible_cells),
        selected_is_planted_group=bool(row["is_planted_group"]),
    )


def _audit_decisions(
    *,
    cell_frame: pd.DataFrame,
    scenario: str,
    shift_strength: float,
    seed: int,
    tolerance: float,
    global_ess_min: float,
    local_ess_min: float,
) -> list[AuditDecision]:
    if cell_frame.empty:
        return [
            _decision_from_row(
                scenario=scenario,
                method=method,
                shift_strength=shift_strength,
                seed=seed,
                row=None,
                alarm=False,
                abstained=True,
                global_ess=float("nan"),
                n_cells=0,
                n_eligible_cells=0,
            )
            for method in ["naive_max", "global_ess_gate", "local_ess_gate", "simultaneous_ci"]
        ]

    global_ess = float(cell_frame["global_ess"].iloc[0])
    n_cells = int(len(cell_frame))
    decisions: list[AuditDecision] = []

    naive_row = _pick_largest(cell_frame)
    decisions.append(
        _decision_from_row(
            scenario=scenario,
            method="naive_max",
            shift_strength=shift_strength,
            seed=seed,
            row=naive_row,
            alarm=bool(naive_row is not None and naive_row["abs_gap"] > tolerance),
            abstained=False,
            global_ess=global_ess,
            n_cells=n_cells,
            n_eligible_cells=n_cells,
        )
    )

    global_abstained = global_ess < global_ess_min
    decisions.append(
        _decision_from_row(
            scenario=scenario,
            method="global_ess_gate",
            shift_strength=shift_strength,
            seed=seed,
            row=naive_row,
            alarm=bool(
                not global_abstained and naive_row is not None and naive_row["abs_gap"] > tolerance
            ),
            abstained=global_abstained,
            global_ess=global_ess,
            n_cells=n_cells,
            n_eligible_cells=n_cells if not global_abstained else 0,
        )
    )

    local_frame = cell_frame.loc[cell_frame["local_ess"] >= local_ess_min].copy()
    local_row = _pick_largest(local_frame)
    decisions.append(
        _decision_from_row(
            scenario=scenario,
            method="local_ess_gate",
            shift_strength=shift_strength,
            seed=seed,
            row=local_row,
            alarm=bool(local_row is not None and local_row["abs_gap"] > tolerance),
            abstained=local_frame.empty,
            global_ess=global_ess,
            n_cells=n_cells,
            n_eligible_cells=int(len(local_frame)),
        )
    )

    certified_frame = cell_frame.loc[
        cell_frame["abs_gap"] - cell_frame["ci_half_width"] > tolerance
    ].copy()
    certified_row = _pick_largest(certified_frame)
    decisions.append(
        _decision_from_row(
            scenario=scenario,
            method="simultaneous_ci",
            shift_strength=shift_strength,
            seed=seed,
            row=certified_row if certified_row is not None else naive_row,
            alarm=bool(certified_row is not None),
            abstained=False,
            global_ess=global_ess,
            n_cells=n_cells,
            n_eligible_cells=n_cells,
        )
    )
    return decisions


def run_study(
    *,
    shifts: Iterable[float],
    seeds: Iterable[int],
    n_reference: int,
    n_target: int,
    n_features: int,
    n_bins: int,
    min_target_mass: float,
    tolerance: float,
    alpha: float,
    global_ess_min: float,
    local_ess_min: float,
    planted_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cell_frames: list[pd.DataFrame] = []
    decisions: list[AuditDecision] = []

    for shift_strength in shifts:
        for seed in seeds:
            dataset = make_covariate_shift_classification(
                n_samples_ref=n_reference,
                n_samples_target=n_target,
                n_features=n_features,
                shift_strength=float(shift_strength),
                random_state=int(seed),
            )
            weights = _oracle_covariate_shift_weights(
                dataset.X_ref,
                shift_strength=float(shift_strength),
                n_features=n_features,
            )
            weights = weights / np.mean(weights)

            scenarios = {
                "null": dataset.y_ref,
                "planted": _plant_subgroup_failure(
                    dataset.X_ref,
                    dataset.reference_predictions,
                    delta=planted_delta,
                    seed=int(seed),
                ),
            }
            for scenario, y_values in scenarios.items():
                cells = _cell_table(
                    scenario=scenario,
                    frame=dataset.X_ref,
                    y_true=y_values,
                    probabilities=dataset.reference_predictions,
                    weights=weights,
                    shift_strength=float(shift_strength),
                    seed=int(seed),
                    bins=bins,
                    min_target_mass=min_target_mass,
                    alpha=alpha,
                )
                if not cells.empty:
                    cell_frames.append(cells)
                decisions.extend(
                    _audit_decisions(
                        cell_frame=cells,
                        scenario=scenario,
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        tolerance=tolerance,
                        global_ess_min=global_ess_min,
                        local_ess_min=local_ess_min,
                    )
                )

    cell_table = pd.concat(cell_frames, ignore_index=True)
    decision_table = pd.DataFrame.from_records(asdict(decision) for decision in decisions)
    summary = (
        decision_table.groupby(["scenario", "method", "shift_strength"], as_index=False)
        .agg(
            alarm_rate=("alarm", "mean"),
            abstention_rate=("abstained", "mean"),
            planted_selection_rate=("selected_is_planted_group", "mean"),
            alarm_and_planted_rate=(
                "selected_is_planted_group",
                lambda values: float(
                    np.mean(
                        values.to_numpy(dtype=bool)
                        & decision_table.loc[values.index, "alarm"].to_numpy(dtype=bool)
                    )
                ),
            ),
            median_selected_gap=("selected_abs_gap", "median"),
            median_selected_local_ess=("selected_local_ess", "median"),
            median_global_ess=("global_ess", "median"),
            median_eligible_cells=("n_eligible_cells", "median"),
        )
        .sort_values(["scenario", "method", "shift_strength"], ignore_index=True)
    )
    return summary, decision_table, cell_table


def _write_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    cells: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "certified_audit_summary.csv", index=False)
    decisions.to_csv(output_dir / "certified_audit_decisions.csv", index=False)
    cells.to_csv(output_dir / "certified_audit_cells.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reference", type=int, default=1000)
    parser.add_argument("--n-target", type=int, default=1500)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=7)
    parser.add_argument("--min-target-mass", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--global-ess-min", type=float, default=50.0)
    parser.add_argument("--local-ess-min", type=float, default=20.0)
    parser.add_argument("--planted-delta", type=float, default=0.60)
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6])
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprint/research/results/certified_audit"),
    )
    args = parser.parse_args()

    summary, decisions, cells = run_study(
        shifts=args.shifts,
        seeds=range(args.n_seeds),
        n_reference=args.n_reference,
        n_target=args.n_target,
        n_features=args.n_features,
        n_bins=args.n_bins,
        min_target_mass=args.min_target_mass,
        tolerance=args.tolerance,
        alpha=args.alpha,
        global_ess_min=args.global_ess_min,
        local_ess_min=args.local_ess_min,
        planted_delta=args.planted_delta,
    )
    _write_outputs(
        output_dir=args.output_dir,
        summary=summary,
        decisions=decisions,
        cells=cells,
    )
    print("Certified audit summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
