"""Adaptive subgroup-search experiment for certified auditing.

The fixed-family simultaneous band remains valid after selecting the worst cell
from that family. Practitioners often do something more adaptive: they inspect
many threshold slices, refine intersections, and then report the worst-looking
subgroup. This script compares same-data search with a simple, defensible
protocol: discover the candidate cell on one labeled source split and certify it
on an independent audit split.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from certified_audit_study import PLANTED_GROUP, _plant_subgroup_failure
from local_ess_study import _active_bin_mask, _oracle_covariate_shift_weights

from shiftstat.datasets.synthetic import make_covariate_shift_classification
from shiftstat.metrics.weighted import compute_effective_sample_size


@dataclass(frozen=True)
class SearchDecision:
    scenario: str
    method: str
    shift_strength: float
    seed: int
    alarm: bool
    selected_group: str
    selected_bin: str
    selected_abs_gap: float
    selected_local_ess: float
    selected_radius: float
    selected_is_planted_group: bool
    n_search_cells: int


def _rich_group_family(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Create a larger threshold/intersection class for adaptive search."""

    groups: dict[str, np.ndarray] = {}
    feature_names = [name for name in frame.columns if name.startswith("x")]
    thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    for feature in feature_names:
        values = frame[feature].to_numpy(dtype=float)
        for threshold in thresholds:
            groups[f"{feature}>{threshold:g}"] = values > threshold
            groups[f"{feature}<{threshold:g}"] = values < threshold

    intersection_features = feature_names[:4]
    for left_index, left in enumerate(intersection_features):
        for right in intersection_features[left_index + 1 :]:
            left_values = frame[left].to_numpy(dtype=float)
            right_values = frame[right].to_numpy(dtype=float)
            for threshold in [0.0, 0.5, 1.0]:
                groups[f"{left}>{threshold:g} & {right}>{threshold:g}"] = (
                    left_values > threshold
                ) & (right_values > threshold)
    return groups


def _cell_table(
    *,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    min_target_mass: float,
    alpha: float,
    family_size: int | None = None,
    only_cell: tuple[str, float, float] | None = None,
) -> pd.DataFrame:
    residual = y_true.astype(float) - probabilities.astype(float)
    groups = _rich_group_family(frame)
    rows: list[dict[str, object]] = []

    if only_cell is not None:
        selected_group, selected_lower, selected_upper = only_cell
        if selected_group not in groups:
            return pd.DataFrame()
        group_items = [(selected_group, groups[selected_group])]
        bin_pairs = [(selected_lower, selected_upper)]
    else:
        group_items = list(groups.items())
        bin_pairs = list(zip(bins[:-1], bins[1:], strict=True))

    for group_name, group_mask in group_items:
        for lower, upper in bin_pairs:
            bin_mask = _active_bin_mask(probabilities, lower=float(lower), upper=float(upper))
            active_mask = group_mask & bin_mask
            if not np.any(active_mask):
                continue
            local_weights = weights[active_mask]
            target_mass_estimate = float(np.sum(local_weights) / len(weights))
            if target_mass_estimate < min_target_mass:
                continue
            local_ess = float(compute_effective_sample_size(local_weights))
            normalized_gap = float(np.average(residual[active_mask], weights=local_weights))
            rows.append(
                {
                    "group": group_name,
                    "lower": float(lower),
                    "upper": float(upper),
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "abs_gap": abs(normalized_gap),
                    "local_ess": local_ess,
                    "target_mass_est": target_mass_estimate,
                    "is_planted_group": bool(group_name == PLANTED_GROUP),
                }
            )

    frame_out = pd.DataFrame.from_records(rows)
    if frame_out.empty:
        return frame_out
    m = family_size if family_size is not None else len(frame_out)
    frame_out["radius"] = np.sqrt(
        np.log(2.0 * max(m, 1) / alpha) / (2.0 * np.maximum(frame_out["local_ess"], 1e-12))
    )
    return frame_out


def _pick_largest(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values("abs_gap", ascending=False).iloc[0]


def _pick_certified(frame: pd.DataFrame, *, tolerance: float) -> pd.Series | None:
    if frame.empty:
        return None
    certified = frame.loc[frame["abs_gap"] - frame["radius"] > tolerance].copy()
    if certified.empty:
        return None
    return certified.sort_values("abs_gap", ascending=False).iloc[0]


def _decision(
    *,
    scenario: str,
    method: str,
    shift_strength: float,
    seed: int,
    row: pd.Series | None,
    alarm: bool,
    n_search_cells: int,
) -> SearchDecision:
    if row is None:
        return SearchDecision(
            scenario=scenario,
            method=method,
            shift_strength=float(shift_strength),
            seed=int(seed),
            alarm=False,
            selected_group="none",
            selected_bin="none",
            selected_abs_gap=float("nan"),
            selected_local_ess=float("nan"),
            selected_radius=float("nan"),
            selected_is_planted_group=False,
            n_search_cells=int(n_search_cells),
        )
    return SearchDecision(
        scenario=scenario,
        method=method,
        shift_strength=float(shift_strength),
        seed=int(seed),
        alarm=bool(alarm),
        selected_group=str(row["group"]),
        selected_bin=str(row["bin"]),
        selected_abs_gap=float(row["abs_gap"]),
        selected_local_ess=float(row["local_ess"]),
        selected_radius=float(row["radius"]),
        selected_is_planted_group=bool(row["is_planted_group"]),
        n_search_cells=int(n_search_cells),
    )


def _subset_dataset(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame.iloc[indices].reset_index(drop=True),
        y_true[indices],
        probabilities[indices],
        weights[indices],
    )


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
    planted_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    decisions: list[SearchDecision] = []

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
            rng = np.random.RandomState(int(seed) + 17)
            permutation = rng.permutation(len(dataset.X_ref))
            midpoint = len(permutation) // 2
            discovery_index = permutation[:midpoint]
            audit_index = permutation[midpoint:]

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
                full_cells = _cell_table(
                    frame=dataset.X_ref,
                    y_true=y_values,
                    probabilities=dataset.reference_predictions,
                    weights=weights,
                    bins=bins,
                    min_target_mass=min_target_mass,
                    alpha=alpha,
                )
                n_search_cells = len(full_cells)
                same_row = _pick_largest(full_cells)
                decisions.append(
                    _decision(
                        scenario=scenario,
                        method="same_data_search",
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        row=same_row,
                        alarm=bool(same_row is not None and same_row["abs_gap"] > tolerance),
                        n_search_cells=n_search_cells,
                    )
                )
                simultaneous_row = _pick_certified(full_cells, tolerance=tolerance)
                decisions.append(
                    _decision(
                        scenario=scenario,
                        method="full_family_simultaneous",
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        row=simultaneous_row if simultaneous_row is not None else same_row,
                        alarm=bool(simultaneous_row is not None),
                        n_search_cells=n_search_cells,
                    )
                )

                discovery = _subset_dataset(
                    dataset.X_ref,
                    y_values,
                    dataset.reference_predictions,
                    weights,
                    discovery_index,
                )
                audit = _subset_dataset(
                    dataset.X_ref,
                    y_values,
                    dataset.reference_predictions,
                    weights,
                    audit_index,
                )
                discovery_cells = _cell_table(
                    frame=discovery[0],
                    y_true=discovery[1],
                    probabilities=discovery[2],
                    weights=discovery[3],
                    bins=bins,
                    min_target_mass=min_target_mass,
                    alpha=alpha,
                )
                selected = _pick_largest(discovery_cells)
                if selected is None:
                    audit_selected = pd.DataFrame()
                else:
                    audit_selected = _cell_table(
                        frame=audit[0],
                        y_true=audit[1],
                        probabilities=audit[2],
                        weights=audit[3],
                        bins=bins,
                        min_target_mass=0.0,
                        alpha=alpha,
                        family_size=1,
                        only_cell=(
                            str(selected["group"]),
                            float(selected["lower"]),
                            float(selected["upper"]),
                        ),
                    )
                audit_row = _pick_largest(audit_selected)
                decisions.append(
                    _decision(
                        scenario=scenario,
                        method="split_search_naive",
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        row=audit_row,
                        alarm=bool(audit_row is not None and audit_row["abs_gap"] > tolerance),
                        n_search_cells=len(discovery_cells),
                    )
                )
                decisions.append(
                    _decision(
                        scenario=scenario,
                        method="split_search_certified",
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        row=audit_row,
                        alarm=bool(
                            audit_row is not None
                            and audit_row["abs_gap"] - audit_row["radius"] > tolerance
                        ),
                        n_search_cells=len(discovery_cells),
                    )
                )

    decision_table = pd.DataFrame.from_records(asdict(decision) for decision in decisions)
    summary = (
        decision_table.groupby(["scenario", "method", "shift_strength"], as_index=False)
        .agg(
            alarm_rate=("alarm", "mean"),
            planted_alarm_rate=(
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
            median_selected_radius=("selected_radius", "median"),
            median_search_cells=("n_search_cells", "median"),
        )
        .sort_values(["scenario", "method", "shift_strength"], ignore_index=True)
    )
    return summary, decision_table


def _write_outputs(*, output_dir: Path, summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "adaptive_search_summary.csv", index=False)
    decisions.to_csv(output_dir / "adaptive_search_decisions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reference", type=int, default=1200)
    parser.add_argument("--n-target", type=int, default=1500)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=7)
    parser.add_argument("--min-target-mass", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--planted-delta", type=float, default=0.60)
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.8, 1.2])
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprint/research/results/adaptive_search"),
    )
    args = parser.parse_args()

    summary, decisions = run_study(
        shifts=args.shifts,
        seeds=range(args.n_seeds),
        n_reference=args.n_reference,
        n_target=args.n_target,
        n_features=args.n_features,
        n_bins=args.n_bins,
        min_target_mass=args.min_target_mass,
        tolerance=args.tolerance,
        alpha=args.alpha,
        planted_delta=args.planted_delta,
    )
    _write_outputs(output_dir=args.output_dir, summary=summary, decisions=decisions)
    print("Adaptive-search summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
