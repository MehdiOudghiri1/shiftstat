"""Estimated-weight stress test for certified subgroup-bin auditing.

This experiment keeps the synthetic covariate-shift benchmark because the
oracle density ratio is known. That lets us separate two uncertainty sources:

* label noise inside a subgroup-bin cell, governed by local ESS;
* nuisance error from replacing the oracle ratio w with an estimate w_hat.

The nuisance radius used here is intentionally transparent. For a cell c,
let Delta_c = sum_i |w_hat_i - w_i| A_c(X_i). Since residuals lie in [-1, 1],
the ratio estimator changes by at most

    2 Delta_c / (sum_i w_i A_c(X_i) - Delta_c)

whenever the denominator is positive. In real applications w is unknown, but
the synthetic setting lets us evaluate whether this is the right failure mode
to account for.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from certified_audit_study import PLANTED_GROUP, _plant_subgroup_failure
from local_ess_study import (
    _active_bin_mask,
    _default_group_family,
    _oracle_covariate_shift_weights,
)
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shiftstat.datasets.synthetic import make_covariate_shift_classification
from shiftstat.metrics.weighted import compute_effective_sample_size


@dataclass(frozen=True)
class EstimatedWeightDecision:
    scenario: str
    weight_mode: str
    method: str
    shift_strength: float
    seed: int
    alarm: bool
    selected_group: str
    selected_bin: str
    selected_abs_gap: float
    selected_local_ess: float
    selected_label_radius: float
    selected_weight_radius: float
    global_ess: float
    weight_rmse: float
    log_weight_corr: float
    selected_is_planted_group: bool


def _domain_classifier_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=3000, random_state=seed),
            ),
        ]
    )


def _weights_from_domain_probabilities(
    probabilities: np.ndarray,
    *,
    prior_ref: float,
    prior_target: float,
    clip_max: float,
) -> np.ndarray:
    target_probability = np.clip(probabilities.astype(float), 1e-6, 1.0 - 1e-6)
    odds = target_probability / (1.0 - target_probability)
    weights = odds * (prior_ref / prior_target)
    weights = np.clip(weights, 1e-3, clip_max)
    return weights / np.mean(weights)


def _estimate_logistic_weights(
    X_ref: pd.DataFrame,
    X_target: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    clip_max: float,
    n_folds: int,
) -> np.ndarray:
    if mode == "logistic_plugin":
        combined = pd.concat([X_ref, X_target], ignore_index=True)
        y_domain = np.concatenate(
            [np.zeros(len(X_ref), dtype=int), np.ones(len(X_target), dtype=int)]
        )
        model = _domain_classifier_pipeline(seed)
        model.fit(combined.to_numpy(dtype=float), y_domain)
        scores = model.predict_proba(X_ref.to_numpy(dtype=float))[:, 1]
        return _weights_from_domain_probabilities(
            scores,
            prior_ref=len(X_ref) / len(combined),
            prior_target=len(X_target) / len(combined),
            clip_max=clip_max,
        )

    if mode != "logistic_crossfit":
        raise ValueError(f"Unsupported estimated weight mode: {mode}")

    weights = np.empty(len(X_ref), dtype=float)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ref_matrix = X_ref.to_numpy(dtype=float)
    target_matrix = X_target.to_numpy(dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(ref_matrix)):
        train_ref = ref_matrix[train_index]
        combined = np.vstack([train_ref, target_matrix])
        y_domain = np.concatenate(
            [np.zeros(len(train_ref), dtype=int), np.ones(len(target_matrix), dtype=int)]
        )
        model = _domain_classifier_pipeline(seed + fold_index + 1)
        model.fit(combined, y_domain)
        scores = model.predict_proba(ref_matrix[test_index])[:, 1]
        fold_weights = _weights_from_domain_probabilities(
            scores,
            prior_ref=len(train_ref) / len(combined),
            prior_target=len(target_matrix) / len(combined),
            clip_max=clip_max,
        )
        weights[test_index] = fold_weights
    return weights / np.mean(weights)


def _cell_table(
    *,
    scenario: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    oracle_weights: np.ndarray,
    shift_strength: float,
    seed: int,
    bins: np.ndarray,
    min_target_mass: float,
    alpha: float,
) -> pd.DataFrame:
    residual = y_true.astype(float) - probabilities.astype(float)
    groups = _default_group_family(frame)
    global_ess = compute_effective_sample_size(weights)
    rows: list[dict[str, object]] = []

    for group_name, group_mask in groups.items():
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
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
            oracle_denominator = float(np.sum(oracle_weights[active_mask]))
            delta_weight = float(np.sum(np.abs(weights[active_mask] - oracle_weights[active_mask])))
            if oracle_denominator > delta_weight:
                weight_radius = min(1.0, 2.0 * delta_weight / (oracle_denominator - delta_weight))
            else:
                weight_radius = 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "shift_strength": float(shift_strength),
                    "seed": int(seed),
                    "group": group_name,
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "target_mass_est": target_mass_estimate,
                    "global_ess": float(global_ess),
                    "local_ess": local_ess,
                    "abs_gap": abs(normalized_gap),
                    "weight_radius": weight_radius,
                    "is_planted_group": bool(group_name == PLANTED_GROUP),
                }
            )

    cell_frame = pd.DataFrame.from_records(rows)
    if cell_frame.empty:
        return cell_frame
    n_cells = len(cell_frame)
    z_value = float(norm.ppf(1.0 - alpha / (2.0 * max(n_cells, 1))))
    cell_frame["label_radius"] = z_value * 0.5 / np.sqrt(np.maximum(cell_frame["local_ess"], 1e-12))
    cell_frame["total_radius"] = cell_frame["label_radius"] + cell_frame["weight_radius"]
    return cell_frame


def _pick_largest(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values("abs_gap", ascending=False).iloc[0]


def _pick_certified(
    frame: pd.DataFrame, *, tolerance: float, radius_column: str
) -> pd.Series | None:
    if frame.empty:
        return None
    certified = frame.loc[frame["abs_gap"] - frame[radius_column] > tolerance].copy()
    if certified.empty:
        return None
    return certified.sort_values("abs_gap", ascending=False).iloc[0]


def _decision(
    *,
    scenario: str,
    weight_mode: str,
    method: str,
    shift_strength: float,
    seed: int,
    row: pd.Series | None,
    alarm: bool,
    global_ess: float,
    weight_rmse: float,
    log_weight_corr: float,
) -> EstimatedWeightDecision:
    if row is None:
        return EstimatedWeightDecision(
            scenario=scenario,
            weight_mode=weight_mode,
            method=method,
            shift_strength=float(shift_strength),
            seed=int(seed),
            alarm=False,
            selected_group="none",
            selected_bin="none",
            selected_abs_gap=float("nan"),
            selected_local_ess=float("nan"),
            selected_label_radius=float("nan"),
            selected_weight_radius=float("nan"),
            global_ess=float(global_ess),
            weight_rmse=float(weight_rmse),
            log_weight_corr=float(log_weight_corr),
            selected_is_planted_group=False,
        )
    return EstimatedWeightDecision(
        scenario=scenario,
        weight_mode=weight_mode,
        method=method,
        shift_strength=float(shift_strength),
        seed=int(seed),
        alarm=bool(alarm),
        selected_group=str(row["group"]),
        selected_bin=str(row["bin"]),
        selected_abs_gap=float(row["abs_gap"]),
        selected_local_ess=float(row["local_ess"]),
        selected_label_radius=float(row["label_radius"]),
        selected_weight_radius=float(row["weight_radius"]),
        global_ess=float(global_ess),
        weight_rmse=float(weight_rmse),
        log_weight_corr=float(log_weight_corr),
        selected_is_planted_group=bool(row["is_planted_group"]),
    )


def _decisions_for_cells(
    *,
    cell_frame: pd.DataFrame,
    scenario: str,
    weight_mode: str,
    shift_strength: float,
    seed: int,
    tolerance: float,
    weight_rmse: float,
    log_weight_corr: float,
) -> list[EstimatedWeightDecision]:
    if cell_frame.empty:
        return []
    global_ess = float(cell_frame["global_ess"].iloc[0])
    naive_row = _pick_largest(cell_frame)
    label_ci_row = _pick_certified(cell_frame, tolerance=tolerance, radius_column="label_radius")
    nuisance_ci_row = _pick_certified(cell_frame, tolerance=tolerance, radius_column="total_radius")
    return [
        _decision(
            scenario=scenario,
            weight_mode=weight_mode,
            method="naive_max",
            shift_strength=shift_strength,
            seed=seed,
            row=naive_row,
            alarm=bool(naive_row is not None and naive_row["abs_gap"] > tolerance),
            global_ess=global_ess,
            weight_rmse=weight_rmse,
            log_weight_corr=log_weight_corr,
        ),
        _decision(
            scenario=scenario,
            weight_mode=weight_mode,
            method="label_ci",
            shift_strength=shift_strength,
            seed=seed,
            row=label_ci_row if label_ci_row is not None else naive_row,
            alarm=bool(label_ci_row is not None),
            global_ess=global_ess,
            weight_rmse=weight_rmse,
            log_weight_corr=log_weight_corr,
        ),
        _decision(
            scenario=scenario,
            weight_mode=weight_mode,
            method="nuisance_ci",
            shift_strength=shift_strength,
            seed=seed,
            row=nuisance_ci_row if nuisance_ci_row is not None else naive_row,
            alarm=bool(nuisance_ci_row is not None),
            global_ess=global_ess,
            weight_rmse=weight_rmse,
            log_weight_corr=log_weight_corr,
        ),
    ]


def run_study(
    *,
    shifts: Iterable[float],
    seeds: Iterable[int],
    weight_modes: Iterable[str],
    n_reference: int,
    n_target: int,
    n_features: int,
    n_bins: int,
    min_target_mass: float,
    tolerance: float,
    alpha: float,
    planted_delta: float,
    clip_max: float,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    all_cells: list[pd.DataFrame] = []
    decisions: list[EstimatedWeightDecision] = []

    for shift_strength in shifts:
        for seed in seeds:
            dataset = make_covariate_shift_classification(
                n_samples_ref=n_reference,
                n_samples_target=n_target,
                n_features=n_features,
                shift_strength=float(shift_strength),
                random_state=int(seed),
            )
            oracle_weights = _oracle_covariate_shift_weights(
                dataset.X_ref,
                shift_strength=float(shift_strength),
                n_features=n_features,
            )
            oracle_weights = oracle_weights / np.mean(oracle_weights)

            scenarios = {
                "null": dataset.y_ref,
                "planted": _plant_subgroup_failure(
                    dataset.X_ref,
                    dataset.reference_predictions,
                    delta=planted_delta,
                    seed=int(seed),
                ),
            }
            for weight_mode in weight_modes:
                if weight_mode == "oracle":
                    weights = oracle_weights.copy()
                else:
                    weights = _estimate_logistic_weights(
                        dataset.X_ref,
                        dataset.X_target,
                        mode=weight_mode,
                        seed=int(seed),
                        clip_max=clip_max,
                        n_folds=n_folds,
                    )
                weight_rmse = float(np.sqrt(np.mean((weights - oracle_weights) ** 2)))
                log_weight_corr = float(
                    np.corrcoef(np.log1p(weights), np.log1p(oracle_weights))[0, 1]
                )
                for scenario, y_values in scenarios.items():
                    cells = _cell_table(
                        scenario=scenario,
                        frame=dataset.X_ref,
                        y_true=y_values,
                        probabilities=dataset.reference_predictions,
                        weights=weights,
                        oracle_weights=oracle_weights,
                        shift_strength=float(shift_strength),
                        seed=int(seed),
                        bins=bins,
                        min_target_mass=min_target_mass,
                        alpha=alpha,
                    )
                    if not cells.empty:
                        cells.insert(1, "weight_mode", weight_mode)
                        all_cells.append(cells)
                    decisions.extend(
                        _decisions_for_cells(
                            cell_frame=cells,
                            scenario=scenario,
                            weight_mode=weight_mode,
                            shift_strength=float(shift_strength),
                            seed=int(seed),
                            tolerance=tolerance,
                            weight_rmse=weight_rmse,
                            log_weight_corr=log_weight_corr,
                        )
                    )

    cell_table = pd.concat(all_cells, ignore_index=True)
    decision_table = pd.DataFrame.from_records(asdict(decision) for decision in decisions)
    summary = (
        decision_table.groupby(
            ["scenario", "weight_mode", "method", "shift_strength"], as_index=False
        )
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
            median_label_radius=("selected_label_radius", "median"),
            median_weight_radius=("selected_weight_radius", "median"),
            median_global_ess=("global_ess", "median"),
            median_weight_rmse=("weight_rmse", "median"),
            median_log_weight_corr=("log_weight_corr", "median"),
        )
        .sort_values(["scenario", "weight_mode", "method", "shift_strength"], ignore_index=True)
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
    summary.to_csv(output_dir / "estimated_weight_summary.csv", index=False)
    decisions.to_csv(output_dir / "estimated_weight_decisions.csv", index=False)
    cells.to_csv(output_dir / "estimated_weight_cells.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-reference", type=int, default=1000)
    parser.add_argument("--n-target", type=int, default=1500)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=7)
    parser.add_argument("--min-target-mass", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--planted-delta", type=float, default=0.60)
    parser.add_argument("--clip-max", type=float, default=20.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.8, 1.2])
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        default=["oracle", "logistic_plugin", "logistic_crossfit"],
        choices=["oracle", "logistic_plugin", "logistic_crossfit"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprint/research/results/estimated_weights"),
    )
    args = parser.parse_args()

    summary, decisions, cells = run_study(
        shifts=args.shifts,
        seeds=range(args.n_seeds),
        weight_modes=args.weight_modes,
        n_reference=args.n_reference,
        n_target=args.n_target,
        n_features=args.n_features,
        n_bins=args.n_bins,
        min_target_mass=args.min_target_mass,
        tolerance=args.tolerance,
        alpha=args.alpha,
        planted_delta=args.planted_delta,
        clip_max=args.clip_max,
        n_folds=args.n_folds,
    )
    _write_outputs(output_dir=args.output_dir, summary=summary, decisions=decisions, cells=cells)
    print("Estimated-weight summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
