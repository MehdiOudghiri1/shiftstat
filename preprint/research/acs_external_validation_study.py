"""ACS/Folktables external-validation audit for certified subgroup alarms.

This case study uses the Folktables ACSIncome task as a public, high-stakes
public-service analogue. A model is trained and audited on labeled source-state
data, then evaluated for a target state using target covariates at audit time
and target labels only retrospectively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shiftstat.metrics.weighted import compute_effective_sample_size

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research" / "results" / "acs_external_validation"
DEFAULT_FIGURE = ROOT / "figures" / "acs_external_validation.png"

NUMERIC = ["AGEP", "WKHP"]
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]


def _load_state(state: str, *, year: int) -> tuple[pd.DataFrame, np.ndarray]:
    data_source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    raw = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_numpy(raw)
    frame = pd.DataFrame(features, columns=ACSIncome.features)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    labels = labels.astype(int)
    return frame, labels


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def _make_model(seed: int, *, c_value: float = 0.6) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2500, C=c_value, random_state=seed),
            ),
        ]
    )


def _domain_weights_crossfit(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    *,
    seed: int,
    n_folds: int,
    clip_max: float,
    c_value: float,
) -> tuple[np.ndarray, float]:
    source = X_source.reset_index(drop=True)
    target = X_target.reset_index(drop=True)
    weights = np.empty(len(source), dtype=float)
    aucs: list[float] = []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_index, (train_index, test_index) in enumerate(splitter.split(source)):
        train_source = source.iloc[train_index]
        combined = pd.concat([train_source, target], ignore_index=True)
        y_domain = np.concatenate(
            [np.zeros(len(train_source), dtype=int), np.ones(len(target), dtype=int)]
        )
        model = _make_model(seed + fold_index, c_value=c_value)
        model.fit(combined, y_domain)
        heldout_scores = model.predict_proba(source.iloc[test_index])[:, 1]
        train_scores = model.predict_proba(combined)[:, 1]
        aucs.append(float(roc_auc_score(y_domain, train_scores)))
        prior_source = len(train_source) / len(combined)
        prior_target = len(target) / len(combined)
        odds = np.clip(heldout_scores, 1e-6, 1 - 1e-6) / np.clip(
            1.0 - heldout_scores,
            1e-6,
            1.0,
        )
        fold_weights = odds * (prior_source / prior_target)
        weights[test_index] = np.clip(fold_weights, 1e-3, clip_max)
    weights = weights / np.mean(weights)
    return weights, float(np.mean(aucs))


def _group_family(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    age = frame["AGEP"].to_numpy(float)
    hours = frame["WKHP"].to_numpy(float)
    sex = frame["SEX"].to_numpy(float)
    race = frame["RAC1P"].to_numpy(float)
    school = frame["SCHL"].to_numpy(float)

    groups: dict[str, np.ndarray] = {
        "Female": sex == 2,
        "Male": sex == 1,
        "White": race == 1,
        "Black": race == 2,
        "Asian": race == 6,
        "Other race": ~np.isin(race, [1, 2, 6]),
        "Age>45": age > 45,
        "Age>60": age > 60,
        "Hours>40": hours > 40,
        "Hours>50": hours > 50,
        "Bachelor+": school >= 21,
        "Female & Hours>40": (sex == 2) & (hours > 40),
        "Male & Hours>40": (sex == 1) & (hours > 40),
        "Age>45 & Bachelor+": (age > 45) & (school >= 21),
        "Female & Bachelor+": (sex == 2) & (school >= 21),
        "Male & Bachelor+": (sex == 1) & (school >= 21),
    }
    return groups


def _bin_mask(probabilities: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if upper < 1.0:
        return (probabilities >= lower) & (probabilities < upper)
    return (probabilities >= lower) & (probabilities <= upper)


def _audit_table(
    *,
    X_source: pd.DataFrame,
    y_source: np.ndarray,
    p_source: np.ndarray,
    source_weights: np.ndarray,
    sensitivity_weights: list[np.ndarray],
    X_target: pd.DataFrame,
    y_target: np.ndarray,
    p_target: np.ndarray,
    n_bins: int,
    min_target_mass: float,
    alpha: float,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    source_groups = _group_family(X_source)
    target_groups = _group_family(X_target)
    source_residual = y_source.astype(float) - p_source.astype(float)
    target_residual = y_target.astype(float) - p_target.astype(float)
    rows: list[dict[str, Any]] = []

    for group_name, source_group_mask in source_groups.items():
        target_group_mask = target_groups[group_name]
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            source_mask = source_group_mask & _bin_mask(p_source, float(lower), float(upper))
            target_mask = target_group_mask & _bin_mask(p_target, float(lower), float(upper))
            if not np.any(source_mask):
                continue
            local_weights = source_weights[source_mask]
            estimated_target_mass = float(np.sum(local_weights) / len(source_weights))
            if estimated_target_mass < min_target_mass:
                continue
            local_ess = float(compute_effective_sample_size(local_weights))
            weighted_gap = float(np.average(source_residual[source_mask], weights=local_weights))
            target_gap = (
                float(np.mean(target_residual[target_mask])) if np.any(target_mask) else np.nan
            )

            rho_values = []
            for alt_weights in sensitivity_weights:
                alt_local = alt_weights[source_mask]
                delta = float(np.sum(np.abs(local_weights - alt_local)))
                denom = float(np.sum(alt_local) - delta)
                if denom > 0:
                    rho_values.append(2.0 * delta / denom)
            rho_sens = float(np.max(rho_values)) if rho_values else 1.0

            rows.append(
                {
                    "group": group_name,
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "weighted_gap": weighted_gap,
                    "abs_weighted_gap": abs(weighted_gap),
                    "target_gap": target_gap,
                    "abs_target_gap": abs(target_gap) if np.isfinite(target_gap) else np.nan,
                    "local_ess": local_ess,
                    "estimated_target_mass": estimated_target_mass,
                    "source_n": int(np.sum(source_mask)),
                    "target_n": int(np.sum(target_mask)),
                    "rho_sensitivity": rho_sens,
                }
            )

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    m = len(table)
    table["label_half_width"] = np.sqrt(
        np.log(2.0 * m / alpha) / (2.0 * np.maximum(table["local_ess"], 1e-12))
    )
    table["label_certified_excess"] = np.maximum(
        table["abs_weighted_gap"] - table["label_half_width"],
        0.0,
    )
    table["sensitivity_certified_excess"] = np.maximum(
        table["abs_weighted_gap"] - table["label_half_width"] - table["rho_sensitivity"],
        0.0,
    )
    return table.sort_values("abs_weighted_gap", ascending=False, ignore_index=True)


def _plot_case(table: pd.DataFrame, *, output: Path, tolerance: float) -> None:
    top = table.head(10).iloc[::-1].copy()
    labels = [f"{row.group}\n{row.bin}" for row in top.itertuples()]
    y_positions = np.arange(len(top))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#FCFAF5",
            "figure.facecolor": "#FCFAF5",
            "savefig.facecolor": "#FCFAF5",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 5.7), gridspec_kw={"width_ratios": [1.9, 0.9, 0.9]}
    )

    ax = axes[0]
    ax.hlines(
        y_positions,
        np.maximum(top["abs_weighted_gap"] - top["label_half_width"], 0.0),
        top["abs_weighted_gap"] + top["label_half_width"],
        color="#B7AA93",
        linewidth=3.0,
        alpha=0.82,
        label="Label-noise interval",
    )
    ax.scatter(
        top["abs_weighted_gap"],
        y_positions,
        s=58,
        color="#A95C20",
        zorder=3,
        label="Weighted audit gap",
    )
    ax.scatter(
        top["abs_target_gap"],
        y_positions,
        s=42,
        marker="D",
        color="#263238",
        alpha=0.78,
        label="Target-label gap",
    )
    ax.axvline(tolerance, color="#B3261E", linestyle="--", linewidth=1.8, label="Tolerance")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Absolute residual gap", fontsize=10)
    ax.set_title("A. Audit gaps", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D2C4", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.barh(y_positions, top["local_ess"], color="#6F7F7A", alpha=0.65)
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel("Local ESS", fontsize=10)
    ax.set_title("B. Local support", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D2C4", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.barh(y_positions, top["rho_sensitivity"], color="#7B6D8D", alpha=0.72)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel(r"Sensitivity $\widehat\rho_c$", fontsize=10)
    ax.set_title("C. Weight nuisance", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D2C4", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "ACS external validation",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")


def run_case_study(
    *,
    source_state: str,
    target_state: str,
    year: int,
    seed: int,
    n_train: int,
    n_source_audit: int,
    n_target: int,
    n_bins: int,
    min_target_mass: float,
    tolerance: float,
    alpha: float,
    clip_max: float,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_source_all, y_source_all = _load_state(source_state, year=year)
    X_target_all, y_target_all = _load_state(target_state, year=year)

    X_train, X_audit, y_train, y_audit = train_test_split(
        X_source_all,
        y_source_all,
        train_size=min(n_train, int(0.55 * len(X_source_all))),
        random_state=seed,
        stratify=y_source_all,
    )
    if len(X_audit) > n_source_audit:
        X_audit, _, y_audit, _ = train_test_split(
            X_audit,
            y_audit,
            train_size=n_source_audit,
            random_state=seed + 1,
            stratify=y_audit,
        )
    if len(X_target_all) > n_target:
        X_target, _, y_target, _ = train_test_split(
            X_target_all,
            y_target_all,
            train_size=n_target,
            random_state=seed + 2,
            stratify=y_target_all,
        )
    else:
        X_target, y_target = X_target_all, y_target_all

    predictor = _make_model(seed, c_value=0.7)
    predictor.fit(X_train, y_train)
    p_source = predictor.predict_proba(X_audit)[:, 1]
    p_target = predictor.predict_proba(X_target)[:, 1]

    source_weights, domain_auc = _domain_weights_crossfit(
        X_audit,
        X_target,
        seed=seed + 100,
        n_folds=n_folds,
        clip_max=clip_max,
        c_value=1.0,
    )
    sensitivity_weights = []
    for c_value, alt_clip in [(0.35, clip_max), (1.8, clip_max), (1.0, 10.0), (1.0, 40.0)]:
        alt_weights, _ = _domain_weights_crossfit(
            X_audit,
            X_target,
            seed=seed + int(c_value * 1000) + int(alt_clip),
            n_folds=n_folds,
            clip_max=alt_clip,
            c_value=c_value,
        )
        sensitivity_weights.append(alt_weights)

    table = _audit_table(
        X_source=X_audit.reset_index(drop=True),
        y_source=y_audit,
        p_source=p_source,
        source_weights=source_weights,
        sensitivity_weights=sensitivity_weights,
        X_target=X_target.reset_index(drop=True),
        y_target=y_target,
        p_target=p_target,
        n_bins=n_bins,
        min_target_mass=min_target_mass,
        alpha=alpha,
    )
    table["naive_alarm"] = table["abs_weighted_gap"] > tolerance
    table["label_certified_alarm"] = table["label_certified_excess"] > tolerance
    table["sensitivity_certified_alarm"] = table["sensitivity_certified_excess"] > tolerance

    top = table.iloc[0]
    summary = pd.DataFrame.from_records(
        [
            {
                "dataset": f"ACSIncome {source_state}->{target_state}",
                "source_state": source_state,
                "target_state": target_state,
                "year": year,
                "n_train": int(len(X_train)),
                "n_source_audit": int(len(X_audit)),
                "n_target": int(len(X_target)),
                "domain_auc": float(domain_auc),
                "global_ess": float(compute_effective_sample_size(source_weights)),
                "n_cells": int(len(table)),
                "naive_alarm_count": int(table["naive_alarm"].sum()),
                "label_certified_alarm_count": int(table["label_certified_alarm"].sum()),
                "sensitivity_certified_alarm_count": int(
                    table["sensitivity_certified_alarm"].sum()
                ),
                "top_group": str(top["group"]),
                "top_bin": str(top["bin"]),
                "top_gap": float(top["abs_weighted_gap"]),
                "top_local_ess": float(top["local_ess"]),
                "top_label_half_width": float(top["label_half_width"]),
                "top_rho_sensitivity": float(top["rho_sensitivity"]),
                "top_label_certified": bool(top["label_certified_alarm"]),
                "top_sensitivity_certified": bool(top["sensitivity_certified_alarm"]),
                "top_target_gap": float(top["abs_target_gap"]),
            }
        ]
    )
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", default="CA")
    parser.add_argument("--target-state", default="NY")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-train", type=int, default=30000)
    parser.add_argument("--n-source-audit", type=int, default=18000)
    parser.add_argument("--n-target", type=int, default=18000)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--min-target-mass", type=float, default=0.006)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--clip-max", type=float, default=25.0)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    summary, table = run_case_study(
        source_state=args.source_state,
        target_state=args.target_state,
        year=args.year,
        seed=args.seed,
        n_train=args.n_train,
        n_source_audit=args.n_source_audit,
        n_target=args.n_target,
        n_bins=args.n_bins,
        min_target_mass=args.min_target_mass,
        tolerance=args.tolerance,
        alpha=args.alpha,
        clip_max=args.clip_max,
        n_folds=args.n_folds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "acs_external_validation_summary.csv", index=False)
    table.to_csv(args.output_dir / "acs_external_validation_cells.csv", index=False)
    _plot_case(table, output=args.figure_path, tolerance=args.tolerance)

    print("ACS external-validation summary:")
    print(summary.to_string(index=False))
    print()
    print("Top audited cells:")
    print(
        table.loc[
            :9,
            [
                "group",
                "bin",
                "abs_weighted_gap",
                "label_half_width",
                "rho_sensitivity",
                "sensitivity_certified_excess",
                "local_ess",
                "abs_target_gap",
                "naive_alarm",
                "label_certified_alarm",
                "sensitivity_certified_alarm",
            ],
        ].to_string(index=False)
    )
    print()
    print(f"Results written to {args.output_dir}")
    print(f"Figure written to {args.figure_path}")


if __name__ == "__main__":
    main()
