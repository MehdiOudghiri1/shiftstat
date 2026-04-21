"""Adult Census case study for certified subgroup-bin auditing.

The case study uses real tabular covariates and labels from the Adult Census
income dataset. We create a deployment shift by sampling the target population
with higher probability for older, higher-hours, higher-education individuals.
Because assignment depends only on covariates, the induced source/target split
is a covariate-shift design with real labels available for retrospective
validation.

The audit itself uses only labeled source-audit data plus target covariates:
cross-fitted domain-classifier weights transport source residuals to the target
population, and target labels are used only to annotate whether selected alarms
match the held-out target evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shiftstat.metrics.weighted import compute_effective_sample_size

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research" / "results" / "adult_case_study"
DEFAULT_FIGURE = ROOT / "figures" / "adult_case_study.png"


def _load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    frame = data.frame.copy()
    target = frame.pop("class").astype(str).str.contains(">50K").to_numpy(dtype=int)
    frame = frame.replace("?", np.nan)
    # Keep the sampling weight out of the prediction/audit model. It is a survey
    # design variable rather than an operational covariate.
    if "fnlwgt" in frame.columns:
        frame = frame.drop(columns=["fnlwgt"])
    return frame, target


def _feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _feature_types(frame)
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def _make_predictor(frame: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _preprocessor(frame)),
            (
                "classifier",
                LogisticRegression(max_iter=3000, C=0.7, random_state=seed),
            ),
        ]
    )


def _deployment_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_train: int,
    n_source_audit: int,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    age = pd.to_numeric(frame["age"], errors="coerce").fillna(frame["age"].median()).to_numpy(float)
    hours = (
        pd.to_numeric(frame["hours-per-week"], errors="coerce")
        .fillna(frame["hours-per-week"].median())
        .to_numpy(float)
    )
    education = (
        pd.to_numeric(frame["education-num"], errors="coerce")
        .fillna(frame["education-num"].median())
        .to_numpy(float)
    )
    shift_score = (
        1.05 * (age - np.mean(age)) / np.std(age)
        + 0.80 * (hours - np.mean(hours)) / np.std(hours)
        + 0.55 * (education >= 13).astype(float)
        - 0.35 * (frame["sex"].astype(str).to_numpy() == "Female").astype(float)
    )
    target_probability = expit(shift_score)
    target_probability = target_probability / np.sum(target_probability)
    all_indices = np.arange(len(frame))
    target_index = rng.choice(all_indices, size=n_target, replace=False, p=target_probability)
    remaining = np.setdiff1d(all_indices, target_index, assume_unique=False)
    rng.shuffle(remaining)
    source_train_index = remaining[:n_train]
    source_audit_index = remaining[n_train : n_train + n_source_audit]
    return source_train_index, source_audit_index, target_index


def _domain_weights_crossfit(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    *,
    seed: int,
    n_folds: int,
    clip_max: float,
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
        model = Pipeline(
            [
                ("preprocessor", _preprocessor(combined)),
                (
                    "classifier",
                    LogisticRegression(max_iter=3000, C=1.0, random_state=seed + fold_index),
                ),
            ]
        )
        model.fit(combined, y_domain)
        heldout_scores = model.predict_proba(source.iloc[test_index])[:, 1]
        train_scores = model.predict_proba(combined)[:, 1]
        aucs.append(float(roc_auc_score(y_domain, train_scores)))
        prior_ref = len(train_source) / len(combined)
        prior_target = len(target) / len(combined)
        odds = np.clip(heldout_scores, 1e-6, 1 - 1e-6) / np.clip(
            1.0 - heldout_scores,
            1e-6,
            1.0,
        )
        fold_weights = odds * (prior_ref / prior_target)
        weights[test_index] = np.clip(fold_weights, 1e-3, clip_max)
    weights = weights / np.mean(weights)
    return weights, float(np.mean(aucs))


def _group_family(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    sex = frame["sex"].astype(str)
    race = frame["race"].astype(str)
    education = frame["education"].astype(str)
    workclass = frame["workclass"].astype(str)
    age = pd.to_numeric(frame["age"], errors="coerce").to_numpy(float)
    hours = pd.to_numeric(frame["hours-per-week"], errors="coerce").to_numpy(float)
    education_num = pd.to_numeric(frame["education-num"], errors="coerce").to_numpy(float)

    for value in ["Female", "Male"]:
        groups[f"sex={value}"] = (sex == value).to_numpy()
    for value in ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo"]:
        groups[f"race={value}"] = (race == value).to_numpy()
    for value in ["Bachelors", "Masters", "HS-grad", "Some-college"]:
        groups[f"education={value}"] = (education == value).to_numpy()
    for value in ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov"]:
        groups[f"workclass={value}"] = (workclass == value).to_numpy()

    for threshold in [30, 40, 50, 60]:
        groups[f"age>{threshold}"] = age > threshold
        groups[f"age<={threshold}"] = age <= threshold
    for threshold in [35, 45, 55]:
        groups[f"hours>{threshold}"] = hours > threshold
    groups["education-num>=13"] = education_num >= 13
    groups["Female & hours>45"] = (sex == "Female").to_numpy() & (hours > 45)
    groups["Male & hours>45"] = (sex == "Male").to_numpy() & (hours > 45)
    groups["age>50 & hours>45"] = (age > 50) & (hours > 45)
    groups["Female & education-num>=13"] = (sex == "Female").to_numpy() & (education_num >= 13)
    groups["Male & education-num>=13"] = (sex == "Male").to_numpy() & (education_num >= 13)
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
            target_mass_estimate = float(np.sum(local_weights) / len(source_weights))
            if target_mass_estimate < min_target_mass:
                continue
            local_ess = float(compute_effective_sample_size(local_weights))
            weighted_gap = float(np.average(source_residual[source_mask], weights=local_weights))
            if np.any(target_mask):
                target_gap = float(np.mean(target_residual[target_mask]))
                target_n = int(np.sum(target_mask))
            else:
                target_gap = float("nan")
                target_n = 0
            rows.append(
                {
                    "group": group_name,
                    "bin": f"[{lower:.2f}, {upper:.2f}]",
                    "weighted_gap": weighted_gap,
                    "abs_weighted_gap": abs(weighted_gap),
                    "target_gap": target_gap,
                    "abs_target_gap": abs(target_gap) if np.isfinite(target_gap) else float("nan"),
                    "local_ess": local_ess,
                    "estimated_target_mass": target_mass_estimate,
                    "source_n": int(np.sum(source_mask)),
                    "target_n": target_n,
                }
            )

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    m = len(table)
    table["ci_half_width"] = np.sqrt(
        np.log(2.0 * m / alpha) / (2.0 * np.maximum(table["local_ess"], 1e-12))
    )
    table["certified_excess"] = np.maximum(
        table["abs_weighted_gap"] - table["ci_half_width"],
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
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw={"width_ratios": [1.9, 1.0]})

    ax = axes[0]
    ax.hlines(
        y_positions,
        np.maximum(top["abs_weighted_gap"] - top["ci_half_width"], 0.0),
        top["abs_weighted_gap"] + top["ci_half_width"],
        color="#B7AA93",
        linewidth=3.2,
        alpha=0.8,
        label="Local simultaneous interval",
    )
    certified = top["certified_excess"] > tolerance
    colors = np.where(certified, "#2E7D59", "#A95C20")
    ax.scatter(
        top["abs_weighted_gap"],
        y_positions,
        s=58,
        color=colors,
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
        label="Held-out target gap",
    )
    ax.axvline(
        tolerance, color="#B3261E", linestyle="--", linewidth=1.8, label="Operational tolerance"
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Absolute subgroup-bin residual gap", fontsize=10)
    ax.set_title("A. Naive Adult Census alarms versus certification", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D2C4", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.barh(y_positions, top["local_ess"], color="#6F7F7A", alpha=0.58)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlabel("Local ESS (log scale)", fontsize=10)
    ax.set_title("B. The scary cells are locally thin", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D2C4", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Real-data audit: certification separates evidence from thin-support alarms",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")


def run_case_study(
    *,
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
    frame, target = _load_adult()
    source_train_index, source_audit_index, target_index = _deployment_split(
        frame,
        seed=seed,
        n_train=n_train,
        n_source_audit=n_source_audit,
        n_target=n_target,
    )
    X_train = frame.iloc[source_train_index].reset_index(drop=True)
    y_train = target[source_train_index]
    X_source = frame.iloc[source_audit_index].reset_index(drop=True)
    y_source = target[source_audit_index]
    X_target = frame.iloc[target_index].reset_index(drop=True)
    y_target = target[target_index]

    predictor = _make_predictor(frame, seed)
    predictor.fit(X_train, y_train)
    p_source = predictor.predict_proba(X_source)[:, 1]
    p_target = predictor.predict_proba(X_target)[:, 1]
    source_weights, domain_auc = _domain_weights_crossfit(
        X_source,
        X_target,
        seed=seed + 101,
        n_folds=n_folds,
        clip_max=clip_max,
    )
    table = _audit_table(
        X_source=X_source,
        y_source=y_source,
        p_source=p_source,
        source_weights=source_weights,
        X_target=X_target,
        y_target=y_target,
        p_target=p_target,
        n_bins=n_bins,
        min_target_mass=min_target_mass,
        alpha=alpha,
    )
    table["naive_alarm"] = table["abs_weighted_gap"] > tolerance
    table["certified_alarm"] = table["certified_excess"] > tolerance
    table["matches_target_direction"] = np.sign(table["weighted_gap"]) == np.sign(
        table["target_gap"]
    )

    summary = pd.DataFrame.from_records(
        [
            {
                "dataset": "Adult Census income",
                "n_train": int(n_train),
                "n_source_audit": int(n_source_audit),
                "n_target": int(n_target),
                "domain_auc": float(domain_auc),
                "global_ess": float(compute_effective_sample_size(source_weights)),
                "n_cells": int(len(table)),
                "naive_alarm_count": int(table["naive_alarm"].sum()),
                "certified_alarm_count": int(table["certified_alarm"].sum()),
                "top_naive_group": str(table.iloc[0]["group"]),
                "top_naive_bin": str(table.iloc[0]["bin"]),
                "top_naive_gap": float(table.iloc[0]["abs_weighted_gap"]),
                "top_naive_local_ess": float(table.iloc[0]["local_ess"]),
                "top_naive_ci_half_width": float(table.iloc[0]["ci_half_width"]),
                "top_naive_certified": bool(table.iloc[0]["certified_alarm"]),
                "top_naive_target_gap": float(table.iloc[0]["abs_target_gap"]),
            }
        ]
    )
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-train", type=int, default=16000)
    parser.add_argument("--n-source-audit", type=int, default=9000)
    parser.add_argument("--n-target", type=int, default=12000)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--min-target-mass", type=float, default=0.008)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--clip-max", type=float, default=25.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    summary, table = run_case_study(
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
    summary.to_csv(args.output_dir / "adult_case_summary.csv", index=False)
    table.to_csv(args.output_dir / "adult_case_cells.csv", index=False)
    _plot_case(table, output=args.figure_path, tolerance=args.tolerance)
    print("Adult case-study summary:")
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
                "ci_half_width",
                "certified_excess",
                "local_ess",
                "abs_target_gap",
                "naive_alarm",
                "certified_alarm",
            ],
        ].to_string(index=False)
    )
    print()
    print(f"Results written to {args.output_dir}")
    print(f"Figure written to {args.figure_path}")


if __name__ == "__main__":
    main()
