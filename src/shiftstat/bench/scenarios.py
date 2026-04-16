"""Scenario definitions for reproducible reliability benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shiftstat.datasets import (
    make_configurable_shift_classification,
    make_covariate_shift_classification,
    make_hidden_subgroup_shift_classification,
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One concrete benchmark case instantiated for a single seed."""

    case_id: str
    case_label: str
    family: str
    seed: int
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    X_ref: Any
    y_ref: np.ndarray
    X_target: Any
    y_target: np.ndarray
    estimator: Any
    categorical_features: list[str | int] | None = None
    subgroup_features: list[str | int] | None = None
    intersectional_features: list[tuple[str | int, ...]] | None = None
    reference_predictions: np.ndarray | None = None
    target_predictions: np.ndarray | None = None
    case_value: float | str | None = None


@dataclass(frozen=True)
class BenchmarkScenario:
    """Serializable benchmark scenario definition."""

    name: str
    family: str
    description: str
    seeds: list[int]
    case_definitions: list[dict[str, Any]]
    baseline_names: list[str]
    publication_metrics: list[str]
    case_builder: Callable[[dict[str, Any], int], BenchmarkCase]
    x_axis_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_case(self, case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        """Build one seeded benchmark case."""

        return self.case_builder(case_definition, seed)

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable scenario description."""

        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "seeds": self.seeds,
            "case_definitions": self.case_definitions,
            "baseline_names": self.baseline_names,
            "publication_metrics": self.publication_metrics,
            "x_axis_label": self.x_axis_label,
            "metadata": self.metadata,
        }


def make_covariate_shift_sweep_scenario(
    *,
    name: str = "covariate_shift_sweep",
    severities: list[float] | None = None,
    seeds: list[int] | None = None,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    n_features: int = 6,
    weighting_method: str = "domain_classifier",
    recalibration_method: str = "temperature",
    target_coverage: float = 0.8,
    baseline_names: list[str] | None = None,
) -> BenchmarkScenario:
    """Create a severity-sweep scenario over covariate shift strength."""

    severity_values = severities or [0.2, 0.6, 1.0, 1.4]
    scenario_seeds = seeds or [7, 19, 43]
    case_definitions = [
        {
            "case_id": f"severity_{severity:.2f}".replace(".", "_"),
            "case_label": f"severity={severity:.2f}",
            "severity": float(severity),
            "case_value": float(severity),
        }
        for severity in severity_values
    ]

    def _builder(case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        data = make_covariate_shift_classification(
            n_samples_ref=n_samples_ref,
            n_samples_target=n_samples_target,
            n_features=n_features,
            shift_strength=float(case_definition["severity"]),
            random_state=seed,
        )
        return BenchmarkCase(
            case_id=str(case_definition["case_id"]),
            case_label=str(case_definition["case_label"]),
            family="covariate_shift_sweep",
            seed=seed,
            parameters={"severity": float(case_definition["severity"])},
            metadata={
                "weighting_method": weighting_method,
                "recalibration_method": recalibration_method,
                "target_coverage": target_coverage,
            },
            X_ref=data.X_ref,
            y_ref=data.y_ref,
            X_target=data.X_target,
            y_target=data.y_target,
            estimator=LogisticRegression(max_iter=3000),
            subgroup_features=["x0", "x1"],
            reference_predictions=data.reference_predictions,
            target_predictions=data.target_predictions,
            case_value=float(case_definition["case_value"]),
        )

    return BenchmarkScenario(
        name=name,
        family="covariate_shift_sweep",
        description="Severity sweep over covariate shift for reliability and selective baselines.",
        seeds=scenario_seeds,
        case_definitions=case_definitions,
        baseline_names=baseline_names
        or [
            "raw_model",
            "weighting_only",
            "recalibration_only",
            "weighting_and_recalibration",
            "confidence_abstention",
        ],
        publication_metrics=["delta_ece", "delta_accuracy", "target_risk_reduction"],
        case_builder=_builder,
        x_axis_label="Shift severity",
        metadata={
            "weighting_method": weighting_method,
            "recalibration_method": recalibration_method,
            "target_coverage": target_coverage,
        },
    )


def make_subgroup_degradation_scenario(
    *,
    name: str = "subgroup_degradation_benchmark",
    patterns: list[str] | None = None,
    pattern_strength: float = 1.0,
    seeds: list[int] | None = None,
    n_samples_ref: int = 600,
    n_samples_target: int = 600,
    weighting_method: str = "domain_classifier",
    recalibration_method: str = "temperature",
    target_coverage: float = 0.8,
    baseline_names: list[str] | None = None,
) -> BenchmarkScenario:
    """Create a hidden-subgroup-failure benchmark scenario."""

    pattern_values = patterns or [
        "masked_subgroup_shift",
        "minority_subgroup_degradation",
    ]
    scenario_seeds = seeds or [11, 23, 37]
    case_definitions = [
        {
            "case_id": pattern,
            "case_label": pattern.replace("_", " "),
            "pattern": pattern,
            "case_value": index,
        }
        for index, pattern in enumerate(pattern_values)
    ]

    def _builder(case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        data = make_hidden_subgroup_shift_classification(
            n_samples_ref=n_samples_ref,
            n_samples_target=n_samples_target,
            pattern=str(case_definition["pattern"]),
            pattern_strength=pattern_strength,
            random_state=seed,
        )
        return BenchmarkCase(
            case_id=str(case_definition["case_id"]),
            case_label=str(case_definition["case_label"]),
            family="subgroup_degradation",
            seed=seed,
            parameters={"pattern": str(case_definition["pattern"])},
            metadata={
                "weighting_method": weighting_method,
                "recalibration_method": recalibration_method,
                "target_coverage": target_coverage,
            },
            X_ref=data.X_ref,
            y_ref=data.y_ref,
            X_target=data.X_target,
            y_target=data.y_target,
            estimator=_mixed_estimator(
                categorical_features=["region", "channel"],
                continuous_features=["score", "load", "signal"],
            ),
            categorical_features=["region", "channel"],
            subgroup_features=["region", "channel", "score", "load"],
            intersectional_features=[("region", "channel")],
            reference_predictions=data.reference_predictions,
            target_predictions=data.target_predictions,
            case_value=str(case_definition["case_label"]),
        )

    return BenchmarkScenario(
        name=name,
        family="subgroup_degradation",
        description=(
            "Hidden subgroup failure benchmarks where aggregate metrics can look "
            "deceptively stable."
        ),
        seeds=scenario_seeds,
        case_definitions=case_definitions,
        baseline_names=baseline_names
        or [
            "raw_model",
            "subgroup_audit",
            "confidence_abstention",
            "weighted_confidence_abstention",
        ],
        publication_metrics=[
            "worst_group_accuracy_gap",
            "worst_group_ece_gap",
            "target_risk_reduction",
        ],
        case_builder=_builder,
        x_axis_label="Failure pattern",
        metadata={
            "pattern_strength": pattern_strength,
            "weighting_method": weighting_method,
            "recalibration_method": recalibration_method,
            "target_coverage": target_coverage,
        },
    )


def make_calibration_drift_scenario(
    *,
    name: str = "calibration_drift_benchmark",
    drift_strengths: list[float] | None = None,
    seeds: list[int] | None = None,
    n_samples_ref: int = 600,
    n_samples_target: int = 600,
    weighting_method: str = "domain_classifier",
    recalibration_method: str = "temperature",
    target_coverage: float = 0.8,
    baseline_names: list[str] | None = None,
) -> BenchmarkScenario:
    """Create a scenario with calibration drift concentrated in operational slices."""

    strengths = drift_strengths or [0.6, 1.0, 1.4]
    scenario_seeds = seeds or [5, 17, 29]
    case_definitions = [
        {
            "case_id": f"drift_{strength:.2f}".replace(".", "_"),
            "case_label": f"drift={strength:.2f}",
            "pattern_strength": float(strength),
            "case_value": float(strength),
        }
        for strength in strengths
    ]

    def _builder(case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        data = make_hidden_subgroup_shift_classification(
            n_samples_ref=n_samples_ref,
            n_samples_target=n_samples_target,
            pattern="operational_calibration_drift",
            pattern_strength=float(case_definition["pattern_strength"]),
            random_state=seed,
        )
        return BenchmarkCase(
            case_id=str(case_definition["case_id"]),
            case_label=str(case_definition["case_label"]),
            family="calibration_drift",
            seed=seed,
            parameters={"pattern_strength": float(case_definition["pattern_strength"])},
            metadata={
                "weighting_method": weighting_method,
                "recalibration_method": recalibration_method,
                "target_coverage": target_coverage,
            },
            X_ref=data.X_ref,
            y_ref=data.y_ref,
            X_target=data.X_target,
            y_target=data.y_target,
            estimator=_mixed_estimator(
                categorical_features=["region", "channel"],
                continuous_features=["score", "load", "signal"],
            ),
            categorical_features=["region", "channel"],
            subgroup_features=["region", "channel", "score", "load"],
            intersectional_features=[("region", "channel")],
            reference_predictions=data.reference_predictions,
            target_predictions=data.target_predictions,
            case_value=float(case_definition["case_value"]),
        )

    return BenchmarkScenario(
        name=name,
        family="calibration_drift",
        description=(
            "Operational calibration drift scenarios with slice-concentrated "
            "failure modes."
        ),
        seeds=scenario_seeds,
        case_definitions=case_definitions,
        baseline_names=baseline_names
        or [
            "raw_model",
            "recalibration_only",
            "weighting_and_recalibration",
            "subgroup_audit",
            "recalibrated_confidence_abstention",
        ],
        publication_metrics=["delta_ece", "worst_group_ece_gap", "target_selective_ece"],
        case_builder=_builder,
        x_axis_label="Calibration-drift strength",
        metadata={
            "weighting_method": weighting_method,
            "recalibration_method": recalibration_method,
            "target_coverage": target_coverage,
        },
    )


def make_selective_shift_scenario(
    *,
    name: str = "selective_shift_benchmark",
    severities: list[float] | None = None,
    seeds: list[int] | None = None,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    n_features: int = 6,
    weighting_method: str = "domain_classifier",
    recalibration_method: str = "temperature",
    target_coverage: float = 0.8,
    baseline_names: list[str] | None = None,
) -> BenchmarkScenario:
    """Create a benchmark scenario focused on selective prediction under shift."""

    severity_values = severities or [0.4, 0.8, 1.2]
    scenario_seeds = seeds or [3, 13, 23]
    case_definitions = [
        {
            "case_id": f"severity_{severity:.2f}".replace(".", "_"),
            "case_label": f"severity={severity:.2f}",
            "severity": float(severity),
            "case_value": float(severity),
        }
        for severity in severity_values
    ]

    def _builder(case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        data = make_covariate_shift_classification(
            n_samples_ref=n_samples_ref,
            n_samples_target=n_samples_target,
            n_features=n_features,
            shift_strength=float(case_definition["severity"]),
            random_state=seed,
        )
        return BenchmarkCase(
            case_id=str(case_definition["case_id"]),
            case_label=str(case_definition["case_label"]),
            family="selective_shift",
            seed=seed,
            parameters={"severity": float(case_definition["severity"])},
            metadata={
                "weighting_method": weighting_method,
                "recalibration_method": recalibration_method,
                "target_coverage": target_coverage,
            },
            X_ref=data.X_ref,
            y_ref=data.y_ref,
            X_target=data.X_target,
            y_target=data.y_target,
            estimator=LogisticRegression(max_iter=3000),
            subgroup_features=["x0", "x1"],
            reference_predictions=data.reference_predictions,
            target_predictions=data.target_predictions,
            case_value=float(case_definition["case_value"]),
        )

    return BenchmarkScenario(
        name=name,
        family="selective_shift",
        description="Selective prediction benchmark under increasing covariate-shift severity.",
        seeds=scenario_seeds,
        case_definitions=case_definitions,
        baseline_names=baseline_names
        or [
            "confidence_abstention",
            "weighted_confidence_abstention",
            "recalibrated_confidence_abstention",
            "learned_risk_abstention",
        ],
        publication_metrics=[
            "target_risk_reduction",
            "target_selective_risk",
            "target_coverage",
        ],
        case_builder=_builder,
        x_axis_label="Shift severity",
        metadata={
            "weighting_method": weighting_method,
            "recalibration_method": recalibration_method,
            "target_coverage": target_coverage,
        },
    )


def make_mixed_tabular_scenario(
    *,
    name: str = "mixed_tabular_benchmark",
    dimensions: list[int] | None = None,
    noise_levels: list[float] | None = None,
    imbalance_levels: list[float] | None = None,
    shift_patterns: list[str] | None = None,
    seeds: list[int] | None = None,
    n_samples_ref: int = 500,
    n_samples_target: int = 500,
    weighting_method: str = "domain_classifier",
    recalibration_method: str = "temperature",
    target_coverage: float = 0.8,
    baseline_names: list[str] | None = None,
) -> BenchmarkScenario:
    """Create a configurable mixed-type synthetic benchmark family."""

    dims = dimensions or [6, 10]
    noises = noise_levels or [0.25, 0.45]
    imbalances = imbalance_levels or [0.3, 0.45]
    patterns = shift_patterns or ["covariate", "subgroup", "mixed"]
    scenario_seeds = seeds or [9, 21]

    case_definitions: list[dict[str, Any]] = []
    for index, (dimension, noise, imbalance, pattern) in enumerate(
        product(dims, noises, imbalances, patterns)
    ):
        case_definitions.append(
            {
                "case_id": f"mixed_case_{index}",
                "case_label": (
                    f"d={dimension}, noise={noise:.2f}, "
                    f"imbalance={imbalance:.2f}, pattern={pattern}"
                ),
                "dimension": int(dimension),
                "noise": float(noise),
                "imbalance": float(imbalance),
                "shift_pattern": str(pattern),
                "case_value": index,
            }
        )

    def _builder(case_definition: dict[str, Any], seed: int) -> BenchmarkCase:
        data = make_configurable_shift_classification(
            n_samples_ref=n_samples_ref,
            n_samples_target=n_samples_target,
            n_numeric_features=int(case_definition["dimension"]),
            n_categorical_features=2,
            shift_strength=1.0,
            noise=float(case_definition["noise"]),
            class_imbalance=float(case_definition["imbalance"]),
            shift_pattern=str(case_definition["shift_pattern"]),
            random_state=seed,
        )
        numeric_features = [
            column for column in data.X_ref.columns if str(column).startswith("x")
        ]
        categorical_features = [
            column for column in data.X_ref.columns if str(column).startswith("cat_")
        ]
        categorical_features.append("segment")
        return BenchmarkCase(
            case_id=str(case_definition["case_id"]),
            case_label=str(case_definition["case_label"]),
            family="mixed_tabular",
            seed=seed,
            parameters={
                "dimension": int(case_definition["dimension"]),
                "noise": float(case_definition["noise"]),
                "imbalance": float(case_definition["imbalance"]),
                "shift_pattern": str(case_definition["shift_pattern"]),
            },
            metadata={
                "weighting_method": weighting_method,
                "recalibration_method": recalibration_method,
                "target_coverage": target_coverage,
            },
            X_ref=data.X_ref,
            y_ref=data.y_ref,
            X_target=data.X_target,
            y_target=data.y_target,
            estimator=_mixed_estimator(
                categorical_features=categorical_features,
                continuous_features=numeric_features,
            ),
            categorical_features=categorical_features,
            subgroup_features=["segment", numeric_features[0], categorical_features[0]],
            intersectional_features=[("segment", categorical_features[0])],
            reference_predictions=data.reference_predictions,
            target_predictions=data.target_predictions,
            case_value=str(case_definition["case_label"]),
        )

    return BenchmarkScenario(
        name=name,
        family="mixed_tabular",
        description=(
            "Configurable mixed-type synthetic benchmark family with interpretable "
            "shift patterns."
        ),
        seeds=scenario_seeds,
        case_definitions=case_definitions,
        baseline_names=baseline_names
        or [
            "raw_model",
            "weighting_only",
            "subgroup_audit",
            "confidence_abstention",
        ],
        publication_metrics=["delta_ece", "worst_group_accuracy_gap", "target_risk_reduction"],
        case_builder=_builder,
        x_axis_label="Case definition",
        metadata={
            "weighting_method": weighting_method,
            "recalibration_method": recalibration_method,
            "target_coverage": target_coverage,
        },
    )


def scenario_from_config(config: Mapping[str, Any]) -> BenchmarkScenario:
    """Construct a benchmark scenario from a config dictionary."""

    preset = str(config["preset"])
    parameters = dict(config.get("parameters", {}))
    if "name" in config:
        parameters["name"] = config["name"]
    if "baseline_names" in config:
        parameters["baseline_names"] = list(config["baseline_names"])
    if "seeds" in config:
        parameters["seeds"] = [int(value) for value in config["seeds"]]

    builders = {
        "covariate_shift_sweep": make_covariate_shift_sweep_scenario,
        "subgroup_degradation": make_subgroup_degradation_scenario,
        "calibration_drift": make_calibration_drift_scenario,
        "selective_shift": make_selective_shift_scenario,
        "mixed_tabular": make_mixed_tabular_scenario,
    }
    if preset not in builders:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unknown scenario preset {preset!r}. Supported presets: {supported}.")
    return builders[preset](**parameters)


def _mixed_estimator(
    *,
    categorical_features: list[str],
    continuous_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("continuous", "passthrough", continuous_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000)),
        ]
    )
