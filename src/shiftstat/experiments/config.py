"""Experiment config parsing for JSON and YAML manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScenarioConfig:
    """One scenario entry in an experiment configuration."""

    preset: str
    name: str | None = None
    seeds: list[int] | None = None
    baseline_names: list[str] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioConfig:
        """Parse a scenario configuration from a dictionary."""

        return cls(
            preset=str(payload["preset"]),
            name=None if payload.get("name") is None else str(payload["name"]),
            seeds=(
                None if payload.get("seeds") is None else [int(seed) for seed in payload["seeds"]]
            ),
            baseline_names=(
                None
                if payload.get("baseline_names") is None
                else [str(name) for name in payload["baseline_names"]]
            ),
            parameters=dict(payload.get("parameters", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable configuration block."""

        payload: dict[str, Any] = {
            "preset": self.preset,
            "parameters": self.parameters,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.seeds is not None:
            payload["seeds"] = self.seeds
        if self.baseline_names is not None:
            payload["baseline_names"] = self.baseline_names
        return payload


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    output_dir: str
    scenarios: list[ScenarioConfig]
    figure_format: str = "png"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Parse a top-level experiment configuration dictionary."""

        if "scenarios" in payload:
            scenarios = [ScenarioConfig.from_dict(item) for item in payload["scenarios"]]
        elif "scenario" in payload:
            scenarios = [ScenarioConfig.from_dict(payload["scenario"])]
        else:
            raise ValueError("Experiment config must include 'scenario' or 'scenarios'.")
        return cls(
            name=str(payload["name"]),
            output_dir=str(payload.get("output_dir", "artifacts")),
            scenarios=scenarios,
            figure_format=str(payload.get("figure_format", "png")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable config dictionary."""

        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "figure_format": self.figure_format,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from JSON or YAML."""

    path = Path(config_path)
    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(content)
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(content)
    else:
        raise ValueError("Config path must end in .json, .yaml, or .yml.")
    if not isinstance(payload, dict):
        raise TypeError("Experiment config must parse to a dictionary.")
    return ExperimentConfig.from_dict(payload)
