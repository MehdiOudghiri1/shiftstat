"""Registries for benchmark baselines and metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineDefinition:
    """Callable benchmark baseline with descriptive metadata."""

    name: str
    runner: Callable[[Any, int], dict[str, Any]]
    description: str
    category: str


class BaselineRegistry:
    """Registry of named benchmark baselines."""

    def __init__(self) -> None:
        self._definitions: dict[str, BaselineDefinition] = {}

    def register(
        self,
        name: str,
        runner: Callable[[Any, int], dict[str, Any]],
        *,
        description: str,
        category: str,
    ) -> BaselineRegistry:
        """Register a new baseline definition."""

        self._definitions[name] = BaselineDefinition(
            name=name,
            runner=runner,
            description=description,
            category=category,
        )
        return self

    def get(self, name: str) -> BaselineDefinition:
        """Return a registered baseline definition."""

        if name not in self._definitions:
            available = ", ".join(sorted(self._definitions))
            raise KeyError(f"Unknown baseline {name!r}. Available baselines: {available}.")
        return self._definitions[name]

    def names(self) -> list[str]:
        """Return the registered baseline names."""

        return sorted(self._definitions)

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return a machine-readable registry description."""

        return {
            name: {
                "description": definition.description,
                "category": definition.category,
            }
            for name, definition in sorted(self._definitions.items())
        }


@dataclass(frozen=True)
class MetricDefinition:
    """Description of a benchmark metric."""

    name: str
    label: str
    higher_is_better: bool
    category: str


class MetricRegistry:
    """Registry describing benchmark metrics and their semantics."""

    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    def register(
        self,
        name: str,
        *,
        label: str,
        higher_is_better: bool,
        category: str,
    ) -> MetricRegistry:
        """Register a metric definition."""

        self._definitions[name] = MetricDefinition(
            name=name,
            label=label,
            higher_is_better=higher_is_better,
            category=category,
        )
        return self

    def get(self, name: str) -> MetricDefinition:
        """Return a registered metric definition."""

        if name not in self._definitions:
            available = ", ".join(sorted(self._definitions))
            raise KeyError(f"Unknown metric {name!r}. Available metrics: {available}.")
        return self._definitions[name]

    def names(self) -> list[str]:
        """Return the registered metric names."""

        return sorted(self._definitions)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return a machine-readable metric schema."""

        return {
            name: {
                "label": definition.label,
                "higher_is_better": definition.higher_is_better,
                "category": definition.category,
            }
            for name, definition in sorted(self._definitions.items())
        }


def default_metric_registry() -> MetricRegistry:
    """Return the default metric registry used by ShiftStat V5 benchmarks."""

    registry = MetricRegistry()
    registry.register(
        "target_accuracy",
        label="Target accuracy",
        higher_is_better=True,
        category="performance",
    )
    registry.register(
        "delta_accuracy",
        label="Delta accuracy",
        higher_is_better=True,
        category="performance",
    )
    registry.register(
        "target_ece",
        label="Target ECE",
        higher_is_better=False,
        category="calibration",
    )
    registry.register(
        "delta_ece",
        label="Delta ECE",
        higher_is_better=False,
        category="calibration",
    )
    registry.register(
        "delta_log_loss",
        label="Delta log loss",
        higher_is_better=False,
        category="performance",
    )
    registry.register(
        "effective_sample_size",
        label="Effective sample size",
        higher_is_better=True,
        category="weighting",
    )
    registry.register(
        "worst_group_accuracy_gap",
        label="Worst-group accuracy gap",
        higher_is_better=False,
        category="subgroup",
    )
    registry.register(
        "worst_group_ece_gap",
        label="Worst-group calibration gap",
        higher_is_better=False,
        category="subgroup",
    )
    registry.register(
        "target_coverage",
        label="Retained coverage",
        higher_is_better=True,
        category="selective",
    )
    registry.register(
        "target_selective_risk",
        label="Selective risk",
        higher_is_better=False,
        category="selective",
    )
    registry.register(
        "target_risk_reduction",
        label="Selective risk reduction",
        higher_is_better=True,
        category="selective",
    )
    registry.register(
        "target_selective_ece",
        label="Selective ECE",
        higher_is_better=False,
        category="selective",
    )
    registry.register(
        "subgroup_abstention_gap",
        label="Subgroup abstention gap",
        higher_is_better=False,
        category="selective",
    )
    registry.register(
        "masked_accuracy_drop",
        label="Masked accuracy drop flag",
        higher_is_better=False,
        category="audit",
    )
    registry.register(
        "masked_calibration_drift",
        label="Masked calibration drift flag",
        higher_is_better=False,
        category="audit",
    )
    registry.register(
        "concentrated_failures",
        label="Failure concentration flag",
        higher_is_better=False,
        category="audit",
    )
    return registry
