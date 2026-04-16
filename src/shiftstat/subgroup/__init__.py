"""Subgroup-aware reliability diagnostics under deployment shift."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SubgroupAnalyzer",
    "SubgroupReport",
    "group_by_feature",
    "group_metrics",
]


def __getattr__(name: str) -> Any:
    if name in {"SubgroupAnalyzer", "group_by_feature", "group_metrics"}:
        from . import analyzer

        return getattr(analyzer, name)
    if name == "SubgroupReport":
        from .results import SubgroupReport

        return SubgroupReport
    raise AttributeError(f"module 'shiftstat.subgroup' has no attribute {name!r}")
