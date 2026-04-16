"""Conditional reliability auditing and failure-slice discovery."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AuditReport",
    "ConditionalReliabilityAuditor",
    "ReliabilityAuditor",
    "SliceDiscoverer",
    "discover_failure_slices",
]


def __getattr__(name: str) -> Any:
    if name in {"ConditionalReliabilityAuditor", "ReliabilityAuditor"}:
        from .auditor import ConditionalReliabilityAuditor, ReliabilityAuditor

        if name == "ConditionalReliabilityAuditor":
            return ConditionalReliabilityAuditor
        return ReliabilityAuditor
    if name in {"SliceDiscoverer", "discover_failure_slices"}:
        from .discovery import SliceDiscoverer, discover_failure_slices

        if name == "SliceDiscoverer":
            return SliceDiscoverer
        return discover_failure_slices
    if name == "AuditReport":
        from .results import AuditReport

        return AuditReport
    raise AttributeError(f"module 'shiftstat.audit' has no attribute {name!r}")
