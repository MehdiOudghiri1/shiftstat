"""Conditional reliability auditing and failure-slice discovery."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AuditReport",
    "CertifiedAuditConfig",
    "CertifiedAuditReport",
    "CertifiedCellResult",
    "CertifiedWorstGroupAuditor",
    "ConditionalReliabilityAuditor",
    "AuditDecision",
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
    if name in {
        "AuditDecision",
        "CertifiedAuditConfig",
        "CertifiedAuditReport",
        "CertifiedCellResult",
        "CertifiedWorstGroupAuditor",
    }:
        from shiftstat.certification import (
            AuditDecision,
            CertifiedAuditConfig,
            CertifiedAuditReport,
            CertifiedCellResult,
            CertifiedWorstGroupAuditor,
        )

        mapping = {
            "AuditDecision": AuditDecision,
            "CertifiedAuditConfig": CertifiedAuditConfig,
            "CertifiedAuditReport": CertifiedAuditReport,
            "CertifiedCellResult": CertifiedCellResult,
            "CertifiedWorstGroupAuditor": CertifiedWorstGroupAuditor,
        }
        return mapping[name]
    raise AttributeError(f"module 'shiftstat.audit' has no attribute {name!r}")
