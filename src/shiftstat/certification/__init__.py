"""Certified worst-group reliability auditing under covariate shift."""

from shiftstat.certification.auditor import CertifiedAuditConfig, CertifiedWorstGroupAuditor
from shiftstat.certification.metrics import (
    certified_excess,
    local_effective_sample_size,
    sensitivity_envelope_radius,
    simultaneous_radius,
    weight_nuisance_radius,
)
from shiftstat.certification.results import (
    AuditDecision,
    CertifiedAuditReport,
    CertifiedCellResult,
)

__all__ = [
    "AuditDecision",
    "CertifiedAuditConfig",
    "CertifiedAuditReport",
    "CertifiedCellResult",
    "CertifiedWorstGroupAuditor",
    "certified_excess",
    "local_effective_sample_size",
    "sensitivity_envelope_radius",
    "simultaneous_radius",
    "weight_nuisance_radius",
]
