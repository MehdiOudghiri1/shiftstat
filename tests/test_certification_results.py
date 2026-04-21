from __future__ import annotations

from shiftstat.certification.results import (
    AuditDecision,
    CertifiedAuditReport,
    CertifiedCellResult,
)


def _cell(
    decision: AuditDecision,
    *,
    gap: float,
    certified_alarm: bool,
    naive_alarm: bool = True,
) -> CertifiedCellResult:
    return CertifiedCellResult(
        group=str(decision.value),
        bin_lower=0.0,
        bin_upper=0.5,
        source_count=20,
        target_count=15,
        estimated_target_mass=0.4,
        weighted_gap=gap,
        abs_weighted_gap=abs(gap),
        local_ess=18.0,
        label_radius=0.02,
        rho_sensitivity=0.01,
        gamma_population=0.0,
        total_radius=0.03,
        label_certified_excess=0.02,
        certified_excess=0.01 if certified_alarm else 0.0,
        naive_alarm=naive_alarm,
        label_certified_alarm=certified_alarm,
        certified_alarm=certified_alarm,
        decision=decision,
        reason="test",
    )


def test_certified_audit_report_views_and_serializers() -> None:
    report = CertifiedAuditReport(
        cells=(
            _cell(AuditDecision.CERTIFIED_FAILURE, gap=0.2, certified_alarm=True),
            _cell(AuditDecision.INSUFFICIENT_EVIDENCE, gap=0.15, certified_alarm=False),
            _cell(AuditDecision.OUT_OF_SCOPE, gap=0.0, certified_alarm=False, naive_alarm=False),
        ),
        tolerance=0.1,
        alpha=0.05,
        n_cells_tested=3,
        assumptions=("fixed family",),
        warnings=("low support",),
        metadata={"scenario": "unit"},
    )

    assert report.to_frame()["abs_weighted_gap"].iloc[0] == 0.2
    assert len(report.certified_failures()) == 1
    assert len(report.insufficient_evidence()) == 1
    assert len(report.out_of_scope()) == 1
    assert report.summary()["n_certified_failures"] == 1
    assert report.to_dict()["cells"][0]["decision"] == "certified_failure"
    assert "Certified worst-group audit" in report.to_markdown()


def test_empty_certified_audit_report_is_well_formed() -> None:
    report = CertifiedAuditReport(
        cells=(),
        tolerance=0.1,
        alpha=0.05,
        n_cells_tested=0,
        assumptions=(),
        warnings=(),
        metadata={},
    )

    assert report.to_frame().empty
    assert report.certified_failures().empty
    assert report.summary()["n_cells"] == 0
    assert "No cells were evaluated" in report.to_markdown()
