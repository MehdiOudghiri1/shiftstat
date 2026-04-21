from __future__ import annotations

import numpy as np

from shiftstat.certification import (
    AuditDecision,
    CertifiedWorstGroupAuditor,
    certified_excess,
    local_effective_sample_size,
    sensitivity_envelope_radius,
    simultaneous_radius,
    weight_nuisance_radius,
)


def test_local_ess_and_radius_behave_monotonically() -> None:
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    one_point = np.array([True, False, False, False])
    four_points = np.array([True, True, True, True])

    assert local_effective_sample_size(weights, one_point) == 1.0
    assert local_effective_sample_size(weights, four_points) == 4.0
    assert simultaneous_radius(4.0, n_cells=10, alpha=0.1) < simultaneous_radius(
        1.0,
        n_cells=10,
        alpha=0.1,
    )


def test_certified_excess_requires_gap_above_radius_and_tolerance() -> None:
    assert np.isclose(certified_excess(0.30, radius=0.15, tolerance=0.10), 0.05)
    assert certified_excess(0.20, radius=0.15, tolerance=0.10) == 0.0


def test_weight_nuisance_radius_is_zero_for_identical_weights() -> None:
    weights = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    assert weight_nuisance_radius(weights, weights.copy(), mask) == 0.0
    perturbed = np.array([1.2, 2.0, 2.5])
    assert sensitivity_envelope_radius(weights, [perturbed], mask) > 0.0


def test_certified_auditor_marks_thin_naive_alarm_as_insufficient_evidence() -> None:
    y = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    scores = np.array([0.1, 0.1, 0.5, 0.5, 0.7, 0.7], dtype=float)
    groups = {
        "thin": np.array([True, False, False, False, False, False]),
        "bulk": np.array([False, True, True, True, True, True]),
    }

    report = (
        CertifiedWorstGroupAuditor(n_bins=3, tolerance=0.2, alpha=0.1)
        .fit(
            y_source=y,
            scores_source=scores,
            groups=groups,
        )
        .report()
    )

    thin = report.to_frame().loc[lambda frame: frame["group"] == "thin"].iloc[0]
    assert thin["naive_alarm"]
    assert thin["decision"] == AuditDecision.INSUFFICIENT_EVIDENCE.value
    assert thin["local_ess"] == 1.0


def test_certified_auditor_can_certify_well_supported_failure() -> None:
    y = np.ones(80, dtype=float)
    scores = np.full(80, 0.1, dtype=float)
    groups = {"all": np.ones(80, dtype=bool)}

    report = (
        CertifiedWorstGroupAuditor(n_bins=2, tolerance=0.2, alpha=0.1)
        .fit(
            y_source=y,
            scores_source=scores,
            groups=groups,
        )
        .report()
    )

    failures = report.certified_failures()
    assert len(failures) == 1
    assert failures.iloc[0]["decision"] == AuditDecision.CERTIFIED_FAILURE.value
