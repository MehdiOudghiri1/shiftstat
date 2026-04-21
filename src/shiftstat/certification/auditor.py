"""High-level certified worst-group audit API."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from shiftstat.exceptions import InsufficientOverlapWarning, NuisanceDominatesWarning
from shiftstat.utils.validation import ensure_1d, validate_same_length

from .metrics import (
    certified_excess,
    local_effective_sample_size,
    sensitivity_envelope_radius,
    simultaneous_radius,
)
from .results import AuditDecision, CertifiedAuditReport, CertifiedCellResult


@dataclass(frozen=True)
class CertifiedAuditConfig:
    """Configuration for certified worst-group auditing."""

    n_bins: int = 6
    score_bins: tuple[float, ...] | None = None
    tolerance: float = 0.10
    alpha: float = 0.10
    min_target_mass: float = 0.0
    min_source_count: int = 1
    min_local_ess: float = 0.0
    include_population_radius: bool = False
    population_radius: float = 0.0


def _score_bins(config: CertifiedAuditConfig) -> np.ndarray:
    if config.score_bins is None:
        if config.n_bins < 1:
            raise ValueError("n_bins must be at least one.")
        return np.linspace(0.0, 1.0, config.n_bins + 1)
    bins = np.asarray(config.score_bins, dtype=float)
    if bins.ndim != 1 or len(bins) < 2:
        raise ValueError("score_bins must be a one-dimensional sequence with at least two edges.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("score_bins must be strictly increasing.")
    if bins[0] < 0.0 or bins[-1] > 1.0:
        raise ValueError("score_bins must lie inside [0, 1].")
    return bins


def _bin_mask(scores: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if upper < 1.0:
        return (scores >= lower) & (scores < upper)
    return (scores >= lower) & (scores <= upper)


def _validate_scores(scores: np.ndarray, *, name: str) -> np.ndarray:
    arr = ensure_1d(scores, name=name).astype(float)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return arr


def _prepare_groups(
    *,
    n_source: int,
    n_target: int | None,
    groups: Mapping[str, np.ndarray] | None,
    target_groups: Mapping[str, np.ndarray] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    if groups is None:
        source_all = np.ones(n_source, dtype=bool)
        target_all = np.ones(n_target, dtype=bool) if n_target is not None else None
        return {"all": (source_all, target_all)}

    prepared: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for name, source_mask_like in groups.items():
        source_mask = ensure_1d(source_mask_like, name=f"groups[{name!r}]").astype(bool)
        if len(source_mask) != n_source:
            raise ValueError(
                f"Source group {name!r} has length {len(source_mask)}, expected {n_source}."
            )
        target_mask = None
        if target_groups is not None and name in target_groups:
            target_mask = ensure_1d(
                target_groups[name],
                name=f"target_groups[{name!r}]",
            ).astype(bool)
            if n_target is None:
                raise ValueError("target_groups require target scores.")
            if len(target_mask) != n_target:
                raise ValueError(
                    f"Target group {name!r} has length {len(target_mask)}, expected {n_target}."
                )
        prepared[str(name)] = (source_mask, target_mask)
    if not prepared:
        raise ValueError("At least one group must be supplied.")
    return prepared


class CertifiedWorstGroupAuditor:
    """Certified subgroup-bin reliability audit under covariate shift.

    This estimator implements the practical audit rule from the paper: compute
    weighted subgroup-bin residuals, attach simultaneous local-ESS radii, add
    optional learned-weight sensitivity, and classify each cell as a certified
    failure, insufficient evidence, no detected failure, or out of scope.
    """

    def __init__(self, config: CertifiedAuditConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")
        self.config = config if config is not None else CertifiedAuditConfig(**kwargs)

    def fit(
        self,
        *,
        y_source: np.ndarray,
        scores_source: np.ndarray,
        weights: np.ndarray | None = None,
        groups: Mapping[str, np.ndarray] | None = None,
        scores_target: np.ndarray | None = None,
        target_groups: Mapping[str, np.ndarray] | None = None,
        y_target: np.ndarray | None = None,
        alternative_weights: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CertifiedWorstGroupAuditor:
        """Fit the certified audit.

        Target labels are optional and are used only for retrospective validation
        columns; certification uses source labels, target covariate-derived groups,
        scores, weights, and optional nuisance envelopes.
        """

        y_arr = ensure_1d(y_source, name="y_source").astype(float)
        score_arr = _validate_scores(scores_source, name="scores_source")
        validate_same_length(y_arr, score_arr)
        n_source = len(y_arr)

        if weights is None:
            weight_arr = np.ones(n_source, dtype=float)
        else:
            weight_arr = ensure_1d(weights, name="weights").astype(float)
            validate_same_length(y_arr, weight_arr)
            if np.any(weight_arr < 0.0):
                raise ValueError("weights must be nonnegative.")

        alt_weight_arrs: list[np.ndarray] = []
        for index, alt in enumerate(alternative_weights or []):
            alt_arr = ensure_1d(alt, name=f"alternative_weights[{index}]").astype(float)
            validate_same_length(weight_arr, alt_arr)
            if np.any(alt_arr < 0.0):
                raise ValueError("alternative weights must be nonnegative.")
            alt_weight_arrs.append(alt_arr)

        target_score_arr = (
            None
            if scores_target is None
            else _validate_scores(
                scores_target,
                name="scores_target",
            )
        )
        target_y_arr = None
        if y_target is not None:
            if target_score_arr is None:
                raise ValueError("y_target requires scores_target.")
            target_y_arr = ensure_1d(y_target, name="y_target").astype(float)
            validate_same_length(target_y_arr, target_score_arr)

        group_masks = _prepare_groups(
            n_source=n_source,
            n_target=None if target_score_arr is None else len(target_score_arr),
            groups=groups,
            target_groups=target_groups,
        )
        bins = _score_bins(self.config)
        n_cells_tested = max(len(group_masks) * (len(bins) - 1), 1)
        source_residual = y_arr - score_arr
        target_residual = None if target_y_arr is None else target_y_arr - target_score_arr
        total_weight = float(np.sum(weight_arr))
        if total_weight <= 0.0:
            raise ValueError("weights must have positive total mass.")

        results: list[CertifiedCellResult] = []
        warnings_list: list[str] = []

        for group_name, (source_group, target_group) in group_masks.items():
            for lower, upper in zip(bins[:-1], bins[1:], strict=True):
                source_mask = source_group & _bin_mask(score_arr, float(lower), float(upper))
                target_mask = (
                    None
                    if target_group is None or target_score_arr is None
                    else target_group & _bin_mask(target_score_arr, float(lower), float(upper))
                )
                result = self._evaluate_cell(
                    group_name=group_name,
                    bin_lower=float(lower),
                    bin_upper=float(upper),
                    source_mask=source_mask,
                    target_mask=target_mask,
                    source_residual=source_residual,
                    target_residual=target_residual,
                    weights=weight_arr,
                    alternative_weights=alt_weight_arrs,
                    total_weight=total_weight,
                    n_cells_tested=n_cells_tested,
                )
                results.append(result)
                if result.decision == AuditDecision.INSUFFICIENT_EVIDENCE and result.local_ess < 5:
                    warnings_list.append(
                        f"{result.group} {result.bin_label} has a naive alarm with local ESS "
                        f"{result.local_ess:.2f}; report insufficient evidence."
                    )
                if (
                    np.isfinite(result.rho_sensitivity)
                    and result.rho_sensitivity > result.abs_weighted_gap
                ):
                    warnings_list.append(
                        f"{result.group} {result.bin_label} has weight sensitivity larger than "
                        "the apparent gap."
                    )

        self.report_ = CertifiedAuditReport(
            cells=tuple(results),
            tolerance=self.config.tolerance,
            alpha=self.config.alpha,
            n_cells_tested=n_cells_tested,
            assumptions=(
                "Fixed finite subgroup-bin family unless sample splitting is used.",
                "Source labels are conditionally independent given covariates and audit weights.",
                "Learned-weight sensitivity is valid only if the alternative envelope is credible.",
            ),
            warnings=tuple(dict.fromkeys(warnings_list)),
            metadata=dict(metadata or {}),
        )
        if warnings_list:
            if any("weight sensitivity" in warning for warning in warnings_list):
                warnings.warn(warnings_list[0], NuisanceDominatesWarning, stacklevel=2)
            else:
                warnings.warn(warnings_list[0], InsufficientOverlapWarning, stacklevel=2)
        return self

    def report(self) -> CertifiedAuditReport:
        """Return the fitted certified audit report."""

        if not hasattr(self, "report_"):
            raise ValueError("CertifiedWorstGroupAuditor must be fitted before report().")
        return self.report_

    def _evaluate_cell(
        self,
        *,
        group_name: str,
        bin_lower: float,
        bin_upper: float,
        source_mask: np.ndarray,
        target_mask: np.ndarray | None,
        source_residual: np.ndarray,
        target_residual: np.ndarray | None,
        weights: np.ndarray,
        alternative_weights: list[np.ndarray],
        total_weight: float,
        n_cells_tested: int,
    ) -> CertifiedCellResult:
        source_count = int(np.sum(source_mask))
        target_count = None if target_mask is None else int(np.sum(target_mask))
        if source_count == 0:
            return self._out_of_scope_result(
                group_name,
                bin_lower,
                bin_upper,
                source_count,
                target_count,
                "empty source cell",
            )

        local_weights = weights[source_mask]
        denominator = float(np.sum(local_weights))
        estimated_target_mass = denominator / total_weight if total_weight > 0 else 0.0
        if denominator <= 0.0:
            return self._out_of_scope_result(
                group_name,
                bin_lower,
                bin_upper,
                source_count,
                target_count,
                "zero weighted denominator",
                estimated_target_mass=estimated_target_mass,
            )
        if source_count < self.config.min_source_count:
            return self._out_of_scope_result(
                group_name,
                bin_lower,
                bin_upper,
                source_count,
                target_count,
                "source count below minimum",
                estimated_target_mass=estimated_target_mass,
            )
        if estimated_target_mass < self.config.min_target_mass:
            return self._out_of_scope_result(
                group_name,
                bin_lower,
                bin_upper,
                source_count,
                target_count,
                "estimated target mass below minimum",
                estimated_target_mass=estimated_target_mass,
            )

        gap = float(np.average(source_residual[source_mask], weights=local_weights))
        abs_gap = abs(gap)
        local_ess = local_effective_sample_size(weights, source_mask)
        label_radius = simultaneous_radius(local_ess, n_cells_tested, self.config.alpha)
        rho = sensitivity_envelope_radius(weights, alternative_weights, source_mask)
        gamma = self.config.population_radius if self.config.include_population_radius else 0.0
        total_radius = label_radius + rho + gamma
        label_excess = certified_excess(abs_gap, label_radius, self.config.tolerance)
        total_excess = certified_excess(abs_gap, total_radius, self.config.tolerance)
        naive_alarm = abs_gap > self.config.tolerance
        label_alarm = label_excess > 0.0
        certified_alarm = total_excess > 0.0

        if local_ess < self.config.min_local_ess:
            decision = (
                AuditDecision.INSUFFICIENT_EVIDENCE if naive_alarm else AuditDecision.OUT_OF_SCOPE
            )
            reason = "local ESS below minimum"
        elif certified_alarm:
            decision = AuditDecision.CERTIFIED_FAILURE
            reason = "certified excess above tolerance"
        elif naive_alarm:
            decision = AuditDecision.INSUFFICIENT_EVIDENCE
            reason = "naive alarm does not clear certified radius"
        else:
            decision = AuditDecision.NO_DETECTED_FAILURE
            reason = "gap below tolerance"

        target_gap = None
        abs_target_gap = None
        if target_mask is not None and target_residual is not None and np.any(target_mask):
            target_gap = float(np.mean(target_residual[target_mask]))
            abs_target_gap = abs(target_gap)

        return CertifiedCellResult(
            group=group_name,
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            source_count=source_count,
            target_count=target_count,
            estimated_target_mass=estimated_target_mass,
            weighted_gap=gap,
            abs_weighted_gap=abs_gap,
            local_ess=local_ess,
            label_radius=label_radius,
            rho_sensitivity=rho,
            gamma_population=gamma,
            total_radius=total_radius,
            label_certified_excess=label_excess,
            certified_excess=total_excess,
            naive_alarm=naive_alarm,
            label_certified_alarm=label_alarm,
            certified_alarm=certified_alarm,
            decision=decision,
            reason=reason,
            target_gap=target_gap,
            abs_target_gap=abs_target_gap,
        )

    def _out_of_scope_result(
        self,
        group_name: str,
        bin_lower: float,
        bin_upper: float,
        source_count: int,
        target_count: int | None,
        reason: str,
        *,
        estimated_target_mass: float = 0.0,
    ) -> CertifiedCellResult:
        return CertifiedCellResult(
            group=group_name,
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            source_count=source_count,
            target_count=target_count,
            estimated_target_mass=estimated_target_mass,
            weighted_gap=float("nan"),
            abs_weighted_gap=float("nan"),
            local_ess=0.0,
            label_radius=float("inf"),
            rho_sensitivity=0.0,
            gamma_population=0.0,
            total_radius=float("inf"),
            label_certified_excess=0.0,
            certified_excess=0.0,
            naive_alarm=False,
            label_certified_alarm=False,
            certified_alarm=False,
            decision=AuditDecision.OUT_OF_SCOPE,
            reason=reason,
        )
