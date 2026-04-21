"""Core numerical routines for certified subgroup-bin auditing."""

from __future__ import annotations

import numpy as np

from shiftstat.utils.validation import ensure_1d, validate_same_length


def local_effective_sample_size(sample_weight: np.ndarray, cell_mask: np.ndarray) -> float:
    """Compute Kish effective sample size inside one subgroup-bin cell."""

    weights = ensure_1d(sample_weight, name="sample_weight").astype(float)
    mask = ensure_1d(cell_mask, name="cell_mask").astype(bool)
    validate_same_length(weights, mask)
    local_weights = weights[mask]
    if local_weights.size == 0:
        return 0.0
    denominator = float(np.sum(local_weights**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(local_weights) ** 2 / denominator)


def simultaneous_radius(local_ess: float, n_cells: int, alpha: float) -> float:
    """Return the finite-family Hoeffding radius for a local weighted residual."""

    if n_cells < 1:
        raise ValueError("n_cells must be at least one.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if local_ess <= 0.0:
        return float("inf")
    return float(np.sqrt(np.log(2.0 * n_cells / alpha) / (2.0 * local_ess)))


def certified_excess(gap: float, radius: float, tolerance: float) -> float:
    """Return the positive certified excess above a practical tolerance."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    return float(max(abs(float(gap)) - float(radius) - float(tolerance), 0.0))


def weight_nuisance_radius(
    base_weight: np.ndarray,
    alternative_weight: np.ndarray,
    cell_mask: np.ndarray,
) -> float:
    """Compute a cellwise learned-weight sensitivity radius.

    The returned value is
    ``2 * sum(abs(base - alternative)) / (sum(alternative) - sum(abs(base - alternative)))``
    inside the cell. If the denominator is nonpositive, the radius is infinite.
    """

    base = ensure_1d(base_weight, name="base_weight").astype(float)
    alternative = ensure_1d(alternative_weight, name="alternative_weight").astype(float)
    mask = ensure_1d(cell_mask, name="cell_mask").astype(bool)
    validate_same_length(base, alternative)
    validate_same_length(base, mask)
    delta = float(np.sum(np.abs(base[mask] - alternative[mask])))
    denominator = float(np.sum(alternative[mask]) - delta)
    if denominator <= 0.0:
        return float("inf")
    return float(2.0 * delta / denominator)


def sensitivity_envelope_radius(
    base_weight: np.ndarray,
    alternative_weights: list[np.ndarray] | tuple[np.ndarray, ...],
    cell_mask: np.ndarray,
) -> float:
    """Return the worst cellwise nuisance radius over plausible alternative weights."""

    if not alternative_weights:
        return 0.0
    radii = [
        weight_nuisance_radius(base_weight, alternative_weight, cell_mask)
        for alternative_weight in alternative_weights
    ]
    return float(np.max(radii))
