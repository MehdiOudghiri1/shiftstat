"""Synthetic datasets designed for subgroup and audit-oriented diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shiftstat.datasets.synthetic import SyntheticShiftDataset
from shiftstat.utils.random import check_random_state


def make_hidden_subgroup_shift_classification(
    *,
    n_samples_ref: int = 600,
    n_samples_target: int = 600,
    pattern: str = "masked_subgroup_shift",
    pattern_strength: float = 1.0,
    random_state: int | np.random.RandomState | None = None,
) -> SyntheticShiftDataset:
    """Generate a binary classification problem with hidden subgroup failures.

    Parameters
    ----------
    pattern:
        One of:
        - ``"masked_subgroup_shift"`` for subgroup-specific degradation with
          stable-looking aggregates
        - ``"minority_subgroup_degradation"`` for a sharper minority-group failure mode
        - ``"operational_calibration_drift"`` for calibration drift concentrated in
          operational slices
    """

    rng = check_random_state(random_state)
    X_ref = _make_frame(n_samples_ref, rng, target=False)
    X_target = _make_frame(n_samples_target, rng, target=True)

    reference_logit = _base_model_logit(X_ref)
    target_logit = _base_model_logit(X_target)

    reference_true_logit = reference_logit
    target_true_logit = target_logit + pattern_strength * _pattern_shift(X_target, pattern=pattern)

    if pattern == "operational_calibration_drift":
        target_prediction_logit = target_logit + pattern_strength * 0.35 * (
            (X_target["channel"] == "urgent").to_numpy(dtype=float)
        )
    else:
        target_prediction_logit = target_logit

    y_ref = rng.binomial(1, _sigmoid(reference_true_logit))
    y_target = rng.binomial(1, _sigmoid(target_true_logit))

    return SyntheticShiftDataset(
        X_ref=X_ref,
        X_target=X_target,
        y_ref=y_ref.astype(int),
        y_target=y_target.astype(int),
        reference_predictions=_sigmoid(reference_logit).astype(float),
        target_predictions=_sigmoid(target_prediction_logit).astype(float),
        task="classification",
    )


def _make_frame(
    n_samples: int,
    rng: np.random.RandomState,
    *,
    target: bool,
) -> pd.DataFrame:
    minority_rate = 0.18 if not target else 0.24
    urgent_rate = 0.22 if not target else 0.38
    region = rng.choice(
        ["majority", "minority"], size=n_samples, p=[1 - minority_rate, minority_rate]
    )
    channel = rng.choice(["standard", "urgent"], size=n_samples, p=[1 - urgent_rate, urgent_rate])
    score = rng.normal(loc=0.0 if not target else 0.25, scale=1.0, size=n_samples)
    load = rng.normal(loc=0.0 if not target else 0.35, scale=1.0, size=n_samples)
    load += 0.75 * (channel == "urgent").astype(float)
    load += 0.55 * (region == "minority").astype(float)
    signal = rng.normal(loc=0.2 * (target), scale=1.0, size=n_samples)
    return pd.DataFrame(
        {
            "score": score,
            "load": load,
            "signal": signal,
            "region": region,
            "channel": channel,
        }
    )


def _base_model_logit(frame: pd.DataFrame) -> np.ndarray:
    score = frame["score"].to_numpy(dtype=float)
    load = frame["load"].to_numpy(dtype=float)
    signal = frame["signal"].to_numpy(dtype=float)
    minority = (frame["region"] == "minority").to_numpy(dtype=float)
    urgent = (frame["channel"] == "urgent").to_numpy(dtype=float)
    return np.asarray(
        0.95 * score - 0.75 * load + 0.55 * signal + 0.3 * minority + 0.25 * urgent,
        dtype=float,
    )


def _pattern_shift(frame: pd.DataFrame, *, pattern: str) -> np.ndarray:
    minority = (frame["region"] == "minority").to_numpy(dtype=float)
    urgent = (frame["channel"] == "urgent").to_numpy(dtype=float)
    high_load = (frame["load"].to_numpy(dtype=float) > 0.7).astype(float)
    negative_score = (frame["score"].to_numpy(dtype=float) < 0.0).astype(float)

    if pattern == "masked_subgroup_shift":
        return np.asarray(
            1.15 * minority * high_load - 0.18 * (1.0 - minority) * urgent,
            dtype=float,
        )
    if pattern == "minority_subgroup_degradation":
        return np.asarray(1.6 * minority + 0.9 * minority * urgent, dtype=float)
    if pattern == "operational_calibration_drift":
        return np.asarray(0.95 * urgent * negative_score - 0.1 * minority, dtype=float)
    raise ValueError(
        "pattern must be one of 'masked_subgroup_shift', "
        "'minority_subgroup_degradation', or 'operational_calibration_drift'."
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)
