from __future__ import annotations

import numpy as np

from shiftstat.metrics import (
    compute_effective_sample_size,
    source_separability_score,
    weighted_accuracy,
    weighted_brier_score,
    weighted_log_loss,
    weighted_mae,
    weighted_mse,
)


def test_weighted_metrics_sanity() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.7, 0.3])
    weights = np.array([1.0, 2.0, 1.0, 1.0])

    assert weighted_accuracy(y_true, y_pred, sample_weight=weights) > 0.7
    assert weighted_log_loss(y_true, y_pred, sample_weight=weights) > 0.0
    assert weighted_brier_score(y_true, y_pred, sample_weight=weights) >= 0.0
    assert weighted_mse(y_true, y_pred, sample_weight=weights) >= 0.0
    assert weighted_mae(y_true, y_pred, sample_weight=weights) >= 0.0
    assert compute_effective_sample_size(weights) <= len(weights)
    assert source_separability_score(y_true, y_pred) >= 0.5

