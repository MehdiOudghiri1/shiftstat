"""Probability helpers for binary reliability analysis."""

from __future__ import annotations

import numpy as np

from shiftstat.exceptions import ValidationError
from shiftstat.utils.validation import ensure_1d


def extract_positive_class_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from binary classifier outputs."""

    array = np.asarray(probabilities, dtype=float)
    if array.ndim == 1:
        result = array
    elif array.ndim == 2 and array.shape[1] == 2:
        result = array[:, 1]
    elif array.ndim == 2 and array.shape[1] == 1:
        result = array[:, 0]
    else:
        raise ValidationError(
            "Binary probability inputs must be one-dimensional or have shape (n_samples, 2)."
        )
    clipped = np.clip(ensure_1d(result, name="probabilities").astype(float), 1e-12, 1 - 1e-12)
    return np.asarray(clipped, dtype=float)  # type: ignore[no-any-return]


def confidence_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return binary classification confidence as ``max(p, 1 - p)``."""

    positive = extract_positive_class_probabilities(probabilities)
    return np.asarray(np.maximum(positive, 1.0 - positive), dtype=float)  # type: ignore[no-any-return]


def labels_from_probabilities(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert binary probabilities to hard labels."""

    positive = extract_positive_class_probabilities(probabilities)
    return np.asarray(positive >= threshold, dtype=int)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute predictive entropy for binary probabilities."""

    positive = extract_positive_class_probabilities(probabilities)
    entropy = -(positive * np.log(positive) + (1.0 - positive) * np.log(1.0 - positive))
    return np.asarray(entropy, dtype=float)
