"""Convenience helpers for reference-target splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shiftstat.typing import TabularLike, VectorLike
from shiftstat.utils.random import random_state_to_int


@dataclass(frozen=True)
class ReferenceTargetSplit:
    """Container for a reference-target split used in demos and examples."""

    X_ref: pd.DataFrame | np.ndarray
    X_target: pd.DataFrame | np.ndarray
    y_ref: np.ndarray | None = None
    y_target: np.ndarray | None = None


def make_reference_target_split(
    X: TabularLike,
    y: VectorLike | None = None,
    *,
    target_size: float = 0.3,
    random_state: int | np.random.RandomState | None = None,
    stratify: VectorLike | None = None,
) -> ReferenceTargetSplit:
    """Split tabular data into reference and target partitions."""

    seed = random_state_to_int(random_state)
    if y is None:
        X_ref, X_target = train_test_split(
            X,
            test_size=target_size,
            random_state=seed,
            stratify=stratify,
        )
        return ReferenceTargetSplit(X_ref=X_ref, X_target=X_target)

    X_ref, X_target, y_ref, y_target = train_test_split(
        X,
        y,
        test_size=target_size,
        random_state=seed,
        stratify=stratify,
    )
    return ReferenceTargetSplit(
        X_ref=X_ref,
        X_target=X_target,
        y_ref=np.asarray(y_ref),
        y_target=np.asarray(y_target),
    )
