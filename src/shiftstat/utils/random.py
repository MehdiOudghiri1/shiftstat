"""Random-state management helpers."""

from __future__ import annotations

import numpy as np


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn a seed into a :class:`numpy.random.RandomState`.

    Parameters
    ----------
    seed:
        `None`, an integer seed, or an existing random state.

    Returns
    -------
    numpy.random.RandomState
        A reproducible random state instance.
    """

    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(int(seed))
    raise TypeError(f"Cannot create random state from seed of type {type(seed)!r}.")


def random_state_to_int(seed: int | np.random.RandomState | None) -> int | None:
    """Convert a supported random state representation to an integer seed when possible."""

    if seed is None:
        return None
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    if isinstance(seed, np.random.RandomState):
        return int(seed.randint(0, 2**31 - 1))
    raise TypeError(f"Unsupported random state type: {type(seed)!r}.")
