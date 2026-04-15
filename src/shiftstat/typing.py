"""Project-wide type aliases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayLike: TypeAlias = npt.NDArray[np.generic] | pd.DataFrame | pd.Series | Sequence[Any]
TabularLike: TypeAlias = npt.NDArray[np.generic] | pd.DataFrame
VectorLike: TypeAlias = npt.NDArray[np.generic] | pd.Series | Sequence[float] | Sequence[int]
FeatureName: TypeAlias = str
FeatureTypes: TypeAlias = dict[str, str]

