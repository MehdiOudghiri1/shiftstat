"""ShiftStat public package API."""

from shiftstat._version import __version__
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import (
    ImportanceWeighter,
    compute_effective_sample_size,
    weighted_mean,
    weighted_risk,
)

__all__ = [
    "ImportanceWeighter",
    "ShiftDetector",
    "__version__",
    "compute_effective_sample_size",
    "weighted_mean",
    "weighted_risk",
]

