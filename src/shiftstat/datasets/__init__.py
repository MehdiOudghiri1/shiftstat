"""Synthetic datasets for examples, tests, and benchmarks."""

from shiftstat.datasets.synthetic import (
    SyntheticShiftDataset,
    make_covariate_shift_classification,
    make_covariate_shift_regression,
    make_mixed_type_shift,
    make_severity_controlled_shift,
)

__all__ = [
    "SyntheticShiftDataset",
    "make_covariate_shift_classification",
    "make_covariate_shift_regression",
    "make_mixed_type_shift",
    "make_severity_controlled_shift",
]

