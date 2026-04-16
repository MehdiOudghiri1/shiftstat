"""Synthetic datasets for examples, tests, and benchmarks."""

from shiftstat.datasets.synthetic import (
    SyntheticShiftDataset,
    make_configurable_shift_classification,
    make_covariate_shift_classification,
    make_covariate_shift_regression,
    make_mixed_type_shift,
    make_severity_controlled_shift,
)
from shiftstat.datasets.audit_synthetic import make_hidden_subgroup_shift_classification

__all__ = [
    "SyntheticShiftDataset",
    "make_configurable_shift_classification",
    "make_covariate_shift_classification",
    "make_covariate_shift_regression",
    "make_hidden_subgroup_shift_classification",
    "make_mixed_type_shift",
    "make_severity_controlled_shift",
]
