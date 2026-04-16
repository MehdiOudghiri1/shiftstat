"""Utility helpers for validation, schemas, randomness, and splitting."""

from shiftstat.utils.probabilities import (
    confidence_from_probabilities,
    extract_positive_class_probabilities,
    labels_from_probabilities,
    predictive_entropy,
)
from shiftstat.utils.random import check_random_state
from shiftstat.utils.schema import (
    align_tabular_inputs,
    extract_feature_names,
    infer_feature_types,
    validate_tabular_pair_schema,
)
from shiftstat.utils.split import make_reference_target_split
from shiftstat.utils.validation import (
    ensure_1d,
    ensure_2d_tabular,
    validate_same_length,
)

__all__ = [
    "align_tabular_inputs",
    "check_random_state",
    "confidence_from_probabilities",
    "ensure_1d",
    "ensure_2d_tabular",
    "extract_positive_class_probabilities",
    "extract_feature_names",
    "infer_feature_types",
    "labels_from_probabilities",
    "make_reference_target_split",
    "predictive_entropy",
    "validate_same_length",
    "validate_tabular_pair_schema",
]
