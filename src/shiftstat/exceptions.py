"""Custom exceptions and warnings for ShiftStat."""


class ShiftStatError(Exception):
    """Base exception for ShiftStat."""


class ValidationError(ShiftStatError):
    """Raised when tabular inputs fail validation."""


class SchemaMismatchError(ValidationError):
    """Raised when reference and target schemas are incompatible."""


class NotFittedError(ShiftStatError):
    """Raised when an estimator-like object is used before fitting."""


class ShiftStatWarning(UserWarning):
    """Base warning class for ShiftStat."""


class NumericalStabilityWarning(ShiftStatWarning):
    """Warning emitted when numerical safeguards are activated."""

