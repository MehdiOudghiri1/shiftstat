"""Post-hoc probability recalibration models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from shiftstat.exceptions import NotFittedError
from shiftstat.metrics import negative_log_likelihood
from shiftstat.utils.probabilities import extract_positive_class_probabilities
from shiftstat.utils.validation import ensure_1d, validate_same_length


class _BaseProbabilityCalibrator:
    """Shared interface for binary probability calibrators."""

    def fit(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> _BaseProbabilityCalibrator:
        raise NotImplementedError

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using a fitted calibrator."""

        return self.predict_proba(probabilities)

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before use.")


class TemperatureScaler(_BaseProbabilityCalibrator):
    """Temperature scaling on binary probabilities via logit rescaling."""

    def __init__(self, *, bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.bounds = bounds

    def fit(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> TemperatureScaler:
        """Fit the temperature parameter by minimizing weighted log loss."""

        prob_arr = extract_positive_class_probabilities(probabilities)
        y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
        validate_same_length(prob_arr, y_true_arr)
        logit = np.log(prob_arr / (1.0 - prob_arr))

        def objective(temperature: float) -> float:
            scaled = 1.0 / (1.0 + np.exp(-(logit / temperature)))
            return negative_log_likelihood(y_true_arr, scaled, sample_weight=sample_weight)

        result = minimize_scalar(objective, bounds=self.bounds, method="bounded")
        self.temperature_ = float(result.x if result.success else 1.0)
        self._is_fitted = True
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to binary probabilities."""

        self._check_is_fitted()
        prob_arr = extract_positive_class_probabilities(probabilities)
        logit = np.log(prob_arr / (1.0 - prob_arr))
        return np.asarray(1.0 / (1.0 + np.exp(-(logit / self.temperature_))), dtype=float)


class IsotonicCalibrator(_BaseProbabilityCalibrator):
    """Binary isotonic regression calibrator."""

    def __init__(self) -> None:
        self.model_ = IsotonicRegression(out_of_bounds="clip")

    def fit(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> IsotonicCalibrator:
        """Fit isotonic regression on predicted probabilities."""

        prob_arr = extract_positive_class_probabilities(probabilities)
        y_true_arr = ensure_1d(y_true, name="y_true").astype(float)
        validate_same_length(prob_arr, y_true_arr)
        self.model_.fit(prob_arr, y_true_arr, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""

        self._check_is_fitted()
        prob_arr = extract_positive_class_probabilities(probabilities)
        return np.asarray(self.model_.predict(prob_arr), dtype=float)


class PlattCalibrator(_BaseProbabilityCalibrator):
    """Logistic recalibration baseline on probability logits."""

    def __init__(self) -> None:
        self.model_ = LogisticRegression(max_iter=3000)

    def fit(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> PlattCalibrator:
        """Fit logistic calibration on binary probability logits."""

        prob_arr = extract_positive_class_probabilities(probabilities)
        y_true_arr = ensure_1d(y_true, name="y_true").astype(int)
        validate_same_length(prob_arr, y_true_arr)
        logits = np.log(prob_arr / (1.0 - prob_arr)).reshape(-1, 1)
        self.model_.fit(logits, y_true_arr, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply logistic recalibration."""

        self._check_is_fitted()
        prob_arr = extract_positive_class_probabilities(probabilities)
        logits = np.log(prob_arr / (1.0 - prob_arr)).reshape(-1, 1)
        return np.asarray(self.model_.predict_proba(logits)[:, 1], dtype=float)
