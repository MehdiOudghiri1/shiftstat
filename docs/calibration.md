# Calibration Under Shift

`shiftstat.calibration` provides post-hoc reliability analysis for binary classification probabilities when the deployment distribution differs from the reference distribution.

## Included functionality

- Expected Calibration Error and Maximum Calibration Error
- Weighted calibration analysis using importance weights
- Bin-wise probability summaries and calibration curves
- Brier decomposition-oriented diagnostics
- Post-hoc recalibration with:
  - temperature scaling
  - isotonic regression
  - logistic or Platt calibration

## Main entry points

- `CalibrationEvaluator`
- `TemperatureScaler`
- `IsotonicCalibrator`
- `PlattCalibrator`
- `compare_calibration(...)`

