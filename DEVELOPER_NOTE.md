# Developer note

## API changes

- Added `shiftstat.calibration` with:
  - `CalibrationEvaluator`
  - `TemperatureScaler`
  - `IsotonicCalibrator`
  - `PlattCalibrator`
  - `compare_calibration(...)`
- Added `shiftstat.reliability` with:
  - `ReliabilityAnalyzer`
  - `ReliabilityProfile`
  - `ReliabilityShiftReport`
  - `ShiftEvaluationResult`
  - `evaluate_under_shift(...)`
- Extended `shiftstat.metrics` with calibration and reliability-oriented metrics and summaries
- Extended `shiftstat.plotting` with reliability diagrams, calibration comparisons, confidence histograms, and confidence-error curves

## Refactors introduced

- Package `__init__` files for `detect`, `calibration`, and `reliability` now use lazy exports to avoid circular imports while preserving the public API
- Reliability reporting is represented through structured dataclasses rather than ad hoc dictionaries
- Workflow orchestration is kept in a dedicated module so detection, weighting, calibration, and reporting remain separable

## Scientific rationale of V2

V1 established how much covariates move and how to reweight source samples under covariate shift. V2 asks a sharper question: when predictive probabilities are deployed under shift, how do calibration and confidence-conditioned reliability degrade, and how much can weighting or post-hoc recalibration change that conclusion?

This motivates:

- weighted and unweighted calibration analysis
- explicit comparison between reference-domain and target-domain reliability
- structured pre- and post-recalibration evaluation
- benchmark scenarios centered on reliability degradation rather than only shift detectability

## Deferred items for V3

- multiclass calibration and reliability support
- richer regression reliability diagnostics
- subgroup and conditional robustness reporting
- selective prediction and abstention analysis
- online monitoring and sequential reliability tracking
- larger benchmark suites and richer experiment aggregation
