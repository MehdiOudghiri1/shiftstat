# User Guide

## Typical V2 workflow

1. Detect whether covariates shifted between reference and target data.
2. Fit an importance weighting model if a covariate-shift correction is appropriate.
3. Evaluate reference and target reliability with `ReliabilityAnalyzer` or `evaluate_under_shift(...)`.
4. Fit a post-hoc calibrator when deployment reliability is degraded.
5. Compare target reliability before and after recalibration.

## Practical guidance

- Start with the unweighted target profile when target labels are available.
- Use weighted reference diagnostics to understand whether covariate correction changes the reliability picture.
- Prefer temperature scaling as a stable baseline, and isotonic calibration when enough calibration data is available.
- Inspect both scalar metrics and confidence-conditioned tables; a single ECE value can hide concentrated high-confidence failures.

