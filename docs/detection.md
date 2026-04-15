# Detection module

`shiftstat.detect` provides feature-wise drift statistics and dataset-level summaries.

## Included methods

- Kolmogorov-Smirnov tests for continuous features
- Chi-square tests for categorical features
- Population Stability Index style drift scores
- Wasserstein distances for continuous discrepancy
- Classifier two-sample testing via source-target discrimination

## Main entry point

Use `ShiftDetector` to fit on a reference and deployment dataset, then inspect `summary()` or render plots.

