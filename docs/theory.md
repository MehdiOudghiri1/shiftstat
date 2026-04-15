# Scientific Motivation

Distribution shift breaks the assumptions that support standard empirical risk estimation. In the tabular setting, this often appears as a mismatch between the training distribution and the deployment distribution across continuous and categorical covariates.

ShiftStat V1 focuses on two scientific tasks:

1. Detecting and ranking which features have changed.
2. Estimating importance weights that re-express source-domain observations under a target-domain covariate distribution.

The package intentionally keeps inferential tests, effect sizes, weighting models, and reporting abstractions separate so users can inspect each stage of the workflow.

