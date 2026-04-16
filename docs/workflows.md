# Model Evaluation Workflows

ShiftStat V2 adds a high-level workflow for evaluating a classifier under distribution shift.

## Entry point

Use `evaluate_under_shift(...)` to:

- fit or consume a classifier
- evaluate reference and target reliability
- optionally estimate importance weights
- optionally fit a recalibration model
- compare pre- and post-recalibration reliability
- export a structured report

This layer is intentionally thin: it coordinates the existing detection, weighting, calibration, and reliability modules without hiding the intermediate objects.

