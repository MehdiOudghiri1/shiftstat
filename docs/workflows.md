# Model Evaluation Workflows

ShiftStat now provides high-level workflows for reliability analysis and selective deployment under distribution shift.

## Entry point

Use `evaluate_under_shift(...)` to:

- fit or consume a classifier
- evaluate reference and target reliability
- optionally estimate importance weights
- optionally fit a recalibration model
- compare pre- and post-recalibration reliability
- export a structured report

This layer is intentionally thin: it coordinates the existing detection, weighting, calibration, and reliability modules without hiding the intermediate objects.

## Selective workflow

Use `evaluate_selective_under_shift(...)` to:

- fit or consume a classifier
- tune or apply an abstention policy
- evaluate risk-coverage tradeoffs on the target distribution
- compare weighted and unweighted threshold tuning
- summarize subgroup-specific rejection behavior
- export a selective deployment report

This workflow builds on the same modular pieces as the reliability stack: model fitting, optional importance weighting, optional recalibration, and transparent post-hoc evaluation.

## Benchmark and experiment workflow

Use `BenchmarkRunner` when you want direct Python control over scenario definitions, repeated seeds, and publication metrics.

Use `run_experiment(...)` or the `shiftstat-experiment` CLI when you want:

- config-driven runs
- artifact directories with manifests
- markdown summaries and logs
- paper-ready figures and LaTeX tables

These V5 layers remain intentionally lightweight. They orchestrate the existing scientific modules rather than hiding them.
