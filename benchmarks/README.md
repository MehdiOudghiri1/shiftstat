# Benchmark scaffolding

ShiftStat V2 adds the first reliability-focused benchmark runner.

Included here:

- `run_synthetic.py`: a synthetic experiment runner
- `run_reliability_benchmark.py`: a reliability degradation severity sweep
- `configs/`: JSON experiment configurations
- `results/`: persisted benchmark outputs

The reliability benchmark compares:

- no correction
- weighting only
- recalibration only
- weighting plus recalibration

across a controlled covariate-shift severity sweep with a stable synthetic Bayes rule.
