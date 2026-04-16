# Benchmarking Guide

ShiftStat V5 adds a lightweight benchmark framework for repeated-seed studies on reliability under shift.

## Core ideas

- Define a `BenchmarkScenario` instead of hand-writing loops around datasets, seeds, and baselines.
- Use the `BaselineRegistry` to compare raw evaluation, weighting, recalibration, subgroup-aware auditing, and abstention policies in a consistent format.
- Aggregate results across seeds with `BenchmarkRunner` and export CSV, LaTeX, markdown, and figure artifacts directly from `BenchmarkResult`.

## Built-in scenario families

- Covariate-shift severity sweeps
- Subgroup-degradation scenarios
- Calibration-drift scenarios
- Selective-prediction scenarios
- Mixed synthetic tabular settings with configurable dimensions, noise, imbalance, and shift patterns

## Minimal example

```python
from shiftstat.bench import BenchmarkRunner, make_covariate_shift_sweep_scenario

scenario = make_covariate_shift_sweep_scenario(
    severities=[0.2, 0.8, 1.4],
    seeds=[7, 19, 43],
)

result = BenchmarkRunner().run(scenario)
print(result.aggregate_frame())
print(result.to_markdown())
```

## Practical guidance

- Keep benchmark families narrow and scientifically motivated.
- Prefer interpretable synthetic patterns over overly realistic but opaque generators.
- Use multiple seeds for publication claims and treat single-seed runs as exploratory only.
- Read subgroup and abstention metrics alongside aggregate metrics, not after them.
