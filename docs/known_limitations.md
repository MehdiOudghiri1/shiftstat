# Known Limitations

ShiftStat is designed for transparent scientific diagnostics under tabular
distribution shift. It does not replace domain validation, causal analysis, or
deployment monitoring.

## Statistical Scope

- Most high-level workflows assume tabular binary classification unless a
  function explicitly documents regression support.
- Importance weighting can reduce covariate-shift bias only when the target
  support is sufficiently represented in the reference data.
- Low local effective sample size can make subgroup and bin-level alarms look
  dramatic while remaining statistically unsupported.
- Adaptive slice discovery should be treated as exploratory unless the search
  procedure is separated from confirmatory inference.
- Synthetic benchmark conclusions are controlled stress tests, not evidence of
  performance on a specific deployment population.

## Engineering Scope

- Plotting APIs are convenience helpers; exact visual styling is not a stable
  contract.
- Benchmark and experiment artifact schemas may evolve before `1.0`.
- External real-data examples may require optional dependencies and network or
  dataset access outside the base package.

## Responsible Use

When using ShiftStat in applied work:

1. report support, local ESS, and target exposure beside subgroup alarms
2. separate exploratory discovery from confirmatory claims
3. retain seeds, configs, package version, Python version, and dependency
   versions for reproducibility
4. validate important findings on real deployment-like data whenever possible
