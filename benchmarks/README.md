# Benchmarks

ShiftStat V5 turns the benchmark layer into a lightweight experiment platform for repeated-seed scientific studies.

## Included entry points

- `run_reliability_benchmark.py`: legacy V2-style severity sweep
- `run_v3_audit_benchmark.py`: subgroup-aware audit benchmark
- `run_selective_benchmark.py`: V4 selective benchmark
- `run_v5_benchmark_suite.py`: config-driven V5 benchmark and artifact runner

## Configs

- `configs/reliability_sweep.json`
- `configs/v3_audit_benchmark.json`
- `configs/selective_benchmark.json`
- `configs/v5_publication_suite.yaml`

## V5 benchmark focus

- covariate-shift severity sweeps
- subgroup-specific degradation scenarios
- calibration-drift scenarios
- selective-prediction scenarios
- publication-friendly figure and table generation

For manuscript-oriented runs, see `paper_assets/configs/` and the `shiftstat-experiment` CLI.
