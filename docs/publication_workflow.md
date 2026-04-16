# Publication Workflow Guide

ShiftStat V5 aims to support a clean path from benchmark idea to paper-ready artifact.

## Recommended workflow

1. Prototype the scenario with `BenchmarkRunner`.
2. Move the finalized setup into a JSON or YAML experiment config.
3. Run the experiment through `run_experiment(...)` or the CLI.
4. Inspect CSV summaries and markdown reports.
5. Reuse the generated LaTeX tables and figures in the manuscript.

## Artifact types

- `*_runs.csv` for appendix-level detail
- `*_summary.csv` for main-table aggregation
- `tables/*.tex` for manuscript integration
- `figures/*` for plots
- `*_benchmark.md` and experiment summaries for narrative scaffolding

## Interpreting outputs responsibly

- Treat benchmark figures as summaries of the configured scenarios, not universal claims.
- Use subgroup and abstention outputs to surface operational tradeoffs, not to overstate fairness or safety guarantees.
- When aggregate and worst-group conclusions diverge, report both.
