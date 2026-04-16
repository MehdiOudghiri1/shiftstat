# Developer note

## Benchmark and experiment architecture choices

- `shiftstat.bench` is intentionally lightweight:
  - `BenchmarkScenario` defines a seeded scenario family and case grid
  - `BaselineRegistry` holds named comparison baselines
  - `MetricRegistry` tracks benchmark metric semantics
  - `BenchmarkRunner` executes repeated-seed studies and aggregates outputs
  - `BenchmarkResult` exports CSV, markdown, LaTeX, and figure artifacts
- `shiftstat.experiments` wraps that core with JSON/YAML configs, manifests, logs, artifact directories, and a minimal CLI.
- V5 reuses the V1-V4 scientific stack rather than replacing it:
  - reliability workflows for raw, weighted, and recalibrated evaluation
  - V3 subgroup auditing for hidden-failure baselines
  - V4 selective workflows for abstention comparisons

## Reproducibility guarantees

- Scenario generators are seed-controlled and benchmark aggregation is deterministic given seeds, configs, and library behavior.
- Experiment runs persist copied configs, logs, markdown summaries, JSON manifests, run-level CSVs, aggregated CSVs, and figure/table outputs.
- The `paper_assets/` directory now provides reproducible configs and generated-output inventory so manuscript assets can be traced back to exact experiment manifests.

## Public release polish summary

- Added benchmark and experiment docs pages, API reference pages, and publication-workflow guidance.
- Added benchmark-oriented examples and a V5 benchmark-suite entrypoint.
- Added `CHANGELOG.md`, `CITATION.cff`, issue templates, a PR template, and a release workflow.
- Extended CI with a docs build step so documentation is checked in automation.

## Recommendations for preprint-ready usage

- Run final experiments from committed YAML/JSON configs under `paper_assets/configs/`.
- Use at least two to three seeds for manuscript figures and keep the exact generated artifact directories under version control or archival storage.
- Report aggregate, worst-group, and selective metrics together when claims involve deployment robustness.
- Treat synthetic benchmark conclusions as controlled stress tests, not direct empirical substitutes for real deployment data.

## Deferred items for V6

- uncertainty intervals and statistical testing for aggregated benchmark comparisons
- richer benchmark datasets beyond synthetic tabular settings
- multiclass and regression benchmark families
- stronger publication tooling for automatic appendix assembly
