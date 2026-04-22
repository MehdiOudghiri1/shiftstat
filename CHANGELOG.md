# Changelog

## 0.6.3

- Rendered README scientific equations as LaTeX-generated SVG assets for a more
  polished PyPI project page.
- Added a reproducible equation-rendering helper for regenerating the README
  equation assets from TeX source.
- Updated citation metadata to point at the public GitHub repository.

## 0.6.0

- Added `shiftstat.certification` for certified worst-group reliability audits under covariate shift.
- Added `CertifiedWorstGroupAuditor`, `CertifiedAuditReport`, `CertifiedCellResult`, and `AuditDecision`.
- Added local effective sample size, simultaneous radii, certified excess, and learned-weight sensitivity utilities.
- Added `CrossFittedImportanceWeighter` for label-independent source-side density-ratio estimation.
- Added documentation for certified worst-group audits and exposed the certification API in the public package.
- Added tests covering local ESS, certification decisions, nuisance radii, and cross-fitted weighting.

## 0.5.0

- Added the new `shiftstat.bench` benchmark framework with scenario presets, baseline and metric registries, repeated-seed aggregation, publication-friendly tables, and figure exports.
- Added the new `shiftstat.experiments` module with JSON/YAML config parsing, CLI execution, manifests, logs, and artifact directories.
- Added V5 benchmark examples, a benchmark-suite entrypoint, new docs for benchmarking and publication workflows, and paper-asset scaffolding.
- Added configurable mixed synthetic benchmark datasets and benchmark plotting helpers.
- Expanded repository polish with citation metadata, changelog, issue/PR templates, and release workflow scaffolding.

## 0.4.0

- Added selective prediction and abstention workflows.
- Added abstention policies, threshold tuning, risk-coverage analysis, and subgroup-aware rejection summaries.
- Added selective deployment reports, plots, examples, and benchmarks.

## 0.3.0

- Added subgroup-aware reliability analysis and conditional auditing.
- Added slice discovery, hidden-failure diagnostics, audit reports, and subgroup plots.
- Added audit-oriented examples, docs, and benchmarks.

## 0.2.0

- Added reliability workflows, weighting-aware evaluation, and recalibration support.

## 0.1.0

- Initial public release with tabular shift detection, weighting primitives, and core metrics.
