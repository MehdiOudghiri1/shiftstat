# Reproducibility Guide

ShiftStat V5 is designed to make repeated scientific runs easy to replay and audit.

## Reproducibility guarantees

- Scenario generators are seed-controlled.
- Benchmark aggregation is deterministic given scenario definitions, seeds, and baselines.
- Experiment manifests preserve the original config path, copied config content, and generated artifact locations.
- Output directories include markdown, CSV, JSON, and figure files that can be archived alongside a preprint.

## Recommended practice

- Pin package versions before producing final paper assets.
- Commit the exact experiment config used for each table and figure.
- Keep generated benchmark outputs in a dedicated directory such as `paper_assets/generated/`.
- Use multiple seeds for headline comparisons and avoid selecting a single favorable seed for publication.

## Limits

- Determinism assumes deterministic upstream estimator behavior for the chosen libraries and environment.
- Small numerical differences can still appear across platforms or library versions.
- The framework does not yet provide bootstrap confidence intervals or cluster-aware uncertainty accounting.
