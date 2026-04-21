# Artifact Inventory

This directory keeps reproducible experiment configs, not generated result
payloads. Generated CSVs, JSON manifests, logs, LaTeX tables, and benchmark
figures are intentionally ignored by git under `paper_assets/generated/`.

## Publication Suite

- Config: `paper_assets/configs/publication_suite.yaml`
- Default output root: `paper_assets/generated/publication_suite/`
- Purpose: repeated-seed benchmark suite spanning covariate shift, subgroup
  failures, and selective prediction.

Run with:

```bash
shiftstat-experiment paper_assets/configs/publication_suite.yaml
```

Typical regenerated files include:

- scenario-level run CSVs
- summary CSVs and markdown reports
- LaTeX tables
- figure files
- portable artifact manifests with checksums

## Subgroup Case Study

- Config: `paper_assets/configs/subgroup_case_study.yaml`
- Default output root: `paper_assets/generated/subgroup_case_study/`
- Purpose: smaller hidden-failure case study for figure and table generation.

Run with:

```bash
shiftstat-experiment paper_assets/configs/subgroup_case_study.yaml
```
