# Artifact inventory

## publication_suite

- Config: `paper_assets/configs/publication_suite.yaml`
- Purpose: main repeated-seed benchmark suite spanning covariate shift, subgroup failures, and selective prediction
- Manifest: `paper_assets/generated/publication_suite/paper_publication_suite_manifest.json`
- Summary markdown: `paper_assets/generated/publication_suite/paper_publication_suite_summary.md`
- Main figure examples:
  - `paper_assets/generated/publication_suite/paper_covariate_shift/figures/paper_covariate_shift_delta_ece.png`
  - `paper_assets/generated/publication_suite/paper_subgroup_failures/figures/paper_subgroup_failures_worst_group_accuracy_gap.png`
  - `paper_assets/generated/publication_suite/paper_selective_shift/figures/paper_selective_shift_target_risk_reduction.png`
- Main table examples:
  - `paper_assets/generated/publication_suite/paper_covariate_shift/tables/paper_covariate_shift_delta_ece.tex`
  - `paper_assets/generated/publication_suite/paper_subgroup_failures/tables/paper_subgroup_failures_worst_group_accuracy_gap.tex`
  - `paper_assets/generated/publication_suite/paper_selective_shift/tables/paper_selective_shift_target_risk_reduction.tex`

## subgroup_case_study

- Config: `paper_assets/configs/subgroup_case_study.yaml`
- Purpose: smaller hidden-failure case study for figure and table generation
- Manifest: `paper_assets/generated/subgroup_case_study/paper_subgroup_case_study_manifest.json`
- Summary markdown: `paper_assets/generated/subgroup_case_study/paper_subgroup_case_study_summary.md`
- Figure:
  - `paper_assets/generated/subgroup_case_study/paper_hidden_failure_case/figures/paper_hidden_failure_case_worst_group_accuracy_gap.png`
