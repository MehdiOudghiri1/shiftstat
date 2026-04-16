# Paper assets

This directory is the V5 bridge between ShiftStat benchmarks and a future preprint or software-paper workflow.

## Layout

- `configs/`: reproducible experiment manifests used to generate benchmark tables and figures
- `generated/`: saved outputs produced by the experiment runner
- `inventory.md`: short map from experiments to figures and tables

Generated assets already included in this repository:

- `generated/publication_suite/`
- `generated/subgroup_case_study/`

## Reproduction

```bash
shiftstat-experiment paper_assets/configs/publication_suite.yaml
```

Generated artifacts are intentionally ordinary files so they can be inspected, versioned, and cited in appendices.
