# Paper assets

This directory is the V5 bridge between ShiftStat benchmarks and a future preprint or software-paper workflow.

## Layout

- `configs/`: reproducible experiment manifests used to generate benchmark tables and figures
- `generated/`: ignored local outputs produced by the experiment runner
- `inventory.md`: short map from experiments to figures and tables

Generated assets are not kept in git. The committed source of truth is the
config file plus the code version used to run it.

## Reproduction

```bash
shiftstat-experiment paper_assets/configs/publication_suite.yaml
```

Generated artifacts are intentionally ordinary files so they can be inspected, versioned, and cited in appendices.
For archival releases, attach generated outputs to a release or external
archive rather than committing local benchmark payloads to the library tree.
