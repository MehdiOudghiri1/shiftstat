# Generating Tables And Figures For Papers

ShiftStat V5 can export paper-friendly artifacts directly from benchmark results.

## From Python

```python
from shiftstat.bench import BenchmarkRunner, make_covariate_shift_sweep_scenario

scenario = make_covariate_shift_sweep_scenario()
result = BenchmarkRunner().run(scenario)
paths = result.export_artifacts("paper_assets/generated/covariate_sweep")
```

## From config

```bash
shiftstat-experiment paper_assets/configs/publication_suite.yaml
```

## Typical outputs

- CSV tables for downstream analysis
- LaTeX tables per publication metric
- summary markdown reports
- figure files suitable for arXiv or software-paper drafts

## Suggested manuscript mapping

- Main text: aggregated summary figures and selected LaTeX tables
- Appendix: per-seed CSVs, full manifests, and experiment configs
- Repository companion material: complete generated directories under `paper_assets/generated/`
