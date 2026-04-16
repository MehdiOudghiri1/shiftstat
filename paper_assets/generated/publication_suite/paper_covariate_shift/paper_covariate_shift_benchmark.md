## Benchmark summary

- Scenario: paper_covariate_shift
- Family: covariate_shift_sweep
- Seeds: 7, 19
- Baselines: raw_model, weighting_only, recalibration_only, confidence_abstention

### Publication metrics

- Delta ECE: best baseline `recalibration_only` on `severity=1.40` with mean -0.075
- Delta accuracy: best baseline `recalibration_only` on `severity=1.40` with mean 0.045
- Selective risk reduction: best baseline `confidence_abstention` on `severity=0.20` with mean 0.052