## Experiment summary

- Experiment: paper_publication_suite
- Config: `C:\Users\medhi.oudghiri\OneDrive - Setec\temp\librairie_open_source\shiftstat\paper_assets\configs\publication_suite.yaml`
- Output directory: `C:\Users\medhi.oudghiri\OneDrive - Setec\temp\librairie_open_source\shiftstat\paper_assets\generated\publication_suite`
- Scenarios: 3

### paper_covariate_shift

## Benchmark summary

- Scenario: paper_covariate_shift
- Family: covariate_shift_sweep
- Seeds: 7, 19
- Baselines: raw_model, weighting_only, recalibration_only, confidence_abstention

### Publication metrics

- Delta ECE: best baseline `recalibration_only` on `severity=1.40` with mean -0.075
- Delta accuracy: best baseline `recalibration_only` on `severity=1.40` with mean 0.045
- Selective risk reduction: best baseline `confidence_abstention` on `severity=0.20` with mean 0.052

### paper_subgroup_failures

## Benchmark summary

- Scenario: paper_subgroup_failures
- Family: subgroup_degradation
- Seeds: 11, 23
- Baselines: raw_model, subgroup_audit, weighted_confidence_abstention

### Publication metrics

- Worst-group accuracy gap: best baseline `subgroup_audit` on `masked subgroup shift` with mean 0.137
- Worst-group calibration gap: best baseline `subgroup_audit` on `masked subgroup shift` with mean 0.182
- Selective risk reduction: best baseline `weighted_confidence_abstention` on `minority subgroup degradation` with mean 0.027

### paper_selective_shift

## Benchmark summary

- Scenario: paper_selective_shift
- Family: selective_shift
- Seeds: 3, 13
- Baselines: confidence_abstention, weighted_confidence_abstention, recalibrated_confidence_abstention

### Publication metrics

- Selective risk reduction: best baseline `recalibrated_confidence_abstention` on `severity=0.40` with mean 0.079
- Selective risk: best baseline `recalibrated_confidence_abstention` on `severity=1.00` with mean 0.151
- Retained coverage: best baseline `recalibrated_confidence_abstention` on `severity=1.00` with mean 0.759
