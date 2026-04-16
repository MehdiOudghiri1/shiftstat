## Benchmark summary

- Scenario: paper_subgroup_failures
- Family: subgroup_degradation
- Seeds: 11, 23
- Baselines: raw_model, subgroup_audit, weighted_confidence_abstention

### Publication metrics

- Worst-group accuracy gap: best baseline `subgroup_audit` on `masked subgroup shift` with mean 0.137
- Worst-group calibration gap: best baseline `subgroup_audit` on `masked subgroup shift` with mean 0.182
- Selective risk reduction: best baseline `weighted_confidence_abstention` on `minority subgroup degradation` with mean 0.027