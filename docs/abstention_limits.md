# When Abstention Helps And Misleads

Abstention can help when uncertainty scores rank failures reasonably well. It can mislead when:

- rejected samples are operationally critical even if accepted-set risk drops
- low-risk majority slices dominate the accepted set
- the abstention score is poorly aligned with deployment errors under shift
- calibration changes after selection are ignored

Recommended interpretation:

1. compare full-set and accepted-set metrics together
2. inspect subgroup-specific abstention rates
3. check whether accepted-set calibration improved or merely shifted
4. report the rejected fraction as part of the deployment decision
