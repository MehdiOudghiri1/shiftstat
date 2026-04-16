# Practical Deployment Guidance

Selective deployment should be framed as a conservative operating policy, not a guarantee.

Useful practice patterns:

- tune thresholds on a held-out reference validation set
- use weighted tuning when covariate shift estimates suggest the reference set is not deployment-like
- compare raw and recalibrated probabilities when thresholds depend on confidence
- audit subgroup rejection patterns before deployment
- monitor both accepted-set quality and rejected-set volume after launch

ShiftStat V4 is designed to support these checks with explicit threshold summaries, subgroup abstention tables, and accepted-set reliability diagnostics.
