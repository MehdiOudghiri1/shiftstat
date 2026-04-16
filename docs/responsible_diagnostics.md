# Interpreting Subgroup Diagnostics Responsibly

ShiftStat V3 is a scientific auditing toolkit, not a fairness-compliance engine.

Keep the following points explicit when reporting results:

- subgroup gaps are descriptive statistical findings
- discovered slices are hypotheses for further investigation
- low-support groups require caution even when effect sizes look large
- worst-group results should be read beside deployment exposure and coverage diagnostics
- domain knowledge is required before translating statistical findings into operational action

Good practice:

1. report aggregate and disaggregated metrics together
2. show support thresholds and unsupported coverage
3. describe slice discovery as exploratory
4. avoid causal or legal claims not supported by the analysis
