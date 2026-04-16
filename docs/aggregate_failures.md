# Why Aggregate Metrics Can Hide Deployment Failures

Aggregate deployment metrics answer an important question, but not the whole question. A target-domain accuracy or calibration score can still look acceptable even when a smaller operational slice deteriorates sharply.

This happens when:

- the failing slice has low prevalence
- the majority slice remains stable or improves
- calibration drift is concentrated in specific feature regions
- deployment exposure shifts toward slices the reference analysis barely covered

ShiftStat V3 is built around this observation. The goal is not to replace aggregate metrics, but to put them beside:

- subgroup degradation tables
- conditional error summaries
- worst-group comparisons
- slice discovery outputs
- support and coverage caveats

Recommended workflow:

1. Start with the aggregate reliability summary.
2. Inspect subgroup degradation and support diagnostics.
3. Compare aggregate conclusions with worst-group outcomes.
4. Use slice discovery to localize concentrated failures.
5. Report statistical caveats alongside any high-risk findings.
