# Risk-Coverage Analysis

Risk-coverage analysis describes how accepted-set risk changes as the abstention threshold moves.

In ShiftStat V4, the risk-coverage curve is computed directly from selection scores:

1. score each prediction with a policy-specific safety score
2. sweep candidate thresholds
3. recompute coverage and accepted-set metrics at each threshold

This gives a transparent operating curve for:

- fixed-threshold deployment
- target coverage selection
- target risk selection
- comparison between weighted and unweighted tuning

The `RiskCoverageCurve` object is machine-readable and plottable through the selective plotting helpers.
