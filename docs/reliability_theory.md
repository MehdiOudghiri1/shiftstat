# Theory Notes on Weighted Calibration and Reliability

Under covariate shift, the target risk is not generally estimated well by naïve source-domain averages. Importance weighting re-expresses source observations under the target covariate distribution, which makes weighted calibration summaries useful when direct target labels are limited or when one wants to study how reliability conclusions change after covariate correction.

ShiftStat V2 exposes both unweighted and weighted calibration diagnostics because they answer different questions:

- unweighted source calibration measures reliability on the observed reference distribution
- weighted source calibration approximates reliability under the target covariate distribution
- target calibration measures empirical deployment reliability directly when target labels are available

These diagnostics do not imply causal transport guarantees. They are descriptive statistical tools whose validity depends on the assumptions of the chosen weighting method and the available labeled data.

