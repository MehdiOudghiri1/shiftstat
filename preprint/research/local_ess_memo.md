# Local ESS Memo

## Candidate novelty angle

The current draft treats effective sample size as a global property of the importance weights. A sharper and potentially more novel claim is that **worst-group reliability under covariate shift is governed by subgroup-bin-local effective sample size, not by global effective sample size alone**.

This matters because calibration is not a single weighted average. It is built from many subgroup-bin residuals. If even one operational subgroup-bin has poor overlap, that term can become statistically unstable long before the aggregate weighted sample looks obviously weak.

## Core theoretical object

For a score bin `b` and subgroup `g`, let

```text
A_i(b, g) = 1{ s(X_i) in I_b } g(X_i)
```

and define the self-normalized weighted residual mean

```text
R_hat(b, g)
=
sum_i w_i (Y_i - s(X_i)) A_i(b, g)
/
sum_i w_i A_i(b, g).
```

The corresponding subgroup-bin-local effective sample size is

```text
ESS(b, g)
=
(sum_i w_i A_i(b, g))^2
/
sum_i w_i^2 A_i(b, g).
```

If the conditional residual variance inside the active subgroup-bin is bounded by `sigma^2(b, g)`, then conditional on the features,

```text
Var( R_hat(b, g) | X_1:n )
<=
sigma^2(b, g) / ESS(b, g).
```

If the residual variance is constant inside the subgroup-bin, the inequality is exact.

This is the cleanest statement in the angle. It shows that subgroup-bin ESS is not just a heuristic. It is the exact variance scale for the subgroup-bin residual estimator.

## Why global ESS is not enough

Global ESS only sees

```text
ESS_global = (sum_i w_i)^2 / sum_i w_i^2.
```

It does not control how the weighted information is distributed across subgroup-bins. A moderate global ESS can coexist with a nearly degenerate subgroup-bin ESS.

One simple construction is:

- a subgroup carries target mass `pi`,
- almost all of that subgroup mass is concentrated on one source point,
- the rest of the sample is well behaved outside the subgroup.

Then:

- `ESS(b, g)` can be as small as `1`,
- while `ESS_global` can still be of order `1 / pi^2`.

So for a subgroup with target mass `5%`, the global ESS can still be around `400` while the relevant subgroup estimate is effectively driven by one weighted point.

## Empirical result from the repo

Script:

- `preprint/research/local_ess_study.py`

Generated outputs:

- `preprint/research/results/local_ess_oracle/local_ess_summary.csv`
- `preprint/research/results/local_ess_oracle/local_ess_diagnostics.json`
- `preprint/research/results/local_ess_logistic/local_ess_summary.csv`

The `preprint/research/results/` directory is a regenerated local-output
directory and is not tracked by git.

The experiment uses `make_covariate_shift_classification`, where the model is calibrated by construction under pure covariate shift. That means the true subgroup-bin residuals are zero. Any estimated worst-group reliability signal is therefore a false alarm caused by finite-sample weighted noise.

Main findings from the saved run with `200` seeds:

- As shift severity increases from `0.4` to `1.6`, median global ESS falls from about `312.0` to `10.9`.
- Over the same range, the lower tail of subgroup-bin-local ESS collapses from about `2.76` to `1.00`.
- Median estimated worst-group error rises from about `0.040` to `0.138` even though the true worst-group error is zero.
- The false-alarm rate for estimated worst-group error above `0.05` rises from `19.0%` to `89.0%`.

Most importantly:

- pooled correlation between `log(1 + local ESS)` and subgroup-bin gap magnitude is about `-0.492`,
- pooled correlation between `log(1 + global ESS)` and subgroup-bin gap magnitude is about `-0.303`.

Within each fixed shift level, the lower-tail local ESS is also a better predictor of worst-group noise than global ESS.

The same qualitative picture also appears with learned logistic density-ratio weights:

- at shift `0.8`, median global ESS is about `103.0`, median lower-tail local ESS is about `1.38`, and the false-alarm rate above `0.05` is `82%`;
- at shift `1.2`, median global ESS is about `50.2`, median lower-tail local ESS is `1.00`, and the false-alarm rate above `0.05` is `92%`;
- within each fixed shift level, worst-group noise correlates more strongly with lower-tail local ESS than with global ESS.

## Why this could raise the paper's novelty

This angle shifts the paper from:

- weighted transport under covariate shift,
- plus worst-group auditing,

to:

- **the statistical geometry of worst-group auditing under shift**.

That is stronger because it says the hard problem is not only estimating the target metric. It is understanding when the audit itself becomes unstable and can report subgroup failures that are not real deployment failures.

In that form, the paper is not just about a new metric. It becomes a theory paper about:

- local overlap,
- subgroup-bin information,
- and false worst-group alarms under shifted deployment.

## Literature positioning note

The surrounding literature already studies:

- importance weighting under covariate shift,
- calibration under shift,
- multicalibration and subgroup calibration,
- subgroup weighting in causal inference.

The part that currently looks underdeveloped is the use of **subgroup-bin-local ESS as the governing quantity for worst-group reliability estimation and worst-group false alarms under covariate shift**.

That should still be positioned carefully as a likely novelty claim rather than an absolute claim until a tighter literature sweep is done around local overlap diagnostics and subgroup-weighted inference.
