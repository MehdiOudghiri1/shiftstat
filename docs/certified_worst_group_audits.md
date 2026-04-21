# Certified Worst-Group Audits

ShiftStat supports inferential worst-group reliability audits under covariate
shift. The goal is not only to rank subgroup-bin gaps, but to decide whether a
reported alarm is statistically supported.

## Core idea

For a subgroup-bin cell `c`, ShiftStat computes:

- the weighted residual gap,
- the local effective sample size inside the cell,
- a simultaneous label-noise radius,
- an optional learned-weight sensitivity radius,
- a certified excess above a practical tolerance.

The resulting decision is one of:

- `certified_failure`: the cell clears the certified radius;
- `insufficient_evidence`: the naive gap is large, but not certified;
- `no_detected_failure`: the gap is below tolerance;
- `out_of_scope`: support or target-mass requirements fail.

## Minimal example

```python
import numpy as np
from shiftstat.certification import CertifiedWorstGroupAuditor

y_source = np.array([1, 0, 1, 0, 1, 0])
scores_source = np.array([0.2, 0.2, 0.6, 0.6, 0.8, 0.8])
groups = {
    "group_a": np.array([True, True, False, False, False, False]),
    "group_b": np.array([False, False, True, True, True, True]),
}

report = CertifiedWorstGroupAuditor(
    n_bins=3,
    tolerance=0.1,
    alpha=0.1,
).fit(
    y_source=y_source,
    scores_source=scores_source,
    groups=groups,
).report()

print(report.to_markdown())
print(report.insufficient_evidence())
```

## Learned weights

Use cross-fitted density-ratio weights when source labels will be reused for
certification:

```python
from shiftstat.reweight import CrossFittedImportanceWeighter

weighter = CrossFittedImportanceWeighter(method="logistic", n_folds=5)
weights = weighter.fit_predict(X_source, X_target)
```

If learned-weight uncertainty matters, pass plausible alternative weights into
the certified auditor. ShiftStat computes a cellwise sensitivity envelope
`rho_sensitivity`:

```python
report = CertifiedWorstGroupAuditor(tolerance=0.12).fit(
    y_source=y_source,
    scores_source=scores_source,
    weights=weights,
    groups=groups,
    alternative_weights=[weights_alt_1, weights_alt_2],
).report()
```

If `rho_sensitivity` dominates the apparent gap, the correct decision is
usually `insufficient_evidence`, not a target-certified alarm.

## Interpretation

The certified audit is intentionally conservative. A large point estimate with
low local ESS is evidence that the current audit is underpowered for that cell,
not evidence that the model is safe. The next operational step is targeted label
collection, a coarser subgroup-bin family, or a stronger density-ratio model.
