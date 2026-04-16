# Conditional Auditing Under Shift

`shiftstat.audit` extends reliability analysis into conditional and slice-level auditing.

Main API:

- `ReliabilityAuditor`
- `ConditionalReliabilityAuditor`
- `AuditReport`

The audit layer combines:

- aggregate reference-versus-target reliability
- subgroup degradation summaries
- confidence-conditioned error tables
- performance by shift-severity slices
- disparity summaries
- hidden-failure flags
- discovered failure slices

Typical usage:

```python
from shiftstat.audit import ReliabilityAuditor
from shiftstat.datasets import make_hidden_subgroup_shift_classification

data = make_hidden_subgroup_shift_classification(random_state=11)
auditor = ReliabilityAuditor(min_group_size=25, random_state=11).fit(
    data.X_ref,
    data.y_ref,
    data.reference_predictions,
    data.X_target,
    data.y_target,
    data.target_predictions,
    subgroup_features=["region", "channel", "load"],
    intersectional_features=[("region", "channel")],
)

report = auditor.to_report()
print(report.aggregate_summary)
print(report.hidden_failure_flags)
```

The hidden-failure flags are heuristic prioritization signals rather than formal statistical guarantees.
