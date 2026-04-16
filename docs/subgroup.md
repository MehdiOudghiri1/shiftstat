# Subgroup Reliability Analysis

`shiftstat.subgroup` provides a disciplined way to ask which slices degrade first under deployment shift.

Core API:

- `SubgroupAnalyzer`
- `SubgroupReport`
- `group_by_feature(...)`
- `group_metrics(...)`

What V3 computes:

- subgroup performance tables
- subgroup calibration tables
- subgroup exposure shifts from reference to deployment
- degradation rankings by severity
- support and coverage diagnostics

The default behavior supports:

- categorical features
- discretized numerical features
- intersectional subgrouping through user-provided tuples

Example:

```python
from shiftstat.datasets import make_hidden_subgroup_shift_classification
from shiftstat.subgroup import SubgroupAnalyzer

data = make_hidden_subgroup_shift_classification(random_state=10)
analyzer = SubgroupAnalyzer(min_group_size=25).fit(
    data.X_ref,
    data.y_ref,
    data.reference_predictions,
    data.X_target,
    data.y_target,
    data.target_predictions,
    subgroup_features=["region", "channel", "load"],
    intersectional_features=[("region", "channel")],
)

print(analyzer.degradation_ranking().head())
print(analyzer.to_report().to_markdown())
```

Interpret subgroup results with the support diagnostics, especially for calibration-oriented metrics.
