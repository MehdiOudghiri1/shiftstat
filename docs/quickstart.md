# Quickstart

```python
from shiftstat.datasets import make_covariate_shift_classification
from sklearn.linear_model import LogisticRegression
from shiftstat.selective import evaluate_selective_under_shift

bundle = make_covariate_shift_classification(random_state=3)

result = evaluate_selective_under_shift(
    LogisticRegression(max_iter=2000),
    bundle.X_ref,
    bundle.y_ref,
    bundle.X_target,
    bundle.y_target,
    apply_importance_weighting=True,
    use_weighted_threshold_tuning=True,
    target_coverage=0.8,
    random_state=3,
)
print(result.summary_frame())
print(result.to_report().to_markdown())
```
