# Quickstart

```python
from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reliability import evaluate_under_shift
from sklearn.linear_model import LogisticRegression

bundle = make_covariate_shift_classification(random_state=3)

result = evaluate_under_shift(
    LogisticRegression(max_iter=2000),
    bundle.X_ref,
    bundle.y_ref,
    bundle.X_target,
    bundle.y_target,
    apply_importance_weighting=True,
    recalibration="temperature",
    random_state=3,
)
print(result.summary_frame())
```
