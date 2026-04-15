# Quickstart

```python
from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import ImportanceWeighter

bundle = make_covariate_shift_classification(random_state=3)

detector = ShiftDetector(random_state=3)
detector.fit(bundle.X_ref, bundle.X_target)
print(detector.summary().head())

weighter = ImportanceWeighter(random_state=3)
weighter.fit(bundle.X_ref, bundle.X_target)
print(weighter.summary())
```

