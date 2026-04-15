# ShiftStat

ShiftStat is an open-source scientific Python library for detecting, characterizing, and correcting reliability loss under tabular distribution shift.

## Why ShiftStat

Machine learning systems often fail quietly when deployment data drifts away from the training distribution. ShiftStat provides a research-oriented toolkit to quantify that mismatch, rank which features have moved, and estimate importance weights for covariate-shift-aware evaluation.

## Key features

- Feature-wise shift testing for continuous and categorical tabular variables
- Dataset-level drift summaries with multiple-testing correction
- Classifier two-sample testing for source-target separability
- Importance weighting via domain classifiers and logistic baselines
- Weighted evaluation metrics for classification and regression
- Publication-friendly matplotlib plots and markdown-friendly reports

## Installation

```bash
pip install shiftstat
```

For development:

```bash
pip install -e .[dev,docs,examples]
```

## Minimal example

```python
from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reweight import ImportanceWeighter, weighted_risk

data = make_covariate_shift_classification(random_state=7)

detector = ShiftDetector(random_state=7)
detector.fit(data.X_ref, data.X_target)
print(detector.summary().head())

weighter = ImportanceWeighter(method="domain_classifier", random_state=7)
weighter.fit(data.X_ref, data.X_target)
weights = weighter.predict_weights()

unweighted = weighted_risk(
    data.y_ref,
    data.reference_predictions,
    loss="log_loss",
)
weighted = weighted_risk(
    data.y_ref,
    data.reference_predictions,
    sample_weight=weights,
    loss="log_loss",
)
print({"unweighted": unweighted, "weighted": weighted})
```

## Scientific motivation

ShiftStat focuses on a foundational scientific question: how stable are learning systems when the deployment distribution differs from the training distribution? V1 emphasizes transparent statistical summaries, reproducible weighting pipelines, and APIs that align with scientific Python practice.

## Documentation

Documentation lives in [docs/](docs/index.md) and is designed for both research users and contributors.

## Roadmap

- Extend shift tests and effect-size estimators
- Add calibration and decision-aware evaluation under shift
- Support richer benchmark suites and experiment tracking
- Expand reporting and estimator interoperability

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidance.

## Status

V1 establishes the core tabular-first package architecture, initial scientific modules, examples, and benchmark scaffolding. Advanced robustness tooling is intentionally deferred until the statistical foundations are stable.

