# ShiftStat

ShiftStat is an open-source scientific Python library for detecting distribution shift and quantifying predictive reliability degradation in tabular machine learning systems.

## Why ShiftStat

Classical performance metrics can stay deceptively stable while calibration, confidence behavior, and empirical reliability degrade under deployment shift. ShiftStat helps researchers and engineers study that degradation with transparent statistical diagnostics, importance weighting, recalibration tools, and structured evaluation workflows.

## Key features

- Mixed-type tabular shift detection with feature-wise tests and dataset summaries
- Importance weighting for covariate-shift-aware evaluation
- Calibration diagnostics including ECE, MCE, Brier decomposition helpers, and bin-wise summaries
- Post-hoc recalibration with temperature scaling, isotonic regression, and logistic calibration
- Reliability profiles comparing reference and target domains
- End-to-end workflow evaluation with optional weighting and recalibration
- Publication-friendly plots and markdown-friendly reports
- Synthetic datasets, examples, and reliability-focused benchmark scaffolding

## Installation

```bash
pip install shiftstat
```

For development:

```bash
pip install -e .[dev,docs,examples]
```

## Minimal V2 example

```python
from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.reliability import evaluate_under_shift

data = make_covariate_shift_classification(random_state=7, shift_strength=1.0)

result = evaluate_under_shift(
    LogisticRegression(max_iter=2000),
    data.X_ref,
    data.y_ref,
    data.X_target,
    data.y_target,
    apply_importance_weighting=True,
    recalibration="temperature",
    random_state=7,
)

print(result.summary_frame())
print(result.to_report().to_markdown())
```

## Scientific motivation

ShiftStat focuses on a central statistical learning question: when the deployment covariate distribution differs from the reference distribution, how do performance, calibration, and confidence-conditioned reliability change? V2 extends the library from shift characterization into model-facing reliability analysis under covariate shift, with explicit support for weighted diagnostics and post-hoc recalibration.

## Documentation

Documentation lives in [docs/](docs/index.md) and includes theory notes, workflow guides, API reference pages, and runnable examples.

## Examples

The repository includes examples for:

- feature drift detection
- importance weighting under covariate shift
- calibration degradation under shift
- weighted versus unweighted calibration evaluation
- recalibration on a semi-real tabular dataset
- end-to-end reliability reporting

## Roadmap

- Extend multiclass and regression reliability diagnostics
- Add calibration-aware benchmark suites and richer experiment aggregation
- Expand robustness reporting to subgroup and decision-aware analyses
- Support broader estimator interoperability and richer reporting export

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidance.

## Status

V2 adds calibration under shift, reliability diagnostics, model evaluation workflows, new plots, new reports, and the first reliability-focused benchmarks. Advanced robustness topics such as selective prediction, subgroup reporting, and online monitoring remain intentionally deferred.
