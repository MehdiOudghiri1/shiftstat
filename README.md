# ShiftStat

[![PyPI](https://img.shields.io/pypi/v/shiftstat.svg)](https://pypi.org/project/shiftstat/)
[![Python](https://img.shields.io/pypi/pyversions/shiftstat.svg)](https://pypi.org/project/shiftstat/)
[![License](https://img.shields.io/pypi/l/shiftstat.svg)](https://github.com/MehdiOudghiri1/shiftstat/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typing-py.typed-blue.svg)](https://peps.python.org/pep-0561/)

**Scientific Python tooling for reliability under tabular distribution shift.**

ShiftStat helps answer a practical research question:

> A model looked reliable on yesterday's data. What exactly breaks when tomorrow's
> deployment population is different?

Most model evaluation pipelines stop at aggregate accuracy, AUC, or loss. Those
numbers can remain calm while calibration collapses in a high-confidence bin,
an operational subgroup loses reliability, or abstention improves average risk
while abandoning the wrong population. ShiftStat is built to expose those
failures with transparent statistical objects: feature-level shift tests,
density-ratio weights, calibration profiles, subgroup audits, selective
prediction curves, certified worst-group alarms, and reproducible benchmark
artifacts.

The package is designed for researchers, applied scientists, and ML engineers
working with tabular classifiers under dataset shift, especially when the goal
is not only to report a score but to understand, reproduce, and defend the
reliability conclusion.

## Installation

```bash
pip install shiftstat
```

ShiftStat supports Python `>=3.10` and follows the standard `src/` package
layout. The public package name is:

```python
import shiftstat
print(shiftstat.__version__)
```

## The Problem In One Picture

In deployment, the reference distribution and the target distribution are rarely
the same:

```text
reference data              target deployment data
P_ref(X, Y)                 P_tgt(X, Y)
     |                            |
     v                            v
train / validate model       audit reliability under shift
```

The target risk of a predictor `f` is

```text
R_tgt(f) = E_{(X,Y) ~ P_tgt}[ ell(Y, f(X)) ].
```

If only reference labels are abundant, the naive reference estimate

```text
(1 / n) * sum_i ell(y_i, f(x_i))
```

can be biased for the target population. Under covariate shift, where
`P_ref(Y | X) = P_tgt(Y | X)` but `P_ref(X) != P_tgt(X)`, the target risk can be
estimated with density-ratio weights:

```text
w(x) = p_tgt(x) / p_ref(x)

R_tgt(f) = E_{P_ref}[ w(X) * ell(Y, f(X)) ].
```

ShiftStat turns this idea into an inspectable workflow rather than a black-box
dashboard.

## What ShiftStat Does

| Layer | Scientific question | Main tools |
| --- | --- | --- |
| Shift detection | Which features moved, and is the full covariate table distinguishable from the reference sample? | Kolmogorov-Smirnov tests, chi-square tests, PSI, Wasserstein distance, classifier two-sample tests |
| Importance weighting | Can reference observations be reweighted toward the target covariate distribution? | Domain-classifier density ratios, logistic weighting, clipping, normalization, effective sample size |
| Calibration and reliability | Are probabilities still meaningful under shift? | ECE, MCE, Brier score, log loss, calibration slope/intercept, reliability diagrams |
| Recalibration | Can post-hoc calibration repair probability quality? | Temperature scaling, Platt/logistic calibration, isotonic calibration |
| Subgroup auditing | Are failures hidden inside operational slices? | Group metrics, support diagnostics, worst-group summaries, hidden-failure flags |
| Selective prediction | Does abstention reduce accepted-set risk without hiding failure modes? | Confidence, entropy, margin, learned-risk policies, risk-coverage curves |
| Certification | Is a worst-group alarm statistically supported, or just noisy? | Local effective sample size, simultaneous radii, learned-weight sensitivity, certified excess |
| Experiments | Can the whole evaluation be reproduced later? | Benchmark scenarios, YAML/JSON experiment configs, CSV/Markdown/LaTeX/figure artifact export |

## Core Scientific Quantities

### 1. Density-ratio weighting

ShiftStat estimates target-over-reference weights by fitting a domain classifier
that predicts whether a row came from the target sample:

```text
d(x) = P(D = target | X = x)

w(x) = [d(x) / (1 - d(x))] * [pi_ref / pi_tgt].
```

The weighted empirical risk is then

```text
R_hat_w(f) =
    sum_i w_i * ell(y_i, f(x_i))
    --------------------------------
             sum_i w_i
```

and the Kish effective sample size diagnoses whether the weighted estimate is
stable:

```text
n_eff = (sum_i w_i)^2 / sum_i w_i^2.
```

Large weights with small `n_eff` are treated as evidence limitations, not as a
license to overclaim.

### 2. Calibration under shift

For a predicted probability `p_i` and binary label `y_i`, ShiftStat computes
bin-wise reliability gaps. With bins `B_1, ..., B_K`, Expected Calibration Error
is

```text
ECE = sum_{k=1}^K (|B_k| / n) *
      | mean_{i in B_k}(p_i) - mean_{i in B_k}(y_i) |.
```

The weighted version replaces counts and means with weighted mass:

```text
ECE_w = sum_{k=1}^K (W_k / W) *
        | avg_w(p_i | i in B_k) - avg_w(y_i | i in B_k) |.
```

ShiftStat also reports Brier score, log loss, maximum calibration error, and
logistic calibration slope/intercept so a single scalar cannot hide the shape of
the failure.

### 3. Selective prediction and risk-coverage curves

Selective prediction introduces an acceptance function `phi(x) in {0, 1}`. The
accepted-set target risk is

```text
R_tgt(f, phi) =
    E_tgt[ phi(X) * ell(Y, f(X)) ]
    --------------------------------
          E_tgt[ phi(X) ]

coverage(phi) = E_tgt[ phi(X) ].
```

ShiftStat sweeps score thresholds to build the risk-coverage curve: lower risk
is only useful if the retained coverage and subgroup composition still make
operational sense.

### 4. Certified worst-group reliability

For a subgroup-bin cell `c`, ShiftStat audits the weighted residual gap

```text
g_c = avg_w( y_i - p_i | i in c ).
```

It then computes a local effective sample size and a simultaneous finite-family
radius over `K` tested cells:

```text
r_c = sqrt( log(2K / alpha) / (2 * n_eff,c) ).
```

When learned weights are uncertain, a sensitivity envelope `rho_c` can be added.
With practical tolerance `tau` and optional population radius `gamma`, the
certified excess is

```text
excess_c = max( |g_c| - r_c - rho_c - gamma - tau, 0 ).
```

The result is not just a sorted list of scary cells. Each cell receives an
explicit decision:

- `certified_failure`
- `insufficient_evidence`
- `no_detected_failure`
- `out_of_scope`

This distinction is central to the library: an alarming point estimate with low
local support should trigger better data collection, not an unsupported claim.

## Quickstart

This example creates a synthetic covariate-shift problem, trains a classifier,
uses importance weighting, tunes an abstention threshold, and prints a compact
deployment report.

```python
from sklearn.linear_model import LogisticRegression

from shiftstat.datasets import make_covariate_shift_classification
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

The output is designed to be inspectable: coverage, abstention rate, accepted-set
risk, calibration, risk reduction, weighting diagnostics, and target-domain
comparison tables remain available as ordinary pandas objects.

## A Full Reliability Workflow

```python
from sklearn.ensemble import RandomForestClassifier

from shiftstat.datasets import make_covariate_shift_classification
from shiftstat.detect import ShiftDetector
from shiftstat.reliability import evaluate_under_shift

bundle = make_covariate_shift_classification(random_state=11)

detector = ShiftDetector(random_state=11).fit(bundle.X_ref, bundle.X_target)
print(detector.summary().head())

report = evaluate_under_shift(
    RandomForestClassifier(random_state=11),
    bundle.X_ref,
    bundle.y_ref,
    bundle.X_target,
    bundle.y_target,
    apply_importance_weighting=True,
    recalibration="temperature",
    random_state=11,
)

print(report.to_report().to_markdown())
```

Conceptually, the workflow is:

```text
tabular reference and target data
        |
        v
detect feature and dataset shift
        |
        v
estimate density-ratio weights
        |
        v
compare reference, weighted-reference, and target reliability
        |
        v
audit subgroups, abstention behavior, and certified worst-group cells
        |
        v
export tables, figures, summaries, and manifests
```

## Certified Worst-Group Audit Example

```python
from shiftstat.certification import CertifiedWorstGroupAuditor
from shiftstat.reweight import CrossFittedImportanceWeighter

weights = CrossFittedImportanceWeighter(
    method="logistic",
    n_folds=5,
    random_state=7,
).fit_predict(X_source, X_target)

audit = CertifiedWorstGroupAuditor(
    n_bins=6,
    tolerance=0.12,
    alpha=0.10,
    min_local_ess=5.0,
).fit(
    y_source=y_source,
    scores_source=scores_source,
    scores_target=scores_target,
    weights=weights,
    groups=source_groups,
    target_groups=target_groups,
)

report = audit.report()

print(report.certified_failures())
print(report.insufficient_evidence())
print(report.to_markdown())
```

Use this when the scientific question is not merely "which group looks worst?"
but "which worst-group findings survive support-aware uncertainty accounting?"

## Reproducible Benchmarks And Paper-Ready Artifacts

ShiftStat includes repeated-seed benchmark scenarios and a config-driven
experiment runner. A benchmark can be launched from Python:

```python
from shiftstat.bench import BenchmarkRunner, make_covariate_shift_sweep_scenario

scenario = make_covariate_shift_sweep_scenario(
    severities=[0.2, 0.8, 1.4],
    seeds=[7, 19, 43],
    baseline_names=[
        "raw_model",
        "weighting_only",
        "recalibration_only",
        "confidence_abstention",
    ],
)

result = BenchmarkRunner().run(scenario)
artifacts = result.export_artifacts("artifacts/covariate_shift_demo")

print(result.aggregate_frame())
print(artifacts["figures"])
```

or from the command line:

```bash
shiftstat-experiment paper_assets/configs/publication_suite.yaml
```

Generated artifacts are ordinary files:

- per-scenario run CSVs
- aggregate summary CSVs
- Markdown reports
- LaTeX tables
- figures
- copied configs
- manifests
- logs

That design is deliberate. Scientific results should be reproducible without
requiring a private dashboard or hidden state.

## Package Map

| Import path | Purpose |
| --- | --- |
| `shiftstat.detect` | Feature-wise and dataset-level shift detection |
| `shiftstat.reweight` | Importance weighting and effective sample size diagnostics |
| `shiftstat.calibration` | Calibration metrics and post-hoc recalibration |
| `shiftstat.reliability` | End-to-end reliability evaluation under shift |
| `shiftstat.subgroup` | Grouped reliability and support-aware subgroup summaries |
| `shiftstat.audit` | Conditional audits and failure-slice discovery |
| `shiftstat.selective` | Abstention policies and risk-coverage analysis |
| `shiftstat.certification` | Certified worst-group reliability audits |
| `shiftstat.bench` | Repeated-seed benchmark scenarios and baselines |
| `shiftstat.experiments` | Config-driven experiment runner and CLI |
| `shiftstat.plotting` | Publication-friendly plotting helpers |
| `shiftstat.reports` | Markdown, table, and dictionary report objects |

## Documentation

- [Installation](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/installation.md)
- [Quickstart](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/quickstart.md)
- [Theory notes](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/theory.md)
- [Reliability theory](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/reliability_theory.md)
- [Subgroup auditing](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/subgroup.md)
- [Selective prediction](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/selective_prediction.md)
- [Certified worst-group audits](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/certified_worst_group_audits.md)
- [Benchmarking](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/benchmarking.md)
- [Publication workflow](https://github.com/MehdiOudghiri1/shiftstat/blob/main/docs/publication_workflow.md)

## Scientific Positioning

ShiftStat is not a monitoring SaaS product and not a single metric. It is a
research-oriented Python library for building auditable evidence about tabular
model behavior under shift.

It is intentionally modular:

- detection does not hide the test statistics;
- weighting exposes clipping, AUC, and effective sample size;
- calibration reports both scalar and bin-level summaries;
- subgroup audits surface support and coverage caveats;
- selective prediction reports the cost of rejection, not only the gain in risk;
- certification separates `certified_failure` from `insufficient_evidence`.

The library is strongest when used as part of a transparent evaluation protocol:
state the target population, inspect overlap, compare weighted and unweighted
conclusions, audit important subgroups, and report uncertainty whenever support
is thin.

## Development Quality

The project includes:

- typed package code with `py.typed`;
- a public API organized around estimator-style objects and result containers;
- regression tests for detection, weighting, calibration, reliability, subgroup
  analysis, selective prediction, certification, benchmarks, examples, and CLI
  behavior;
- Ruff linting and mypy type checking;
- GitHub Actions for Linux, macOS, Windows, and Python 3.10 through 3.12;
- wheel build and smoke-install checks;
- documentation and reproducible experiment configuration.

For local development:

```bash
pip install -e .[dev,docs]
pytest -m "not slow"
ruff check .
mypy src
mkdocs build
```

The optional examples extra includes notebook tooling and external datasets:

```bash
pip install -e .[examples]
```

## Limitations

ShiftStat is designed for scientific diagnostics, not automatic guarantees.

- Most reliability and certification workflows currently focus on binary
  classification probabilities.
- Importance weighting depends on overlap between reference and target
  covariate distributions.
- Target labels, when available, should be used directly for empirical target
  reliability checks.
- A low local effective sample size is a warning about audit power, not evidence
  that a model is safe.
- Multiclass certification and sharper adaptive-search theory are intentionally
  future work.

## Citation And License

ShiftStat is released under the BSD 3-Clause License.

If you use ShiftStat in scientific work, please cite the software metadata in
[`CITATION.cff`](https://github.com/MehdiOudghiri1/shiftstat/blob/main/CITATION.cff).
Release history is maintained in
[`CHANGELOG.md`](https://github.com/MehdiOudghiri1/shiftstat/blob/main/CHANGELOG.md).
