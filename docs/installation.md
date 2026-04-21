# Installation

## Stable installation

```bash
pip install shiftstat
```

If a release has not yet been published for your platform or environment,
install from a checked-out source tree instead:

```bash
pip install .
```

## Development installation

```bash
pip install -e .[dev,docs,examples]
```

Release tooling can be installed with:

```bash
pip install -e .[dev,docs,examples,release]
```

## Core dependencies

ShiftStat depends on NumPy, pandas, SciPy, scikit-learn, and matplotlib.
