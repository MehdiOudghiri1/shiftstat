# Release Process

ShiftStat releases should be reproducible, installable, and traceable to a clean
commit.

## Release Gates

Before tagging:

- the working tree must contain only intended source, test, doc, and artifact
  changes
- `CHANGELOG.md` must describe user-visible changes
- `CITATION.cff` must match the release version for citable releases
- tests, linting, typing, docs, and distribution checks must pass
- generated manifests must avoid machine-specific absolute paths

## Build And Smoke Test

```bash
python -m pip install -e .[dev,docs,examples,release]
python -m pytest --cov=shiftstat --cov-report=term-missing
python -m ruff check .
python -m mypy src
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Install the built wheel into a fresh virtual environment and verify:

```bash
python -c "import shiftstat; print(shiftstat.__version__)"
shiftstat-experiment --help
```

## Publishing

The release workflow supports:

- build-only release artifacts
- TestPyPI publishing through manual dispatch
- PyPI publishing through trusted publishing from release tags

PyPI publishing should be enabled only after trusted publishers are configured
for the repository environments named `testpypi` and `pypi`.
