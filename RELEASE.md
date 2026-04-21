# Release Checklist

This checklist is the release gate for ShiftStat. A release should not be cut
from a dirty or partially generated working tree.

## 1. Prepare

- Confirm the public API changes are documented in `CHANGELOG.md`.
- Confirm `src/shiftstat/_version.py` matches the intended release tag.
- Confirm `CITATION.cff` has the same version and release date when preparing a
  citable software release.
- Decide whether generated paper assets are intentionally part of the release.
  Keep reproducible configs under version control; avoid committing local logs,
  LaTeX scratch files, or machine-specific paths.

## 2. Verify Locally

```bash
python -m pip install -e .[dev,docs,examples,release]
python -m ruff check .
python -m mypy src
python -m pytest --cov=shiftstat --cov-report=term-missing
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Then test the built wheel in a fresh environment:

```bash
python -m venv .wheel-smoke
.wheel-smoke/Scripts/python -m pip install --upgrade pip
.wheel-smoke/Scripts/python -m pip install dist/*.whl
.wheel-smoke/Scripts/python -c "import shiftstat; print(shiftstat.__version__)"
.wheel-smoke/Scripts/shiftstat-experiment --help
```

Use the POSIX-style `.wheel-smoke/bin/...` paths on Linux or macOS.

## 3. Publish

- Push a tag named `vX.Y.Z`.
- Confirm the release workflow builds, checks, and smoke-installs the wheel.
- Publish to TestPyPI first for release candidates or packaging changes.
- Publish to PyPI only from a reviewed tag and with trusted publishing enabled.

## 4. After Release

- Create or update the GitHub release notes.
- Confirm `pip install shiftstat` installs the expected version.
- Confirm the hosted docs point to the released API.
- Archive any paper artifacts that support scientific claims.
