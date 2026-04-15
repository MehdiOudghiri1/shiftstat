# Contributing to ShiftStat

Thank you for contributing to ShiftStat. The project aims to be a dependable scientific codebase, so contributions should optimize for clarity, reproducibility, and maintainability.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev,docs,examples]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev,docs,examples]
```

## Local checks

```bash
pytest
ruff check .
mypy src
mkdocs build
```

## Contribution principles

- Prefer explicit scientific terminology over abbreviations
- Keep public APIs small, typed, and well documented
- Add tests for numerical and edge-case behavior
- Preserve reproducibility when randomness is involved
- Separate core statistical logic from plotting and reporting layers

## Pull requests

- Open focused pull requests with a clear scientific or engineering motivation
- Document new public APIs in both docstrings and user-facing docs
- Add or update examples when behavior changes materially
- Avoid introducing optional dependencies into the base install without strong justification

## Code style

ShiftStat uses NumPy-style docstrings, `ruff` for linting, and `mypy` for type checking. Please keep implementations modular and approachable for future research contributors.

