# Testing And CI

ShiftStat uses a tiered test strategy so ordinary pull requests stay fast while
scientific workflows remain covered.

## Test Categories

- unmarked tests are expected to be fast unit or small integration tests
- `integration` tests exercise subprocesses, file-system artifacts, or
  multi-module workflows
- `benchmark` tests exercise benchmark runners and publication artifacts
- `examples` tests execute runnable example scripts
- `slow` tests are excluded from the cross-platform fast matrix

Run the fast local suite with:

```bash
python -m pytest -m "not slow"
```

Run the full suite with coverage:

```bash
python -m pytest --cov=shiftstat --cov-report=term-missing
```

Run only benchmark and example checks with:

```bash
python -m pytest -m "benchmark or examples"
```

## CI Gates

The CI workflow checks:

- fast tests across Linux, Windows, and macOS on Python 3.10, 3.11, and 3.12
- full coverage on Linux
- ruff linting
- mypy type checking
- strict documentation build
- wheel and sdist build plus wheel install smoke test

Coverage is configured in `pyproject.toml` and should stay high enough to cover
numerical edge cases, public APIs, and artifact-producing workflows.
