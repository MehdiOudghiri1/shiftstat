from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    """Keep plotting tests from leaking figure handles across the suite."""

    yield
    plt.close("all")
