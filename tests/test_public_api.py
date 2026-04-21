from __future__ import annotations

import pytest

import shiftstat
from shiftstat.exceptions import ValidationError


def test_top_level_public_api_resolves_all_exports() -> None:
    resolved = {name: getattr(shiftstat, name) for name in shiftstat.__all__}

    assert resolved["__version__"] == shiftstat.__version__
    assert resolved["ShiftDetector"].__name__ == "ShiftDetector"
    assert resolved["CertifiedWorstGroupAuditor"].__name__ == "CertifiedWorstGroupAuditor"
    assert callable(resolved["weighted_mean"])
    assert callable(resolved["run_experiment"])


def test_top_level_public_api_rejects_unknown_names() -> None:
    unknown_name = "not_a_public_name"
    with pytest.raises(AttributeError):
        getattr(shiftstat, unknown_name)


def test_package_exceptions_are_importable() -> None:
    error = ValidationError("bad input")

    assert isinstance(error, Exception)
