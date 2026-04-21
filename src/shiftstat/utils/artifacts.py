"""Helpers for portable artifact manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    """Return a path string that is stable across machines when possible."""

    path_obj = Path(path)
    if relative_to is None:
        return path_obj.as_posix()

    try:
        relative = path_obj.resolve().relative_to(Path(relative_to).resolve())
    except ValueError:
        return path_obj.as_posix()
    return relative.as_posix()


def file_digest(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Compute a hexadecimal digest for a file."""

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    """Return portable path, size, and digest metadata for one artifact."""

    path_obj = Path(path)
    return {
        "path": portable_path(path_obj, relative_to=relative_to),
        "bytes": path_obj.stat().st_size,
        "sha256": file_digest(path_obj),
    }
