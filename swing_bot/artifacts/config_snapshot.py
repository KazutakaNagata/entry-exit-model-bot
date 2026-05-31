"""Helpers for preserving the config used by a run."""
from __future__ import annotations

import hashlib
from pathlib import Path

from swing_bot.artifacts.io import write_json


def file_sha256(path: Path | str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_paths(paths: dict[str, Path | str | None]) -> dict[str, dict[str, str | None]]:
    """Return path + sha256 metadata for existing config/artifact paths."""
    out: dict[str, dict[str, str | None]] = {}
    for name, raw_path in paths.items():
        if raw_path is None:
            out[name] = {"path": None, "sha256": None}
            continue
        p = Path(raw_path)
        out[name] = {
            "path": str(p),
            "sha256": file_sha256(p) if p.exists() and p.is_file() else None,
        }
    return out


def write_snapshot(paths: dict[str, Path | str | None], output_path: Path | str) -> Path:
    return write_json(snapshot_paths(paths), output_path)
