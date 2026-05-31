"""Small run-id helpers for reproducible research artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import re

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


def safe_slug(value: str) -> str:
    """Return a filesystem-safe slug while keeping it human-readable."""
    text = str(value).strip().replace(" ", "_")
    text = _SAFE_CHARS.sub("-", text)
    return text.strip("-_") or "run"


def make_run_id(prefix: str, *, side: str | None = None, horizon_minutes: int | None = None) -> str:
    """Create a deterministic-format UTC run id.

    The id is not meant to be cryptographically unique; it is just readable and
    sortable.  Example: ``entry_long_H60_20260529T120102Z``.
    """
    pieces = [safe_slug(prefix)]
    if side:
        pieces.append(safe_slug(side))
    if horizon_minutes is not None:
        pieces.append(f"H{int(horizon_minutes)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pieces.append(stamp)
    return "_".join(pieces)
