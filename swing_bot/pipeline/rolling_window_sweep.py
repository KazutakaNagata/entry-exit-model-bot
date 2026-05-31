"""Utilities for leak-safe rolling-window threshold sweeps.

These helpers are intentionally tiny and dependency-free so sweep scripts can
share the same defaults.  They do not read market data or predictions.
"""
from __future__ import annotations


def parse_csv_ints(text: str | None, *, default: list[int]) -> list[int]:
    """Parse a comma-separated integer list while preserving order."""
    if text is None or str(text).strip() == "":
        return list(default)
    out: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("empty integer list")
    return out


def parse_csv_floats(text: str | None, *, default: list[float]) -> list[float]:
    """Parse a comma-separated float list while preserving order."""
    if text is None or str(text).strip() == "":
        return list(default)
    out: list[float] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("empty float list")
    return out


def auto_rolling_min_periods(
    window_days: int,
    *,
    bars_per_day: int = 1440,
    min_floor: int = 5000,
    cap: int = 30000,
) -> int:
    """Return a conservative min-periods value for high quantile thresholds.

    The rolling gate usually uses q99/q99.5.  If ``min_periods`` is too small,
    the threshold is estimated from only a handful of tail observations and can
    become unstable.  This rule uses roughly half the requested window, with a
    floor and cap so short windows have enough history and long windows do not
    skip most of a 3-month validation fold.
    """
    window_days = int(window_days)
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    bars_per_day = int(bars_per_day)
    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")
    samples = window_days * bars_per_day
    value = max(int(samples // 2), int(min_floor))
    value = min(value, int(cap))
    # Do not require more rows than the window can possibly contain.
    return int(min(value, samples))


def resolve_min_periods(value: str | int, window_days: int) -> int:
    """Resolve ``auto`` or an explicit integer min-periods value."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("rolling min periods must be positive")
        return int(value)
    text = str(value).strip().lower()
    if text == "auto":
        return auto_rolling_min_periods(window_days)
    resolved = int(text)
    if resolved <= 0:
        raise ValueError("rolling min periods must be positive")
    return resolved


def threshold_slug(value: float | int | str) -> str:
    """Return a filesystem-safe numeric slug."""
    try:
        text = f"{float(value):g}"
    except (TypeError, ValueError):
        text = str(value)
    return text.replace("-", "m").replace(".", "p")
