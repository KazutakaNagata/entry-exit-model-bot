"""Train/valid/test split manifest utilities.

The split manifest is intentionally explicit.  Research scripts should tune on
train/valid only; the test range is locked-audit only and should be read only by
the final locked evaluation script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


class SplitManifestError(ValueError):
    """Raised when a split manifest is missing or internally inconsistent."""


PLACEHOLDER_PREFIXES = ("YYYY-", "YYYY/", "<", "TODO", "TBD")


@dataclass(frozen=True)
class TimeRange:
    """Inclusive UTC timestamp range."""

    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, values: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.Index:
        parsed = pd.to_datetime(values, utc=True)
        return (parsed >= self.start) & (parsed <= self.end)

    @property
    def duration_minutes(self) -> float:
        return float((self.end - self.start) / pd.Timedelta(minutes=1))

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class FoldSpec:
    name: str
    range: TimeRange

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, **self.range.to_dict()}


@dataclass(frozen=True)
class SplitManifest:
    split_version: str
    symbol: str
    bar_size: str
    timezone: str
    train: TimeRange
    valid: TimeRange
    test: TimeRange
    folds: tuple[FoldSpec, ...]
    purge_minutes: int
    embargo_minutes: int
    test_usage: str = "locked_audit_only"

    def fold(self, name: str) -> FoldSpec:
        for fold in self.folds:
            if fold.name == name:
                return fold
        raise KeyError(f"unknown fold: {name}")

    def mask(self, timestamps: pd.Series | pd.DatetimeIndex, split: str) -> pd.Series | pd.Index:
        """Return a boolean mask for train, valid, test, or a fold name."""
        if split == "train":
            return self.train.contains(timestamps)
        if split == "valid":
            return self.valid.contains(timestamps)
        if split == "test":
            return self.test.contains(timestamps)
        return self.fold(split).range.contains(timestamps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "symbol": self.symbol,
            "bar_size": self.bar_size,
            "timezone": self.timezone,
            "train": self.train.to_dict(),
            "valid": {
                **self.valid.to_dict(),
                "folds": [fold.to_dict() for fold in self.folds],
            },
            "test": {**self.test.to_dict(), "usage": self.test_usage},
            "purge": {
                "purge_minutes": self.purge_minutes,
                "embargo_minutes": self.embargo_minutes,
            },
        }


def _is_placeholder(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or any(text.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def _parse_timestamp(value: object, field_name: str) -> pd.Timestamp:
    if _is_placeholder(value):
        raise SplitManifestError(
            f"split manifest contains placeholder timestamp for {field_name!r}: {value!r}. "
            "Copy the template to a concrete split file and fill real UTC dates."
        )
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise SplitManifestError(f"invalid timestamp for {field_name!r}: {value!r}")
    return pd.Timestamp(parsed)


def _parse_range(raw: dict[str, Any], field_name: str) -> TimeRange:
    if not isinstance(raw, dict):
        raise SplitManifestError(f"{field_name} must be a mapping with start/end")
    start = _parse_timestamp(raw.get("start"), f"{field_name}.start")
    end = _parse_timestamp(raw.get("end"), f"{field_name}.end")
    if start > end:
        raise SplitManifestError(f"{field_name}.start must be <= {field_name}.end")
    return TimeRange(start=start, end=end)


def _require_keys(raw: dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        raise SplitManifestError("split manifest missing required keys: " + ", ".join(missing))


def _validate_chronology(manifest: SplitManifest) -> None:
    if manifest.train.end >= manifest.valid.start:
        raise SplitManifestError("train.end must be before valid.start")
    if manifest.valid.end >= manifest.test.start:
        raise SplitManifestError("valid.end must be before test.start")
    if len(manifest.folds) != 5:
        raise SplitManifestError(f"valid.folds must contain exactly 5 folds, got {len(manifest.folds)}")

    previous_end: pd.Timestamp | None = None
    for fold in manifest.folds:
        if fold.range.start < manifest.valid.start or fold.range.end > manifest.valid.end:
            raise SplitManifestError(f"{fold.name} is outside the valid range")
        if previous_end is not None and fold.range.start <= previous_end:
            raise SplitManifestError("valid folds must be non-overlapping and sorted")
        previous_end = fold.range.end

    if manifest.test_usage != "locked_audit_only":
        raise SplitManifestError("test.usage must be 'locked_audit_only'")



def load_split_manifest(path: Path | str) -> SplitManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SplitManifestError("split manifest must be a YAML mapping")

    _require_keys(raw, ["split_version", "symbol", "bar_size", "timezone", "train", "valid", "test"])

    valid_raw = raw["valid"]
    if not isinstance(valid_raw, dict) or "folds" not in valid_raw:
        raise SplitManifestError("valid must contain a folds list")

    folds = []
    for idx, fold_raw in enumerate(valid_raw.get("folds") or [], start=1):
        if not isinstance(fold_raw, dict):
            raise SplitManifestError(f"valid.folds[{idx}] must be a mapping")
        name = str(fold_raw.get("name") or f"fold_{idx:02d}")
        folds.append(FoldSpec(name=name, range=_parse_range(fold_raw, f"valid.folds.{name}")))

    purge_raw = raw.get("purge", {}) or {}
    manifest = SplitManifest(
        split_version=str(raw["split_version"]),
        symbol=str(raw["symbol"]),
        bar_size=str(raw["bar_size"]),
        timezone=str(raw["timezone"]),
        train=_parse_range(raw["train"], "train"),
        valid=_parse_range(valid_raw, "valid"),
        test=_parse_range(raw["test"], "test"),
        folds=tuple(folds),
        purge_minutes=int(purge_raw.get("purge_minutes", 0)),
        embargo_minutes=int(purge_raw.get("embargo_minutes", 0)),
        test_usage=str((raw.get("test") or {}).get("usage", "locked_audit_only")),
    )
    _validate_chronology(manifest)
    return manifest


def summarize_split_manifest(manifest: SplitManifest) -> pd.DataFrame:
    """Return a compact human-readable split summary."""
    rows: list[dict[str, object]] = []
    for name, time_range in [
        ("train", manifest.train),
        ("valid", manifest.valid),
        ("test", manifest.test),
    ]:
        rows.append(
            {
                "name": name,
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat(),
                "duration_minutes": time_range.duration_minutes,
            }
        )
    for fold in manifest.folds:
        rows.append(
            {
                "name": fold.name,
                "start": fold.range.start.isoformat(),
                "end": fold.range.end.isoformat(),
                "duration_minutes": fold.range.duration_minutes,
            }
        )
    return pd.DataFrame(rows)
