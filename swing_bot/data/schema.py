"""Canonical OHLCV schema utilities for BTCJPY 1-minute data.

Canonical columns are exactly:
    timestamp, open, high, low, close, volume

`timestamp` is timezone-aware UTC and represents the candle open time.  The
normalizer does not fill missing bars, drop duplicates, or repair prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TIMESTAMP_COL = "timestamp"
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
STANDARD_OHLCV_COLUMNS = tuple(REQUIRED_COLUMNS)
BINANCE_KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
)

COLUMN_ALIASES = {
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp", "date": "timestamp",
    "open_time": "timestamp", "open time": "timestamp", "open_time_ms": "timestamp",
    "opentime": "timestamp", "bar_time": "timestamp",
    "o": "open", "open": "open",
    "h": "high", "high": "high",
    "l": "low", "low": "low",
    "c": "close", "close": "close",
    "v": "volume", "vol": "volume", "volume": "volume",
}

class SchemaError(ValueError):
    """Raised when raw data cannot be normalized to the project schema."""

@dataclass(frozen=True)
class OhlcvDatasetId:
    source: str = "binance_japan"
    symbol: str = "BTCJPY"
    timeframe: str = "1m"

    def slug(self) -> str:
        return f"{self.source}_{self.symbol}_{self.timeframe}"


def expected_frequency() -> str:
    return "1min"


def _clean_column_name(column: object) -> str:
    return str(column).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_clean_column_name(c) for c in out.columns]
    out = out.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in out.columns})
    return out


def infer_timestamp_unit(values: pd.Series) -> str | None:
    numeric = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return None
    median_abs = float(np.nanmedian(np.abs(numeric.to_numpy(dtype="float64"))))
    if median_abs >= 1e17:
        return "ns"
    if median_abs >= 1e14:
        return "us"
    if median_abs >= 1e11:
        return "ms"
    if median_abs >= 1e8:
        return "s"
    return None


def parse_timestamp_utc(values: pd.Series) -> pd.Series:
    unit = infer_timestamp_unit(values)
    if unit is not None:
        parsed = pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.Series(parsed, index=values.index, name="timestamp")


def normalize_ohlcv(
    df: pd.DataFrame,
    *,
    sort: bool = True,
    drop_duplicate_timestamps: bool = False,
    keep_extra_columns: bool = False,
) -> pd.DataFrame:
    """Normalize raw OHLCV data to canonical schema.

    The default returns only canonical columns. Extra columns can be preserved for
    audit/debugging by setting `keep_extra_columns=True`, but feature code should
    depend only on manifest-approved columns.
    """
    if df.empty:
        raise SchemaError("OHLCV input is empty")

    normalized = _rename_known_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in normalized.columns]
    if missing:
        raise SchemaError("missing required OHLCV columns after normalization: " + ", ".join(missing))

    if keep_extra_columns:
        canonical = normalized.loc[:, ~normalized.columns.duplicated()].copy()
        first_cols = REQUIRED_COLUMNS
        rest_cols = [c for c in canonical.columns if c not in first_cols]
        out = canonical[first_cols + rest_cols].copy()
    else:
        out = normalized.loc[:, REQUIRED_COLUMNS].copy()

    out["timestamp"] = parse_timestamp_utc(out["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if sort:
        out = out.sort_values("timestamp", kind="mergesort")
    if drop_duplicate_timestamps:
        out = out.drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)

# Backward-friendly alias used by some docs/tests.
normalize_ohlcv_frame = normalize_ohlcv


def ensure_columns_absent(df: pd.DataFrame, forbidden_substrings: Iterable[str]) -> None:
    lowered = {col: str(col).lower() for col in df.columns}
    violations = [col for col, lower in lowered.items() if any(s.lower() in lower for s in forbidden_substrings)]
    if violations:
        raise SchemaError("forbidden columns found: " + ", ".join(map(str, violations)))
