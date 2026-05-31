"""Internal timestamp lookup helpers for label generation."""
from __future__ import annotations

import pandas as pd


def canonical_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Series:
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    if ts.isna().any():
        raise ValueError("timestamp column contains NaT values")
    if ts.duplicated().any():
        raise ValueError("timestamp column contains duplicates; label lookup would be ambiguous")
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamp column must be sorted ascending before label generation")
    return pd.Series(ts, index=df.index, name=timestamp_col)


def price_at_offset_minutes(
    df: pd.DataFrame,
    offset_minutes: int,
    *,
    price_col: str = "open",
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Look up `price_col` at timestamp + offset_minutes.

    Exact timestamp lookup is used instead of row shift.  If a target minute is
    missing because of a data gap or file boundary, the returned value is NaN.
    """
    if price_col not in df.columns:
        raise ValueError(f"missing price column: {price_col}")
    ts = canonical_timestamps(df, timestamp_col=timestamp_col)
    price_by_time = pd.Series(pd.to_numeric(df[price_col], errors="coerce").to_numpy(), index=ts)
    target_times = ts + pd.Timedelta(minutes=int(offset_minutes))
    values = price_by_time.reindex(pd.DatetimeIndex(target_times)).to_numpy(dtype="float64")
    return pd.Series(values, index=df.index)
