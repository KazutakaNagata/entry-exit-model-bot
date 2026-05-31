"""MFE/MAE diagnostics for fixed-hold entry labels.

These diagnostics summarize the path after entry.  They are labels/diagnostics,
not ordinary market features.  Do not include these columns in entry features.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from swing_bot.labels._time_lookup import price_at_offset_minutes
from swing_bot.labels.costs import side_to_sign


def mfe_column(side: str, horizon_minutes: int) -> str:
    return f"diag_mfe_bps_{side}_H{int(horizon_minutes)}"


def mae_column(side: str, horizon_minutes: int) -> str:
    return f"diag_mae_bps_{side}_H{int(horizon_minutes)}"


def _future_rolling_extreme_by_time(
    df: pd.DataFrame,
    values_col: str,
    window: int,
    *,
    kind: str,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Future rolling extreme using exact one-minute timestamps.

    Missing intermediate bars produce NaN because `min_periods=window` is used
    after reindexing to a complete one-minute calendar.
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted and unique for MFE/MAE diagnostics")
    full_index = pd.date_range(ts.min(), ts.max(), freq="1min", tz="UTC")
    full_values = pd.Series(pd.to_numeric(df[values_col], errors="coerce").to_numpy(), index=ts).reindex(full_index)
    shifted = full_values.shift(-1)
    reversed_series = shifted.iloc[::-1]
    if kind == "max":
        rolled = reversed_series.rolling(window=window, min_periods=window).max()
    elif kind == "min":
        rolled = reversed_series.rolling(window=window, min_periods=window).min()
    else:
        raise ValueError("kind must be 'max' or 'min'")
    return rolled.iloc[::-1].reindex(ts).reset_index(drop=True)


def make_mfe_mae(
    df: pd.DataFrame,
    *,
    side: str,
    horizon_minutes: int,
    timestamp_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """Create MFE/MAE diagnostics for a fixed-hold entry.

    Assumes one row per minute.  The output is masked to NaN whenever the exact
    entry or fixed-horizon exit open is missing, which prevents silently spanning
    data gaps.
    """
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    entry_price = price_at_offset_minutes(df, 1, price_col=open_col, timestamp_col=timestamp_col)
    exit_price = price_at_offset_minutes(df, 1 + horizon, price_col=open_col, timestamp_col=timestamp_col)
    valid_path = entry_price.notna() & exit_price.notna()

    future_high = _future_rolling_extreme_by_time(df, high_col, horizon, kind="max", timestamp_col=timestamp_col)
    future_low = _future_rolling_extreme_by_time(df, low_col, horizon, kind="min", timestamp_col=timestamp_col)

    sign = side_to_sign(side)
    if sign == 1:
        mfe = np.log(future_high / entry_price) * 10000.0
        mae = np.log(future_low / entry_price) * 10000.0
    else:
        mfe = -np.log(future_low / entry_price) * 10000.0
        mae = -np.log(future_high / entry_price) * 10000.0
    out = pd.DataFrame(
        {
            mfe_column(side, horizon): pd.Series(mfe, index=df.index).where(valid_path),
            mae_column(side, horizon): pd.Series(mae, index=df.index).where(valid_path),
        }
    )
    return out


def add_mfe_mae_diagnostics(
    df: pd.DataFrame,
    *,
    horizons_minutes: Sequence[int] = (60, 120, 240),
    sides: Sequence[str] = ("long", "short"),
) -> pd.DataFrame:
    out = df.copy()
    for side in sides:
        for horizon in horizons_minutes:
            diag = make_mfe_mae(out, side=side, horizon_minutes=int(horizon))
            for col in diag.columns:
                out[col] = diag[col]
    return out
