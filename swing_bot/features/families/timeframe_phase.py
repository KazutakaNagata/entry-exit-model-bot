"""Calendar phase features for common human-watched timeframes.

These features do not use market future data.  They encode where the current 1m
bar sits inside 5m/15m/60m/240m anchored intervals, e.g. "minute 3 of a 5m bar".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec
from swing_bot.features.families.timeframe_utils import ensure_utc_timestamp_series, timeframe_label

PHASE_TIMEFRAMES = (5, 15, 60, 240)


def build_timeframe_phase_features(
    df: pd.DataFrame,
    *,
    timeframes: tuple[int, ...] = PHASE_TIMEFRAMES,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build deterministic modulo-time features for anchored intervals."""
    ts = ensure_utc_timestamp_series(df)
    # Minutes since Unix epoch; modulo is robust across days and UTC calendar.
    epoch_minutes = (ts.astype("int64") // (60 * 1_000_000_000)).astype("int64")
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    for minutes in timeframes:
        label = timeframe_label(minutes)
        phase = (epoch_minutes % minutes).astype("float64")
        frac = phase / float(minutes)
        to_close = (minutes - 1.0) - phase
        columns = {
            f"phase_{label}_minute": phase,
            f"phase_{label}_frac": frac,
            f"phase_{label}_sin": np.sin(2.0 * np.pi * frac),
            f"phase_{label}_cos": np.cos(2.0 * np.pi * frac),
            f"phase_{label}_minutes_to_close": to_close,
            f"phase_{label}_is_first_minute": (phase == 0.0).astype("float64"),
            f"phase_{label}_is_last_minute": (phase == float(minutes - 1)).astype("float64"),
        }
        for name, values in columns.items():
            out[name] = values
            specs.append(
                FeatureSpec(
                    name,
                    "timeframe_phase",
                    0,
                    source="calendar_timestamp",
                    description=f"UTC anchored {label} phase feature; no market future data.",
                )
            )

    minute_of_day = (epoch_minutes % (24 * 60)).astype("float64")
    day_frac = minute_of_day / float(24 * 60)
    out["phase_day_frac"] = day_frac
    specs.append(FeatureSpec("phase_day_frac", "timeframe_phase", 0, source="calendar_timestamp", description="UTC minute-of-day fraction."))
    out["phase_day_sin"] = np.sin(2.0 * np.pi * day_frac)
    specs.append(FeatureSpec("phase_day_sin", "timeframe_phase", 0, source="calendar_timestamp", description="UTC minute-of-day sine."))
    out["phase_day_cos"] = np.cos(2.0 * np.pi * day_frac)
    specs.append(FeatureSpec("phase_day_cos", "timeframe_phase", 0, source="calendar_timestamp", description="UTC minute-of-day cosine."))

    return out, specs
