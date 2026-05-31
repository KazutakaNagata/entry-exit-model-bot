"""Data-quality audit for canonical 1-minute OHLCV data."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from swing_bot.data.schema import REQUIRED_COLUMNS, TIMESTAMP_COL, expected_frequency, normalize_ohlcv

@dataclass(frozen=True)
class DataQualitySummary:
    row_count: int
    start_timestamp: str | None
    end_timestamp: str | None
    duplicate_timestamp_rows: int
    missing_1m_bars: int
    non_monotonic_timestamp_steps: int
    invalid_ohlc_rows: int
    negative_volume_rows: int
    nan_required_value_rows: int
    zero_volume_rows: int
    zero_volume_run_count: int
    max_zero_volume_run_length: int
    passed: bool

@dataclass
class DataQualityReport:
    summary: DataQualitySummary
    missing_bars: pd.DataFrame
    duplicate_rows: pd.DataFrame
    invalid_ohlc_rows: pd.DataFrame
    negative_volume_rows: pd.DataFrame
    nan_required_rows: pd.DataFrame
    zero_volume_runs: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.summary)


def _empty_timestamp_frame() -> pd.DataFrame:
    return pd.DataFrame({TIMESTAMP_COL: pd.Series(dtype="datetime64[ns, UTC]")})


def find_invalid_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_ohlcv(df, sort=False) if not set(REQUIRED_COLUMNS).issubset(df.columns) else df.copy()
    flags = pd.DataFrame(index=work.index)
    flags["timestamp_missing"] = work["timestamp"].isna()
    flags["open_non_positive"] = work["open"].isna() | (work["open"] <= 0)
    flags["high_non_positive"] = work["high"].isna() | (work["high"] <= 0)
    flags["low_non_positive"] = work["low"].isna() | (work["low"] <= 0)
    flags["close_non_positive"] = work["close"].isna() | (work["close"] <= 0)
    flags["volume_negative_or_missing"] = work["volume"].isna() | (work["volume"] < 0)
    flags["high_below_low"] = work["high"] < work["low"]
    flags["high_below_open"] = work["high"] < work["open"]
    flags["high_below_close"] = work["high"] < work["close"]
    flags["low_above_open"] = work["low"] > work["open"]
    flags["low_above_close"] = work["low"] > work["close"]
    bad = flags.any(axis=1)
    if not bad.any():
        return pd.DataFrame(columns=list(work.columns) + ["reasons"])
    out = work.loc[bad].copy()
    out["reasons"] = [";".join(flags.columns[flags.loc[idx]]) for idx in out.index]
    return out.reset_index(drop=True)


def _zero_volume_runs(df: pd.DataFrame) -> pd.DataFrame:
    zero = df["volume"].fillna(-1).eq(0)
    if not zero.any():
        return pd.DataFrame(columns=["start_timestamp", "end_timestamp", "length"])
    groups = (zero != zero.shift(fill_value=False)).cumsum()
    rows = []
    for _, run in df.loc[zero].groupby(groups[zero]):
        rows.append({"start_timestamp": run[TIMESTAMP_COL].iloc[0], "end_timestamp": run[TIMESTAMP_COL].iloc[-1], "length": int(len(run))})
    return pd.DataFrame(rows)


def audit_ohlcv(df: pd.DataFrame) -> DataQualityReport:
    data = normalize_ohlcv(df, sort=False)
    original_diffs = data[TIMESTAMP_COL].diff()
    non_monotonic = int((original_diffs.dropna() <= pd.Timedelta(0)).sum())

    sorted_df = data.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)
    timestamps = sorted_df[TIMESTAMP_COL]
    valid_timestamps = timestamps.dropna().drop_duplicates()
    if len(valid_timestamps) >= 2:
        expected = pd.date_range(valid_timestamps.iloc[0], valid_timestamps.iloc[-1], freq=expected_frequency())
        missing_idx = expected.difference(pd.DatetimeIndex(valid_timestamps))
        missing_bars = pd.DataFrame({TIMESTAMP_COL: missing_idx})
    else:
        missing_bars = _empty_timestamp_frame()

    duplicate_rows = sorted_df.loc[timestamps.duplicated(keep=False) & timestamps.notna()].copy()
    invalid_rows = find_invalid_ohlcv_rows(data)
    negative_volume_rows = data.loc[data["volume"] < 0].copy()
    nan_required_rows = data.loc[data[REQUIRED_COLUMNS].isna().any(axis=1)].copy()
    zero_runs = _zero_volume_runs(sorted_df)
    max_zero_run = int(zero_runs["length"].max()) if not zero_runs.empty else 0

    fatal = sum([len(duplicate_rows), len(missing_bars), len(invalid_rows), len(negative_volume_rows), len(nan_required_rows), non_monotonic])
    summary = DataQualitySummary(
        row_count=int(len(data)),
        start_timestamp=None if valid_timestamps.empty else str(valid_timestamps.iloc[0]),
        end_timestamp=None if valid_timestamps.empty else str(valid_timestamps.iloc[-1]),
        duplicate_timestamp_rows=int(len(duplicate_rows)),
        missing_1m_bars=int(len(missing_bars)),
        non_monotonic_timestamp_steps=non_monotonic,
        invalid_ohlc_rows=int(len(invalid_rows)),
        negative_volume_rows=int(len(negative_volume_rows)),
        nan_required_value_rows=int(len(nan_required_rows)),
        zero_volume_rows=int(data["volume"].eq(0).sum()),
        zero_volume_run_count=int(len(zero_runs)),
        max_zero_volume_run_length=max_zero_run,
        passed=fatal == 0,
    )
    return DataQualityReport(summary, missing_bars, duplicate_rows, invalid_rows, negative_volume_rows, nan_required_rows, zero_runs)


def write_quality_report(report: DataQualityReport, output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    frames = {
        "missing_1m_bars.csv": report.missing_bars,
        "duplicate_timestamps.csv": report.duplicate_rows,
        "invalid_ohlc.csv": report.invalid_ohlc_rows,
        "negative_volume.csv": report.negative_volume_rows,
        "nan_required_rows.csv": report.nan_required_rows,
        "zero_volume_runs.csv": report.zero_volume_runs,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    return summary_path
