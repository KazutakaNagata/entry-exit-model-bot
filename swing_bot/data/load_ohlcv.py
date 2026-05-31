"""Load Binance Japan BTCJPY 1-minute OHLCV files from disk."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from swing_bot.data.schema import BINANCE_KLINE_COLUMNS, SchemaError, normalize_ohlcv

SUPPORTED_SUFFIXES = (".csv", ".csv.gz", ".txt", ".txt.gz", ".parquet", ".pq")

def _has_supported_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in SUPPORTED_SUFFIXES)


def list_ohlcv_files(path: Path | str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if not _has_supported_suffix(path):
            raise ValueError(f"unsupported OHLCV file type: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"OHLCV path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"OHLCV path must be a file or directory: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file() and _has_supported_suffix(p))


def _read_csv_headerless_binance(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=list(BINANCE_KLINE_COLUMNS))


def read_ohlcv_file(path: Path | str, *, keep_extra_columns: bool = False) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return normalize_ohlcv(pd.read_parquet(path), keep_extra_columns=keep_extra_columns)
    if name.endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        raw = pd.read_csv(path)
        try:
            return normalize_ohlcv(raw, keep_extra_columns=keep_extra_columns)
        except SchemaError as first_error:
            # Binance public klines are often headerless. Retry without treating
            # the first candle as a header row.
            try:
                return normalize_ohlcv(_read_csv_headerless_binance(path), keep_extra_columns=keep_extra_columns)
            except SchemaError:
                raise first_error
    raise ValueError(f"unsupported OHLCV file type: {path}")


def load_ohlcv(path: Path | str, *, keep_extra_columns: bool = False) -> pd.DataFrame:
    files = list_ohlcv_files(path)
    if not files:
        raise FileNotFoundError(f"no supported OHLCV files found under: {path}")
    frames = [read_ohlcv_file(p, keep_extra_columns=keep_extra_columns) for p in files]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def save_normalized_ohlcv(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    elif name.endswith((".csv", ".csv.gz")):
        df.to_csv(path, index=False)
    else:
        raise ValueError("normalized output must end with .parquet, .pq, .csv, or .csv.gz")
    return path


def write_ohlcv_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    return save_normalized_ohlcv(df, path)
