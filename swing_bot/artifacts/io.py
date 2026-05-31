"""Artifact IO helpers used by scripts.

These helpers intentionally stay tiny.  They avoid hidden global state and do
not decide research policy; scripts pass explicit paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_yaml(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(data: Any, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return p



def read_json(path: Path | str) -> Any:
    """Read JSON artifact from disk."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _csv_fallback_path(path: Path) -> Path:
    if path.name.lower().endswith(".parquet"):
        return path.with_suffix(".csv")
    if path.name.lower().endswith(".pq"):
        return path.with_suffix(".csv")
    return path.with_name(path.name + ".csv")


def write_frame(df: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    name = p.name.lower()
    if name.endswith((".parquet", ".pq")):
        try:
            df.to_parquet(p, index=False)
            return p
        except ImportError:
            # Some lightweight review environments do not have pyarrow installed.
            # The project requirements include pyarrow, so normal runs still write
            # parquet; this fallback keeps unit tests and code review usable.
            fallback = _csv_fallback_path(p)
            df.to_csv(fallback, index=False)
            return fallback
    if name.endswith((".csv", ".csv.gz")):
        df.to_csv(p, index=False)
        return p
    raise ValueError(f"unsupported table output extension: {p}")


def read_frame(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() and p.name.lower().endswith((".parquet", ".pq")):
        fallback = _csv_fallback_path(p)
        if fallback.exists():
            p = fallback
    name = p.name.lower()
    if name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(p)
    elif name.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported table input extension: {p}")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
