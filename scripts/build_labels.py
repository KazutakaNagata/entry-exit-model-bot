#!/usr/bin/env python3
"""Build cost-aware entry/exit labels from canonical 1-minute OHLCV.

This script creates labels only.  It does not build features, train models, tune
thresholds, or evaluate test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.data.load_ohlcv import load_ohlcv  # noqa: E402
from swing_bot.labels.entry_net_return import add_entry_net_return_targets  # noqa: E402
from swing_bot.labels.exit_hold_delta import add_exit_hold_delta_targets  # noqa: E402
from swing_bot.labels.mfe_mae import add_mfe_mae_diagnostics  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402

DEFAULT_INPUT = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_OUTPUT = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_ints(csv_text: str | None, default: list[int]) -> list[int]:
    if not csv_text:
        return list(default)
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build entry net-return and exit hold-delta labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical OHLCV parquet/csv or directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path for OHLCV + labels.")
    parser.add_argument("--entry-config", type=Path, default=Path("configs/labels/entry_net_return_v0.yaml"), help="Entry label config.")
    parser.add_argument("--exit-config", type=Path, default=Path("configs/labels/exit_hold_delta_v0.yaml"), help="Exit label config.")
    parser.add_argument("--roundtrip-cost-bps", type=float, default=None, help="Override entry roundtrip cost bps.")
    parser.add_argument("--entry-horizons", type=str, default=None, help="Comma-separated entry horizons in minutes, e.g. 60,120,240.")
    parser.add_argument("--include-mfe-mae", action="store_true", help="Also append MFE/MAE diagnostics. These are not features.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing output.")
    return parser.parse_args(argv)


def _label_summary(df: pd.DataFrame) -> dict[str, dict[str, int | float]]:
    label_cols = [c for c in df.columns if c.startswith("target_") or c.startswith("diag_")]
    summary: dict[str, dict[str, int | float]] = {}
    for col in label_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "non_null": int(s.notna().sum()),
            "null": int(s.isna().sum()),
            "mean": float(s.mean()) if s.notna().any() else float("nan"),
            "p05": float(s.quantile(0.05)) if s.notna().any() else float("nan"),
            "p95": float(s.quantile(0.95)) if s.notna().any() else float("nan"),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entry_cfg = _load_yaml(args.entry_config)
    exit_cfg = _load_yaml(args.exit_config)

    roundtrip_cost = (
        float(args.roundtrip_cost_bps)
        if args.roundtrip_cost_bps is not None
        else float((entry_cfg.get("costs") or {}).get("roundtrip_cost_bps", 15.0))
    )
    entry_horizons = _parse_ints(args.entry_horizons, entry_cfg.get("horizons_minutes") or [60, 120, 240])
    entry_sides = entry_cfg.get("sides") or ["long", "short"]
    exit_lookaheads = exit_cfg.get("lookaheads_minutes") or {"long": [30, 60], "short": [15, 30]}

    df = load_ohlcv(args.input)
    labeled = add_entry_net_return_targets(
        df,
        horizons_minutes=entry_horizons,
        sides=entry_sides,
        roundtrip_cost_bps=roundtrip_cost,
    )
    labeled = add_exit_hold_delta_targets(labeled, lookaheads_minutes=exit_lookaheads)
    if args.include_mfe_mae:
        labeled = add_mfe_mae_diagnostics(labeled, horizons_minutes=entry_horizons, sides=entry_sides)

    summary = {
        "rows": int(len(labeled)),
        "roundtrip_cost_bps": roundtrip_cost,
        "entry_horizons_minutes": entry_horizons,
        "entry_sides": entry_sides,
        "exit_lookaheads_minutes": exit_lookaheads,
        "labels": _label_summary(labeled),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.output.name.lower()
        if suffix.endswith((".parquet", ".pq")):
            labeled.to_parquet(args.output, index=False)
        elif suffix.endswith((".csv", ".csv.gz")):
            labeled.to_csv(args.output, index=False)
        else:
            raise ValueError("--output must end with .parquet, .pq, .csv, or .csv.gz")
        print(f"Wrote labels: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
