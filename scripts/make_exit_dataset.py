#!/usr/bin/env python3
"""Build a supervised exit model dataset from OOF entry predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import read_frame, write_frame, write_json  # noqa: E402
from swing_bot.backtest.position_state import ExitDatasetConfig, build_exit_position_dataset  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402
from swing_bot.splits.split_manifest import load_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", type=Path, default=btcjpy_1m_processed_dir() / "ohlcv.parquet")
    parser.add_argument("--features", type=Path, default=btcjpy_1m_processed_dir() / "features_entry_v0.parquet")
    parser.add_argument("--entry-oof", type=Path, required=True, help="Combined OOF predictions from collect/train_entry_grid")
    parser.add_argument("--split", type=Path, required=True, help="Concrete split manifest YAML")
    parser.add_argument("--side", choices=["long", "short"], required=True)
    parser.add_argument("--entry-horizon", type=int, required=True, help="Entry model horizon in minutes")
    parser.add_argument("--exit-lookahead", type=int, required=True, help="Exit hold-delta lookahead K in minutes")
    parser.add_argument("--max-hold-minutes", type=int, default=None)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--candidate-interval-minutes", type=int, default=5)
    parser.add_argument("--min-entry-pred-bps", type=float, default=0.0)
    parser.add_argument("--top-entry-quantile", type=float, default=None)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--no-market-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _default_max_hold(side: str) -> int:
    return 240 if side == "long" else 180


def main() -> None:
    args = parse_args()
    manifest = load_split_manifest(args.split)
    ohlcv = read_frame(args.ohlcv)
    features = None if args.no_market_features else read_frame(args.features)
    entry_oof = read_frame(args.entry_oof)

    config = ExitDatasetConfig(
        side=args.side,
        entry_horizon_minutes=int(args.entry_horizon),
        exit_lookahead_minutes=int(args.exit_lookahead),
        max_hold_minutes=int(args.max_hold_minutes or _default_max_hold(args.side)),
        decision_interval_minutes=int(args.decision_interval_minutes),
        candidate_interval_minutes=int(args.candidate_interval_minutes),
        min_entry_pred_bps=args.min_entry_pred_bps,
        top_entry_quantile=args.top_entry_quantile,
        roundtrip_cost_bps=float(args.roundtrip_cost_bps),
    )
    dataset, summary = build_exit_position_dataset(
        ohlcv=ohlcv,
        features=features,
        entry_oof=entry_oof,
        split_manifest=manifest,
        config=config,
        include_market_features=not args.no_market_features,
    )

    out_dir = args.output_dir or outputs_dir() / "valid" / "exit_dataset" / config.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = write_frame(dataset, out_dir / "exit_dataset.parquet")
    summary_path = write_json(summary, out_dir / "exit_dataset_summary.json")
    write_json(
        {
            "role": "exit_position_state_dataset",
            "dataset_file": str(dataset_path),
            "summary_file": str(summary_path),
            "source_files": {
                "ohlcv": str(args.ohlcv),
                "features": None if args.no_market_features else str(args.features),
                "entry_oof": str(args.entry_oof),
                "split": str(args.split),
            },
            "include_market_features": not args.no_market_features,
            "config": summary,
        },
        out_dir / "exit_dataset_manifest.json",
    )

    print(f"Wrote exit dataset: {dataset_path}")
    print(f"Rows: {summary['exit_rows']}  Episodes: {summary['episode_count']}  Candidates: {summary['candidate_entries']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
