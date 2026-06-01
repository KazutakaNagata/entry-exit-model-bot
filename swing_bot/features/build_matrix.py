"""Build reviewed feature matrices from canonical 1-minute OHLCV."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from swing_bot.data.schema import REQUIRED_COLUMNS, SchemaError
from swing_bot.features.leakage_audit import audit_feature_frame
from swing_bot.features.manifest import FeatureSpec, ensure_unique_feature_names
from swing_bot.features.registry import available_families, get_feature_builder


class FeatureBuildError(ValueError):
    """Raised when features cannot be built safely."""


def _validate_ohlcv_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError("missing canonical OHLCV columns for feature build: " + ", ".join(missing))


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    *,
    include_families: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    strict_audit: bool = True,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build a timestamp-indexed feature matrix and manifest specs.

    The output contains ``timestamp`` plus feature columns only.  Raw OHLCV columns
    and target/diagnostic columns are intentionally excluded.
    """
    _validate_ohlcv_input(ohlcv)
    families = list(include_families or available_families())
    unknown = sorted(set(families) - set(available_families()))
    if unknown:
        raise FeatureBuildError(f"unknown/unimplemented feature families: {unknown}; available={available_families()}")

    pieces: list[pd.DataFrame] = []
    specs: list[FeatureSpec] = []
    for family in families:
        builder = get_feature_builder(family)
        family_features, family_specs = builder(ohlcv)
        pieces.append(family_features)
        specs.extend(family_specs)

    ensure_unique_feature_names(specs)
    feature_cols = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=ohlcv.index)
    out = pd.DataFrame({"timestamp": ohlcv["timestamp"].to_numpy()}, index=ohlcv.index)
    out = pd.concat([out, feature_cols], axis=1)

    audit = audit_feature_frame(out, specs, exclude_patterns=exclude_patterns)
    if strict_audit and not audit.ok:
        raise FeatureBuildError("feature audit failed: " + "; ".join(audit.violations))
    return out.reset_index(drop=True), specs


def feature_summary(features: pd.DataFrame, specs: Sequence[FeatureSpec]) -> dict:
    """Return a compact JSON-friendly summary of a feature matrix."""
    feature_cols = [c for c in features.columns if c != "timestamp"]
    null_rates = features[feature_cols].isna().mean().sort_values(ascending=False) if feature_cols else pd.Series(dtype="float64")
    family_counts: dict[str, int] = {}
    for spec in specs:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
    top_null_columns = [
        {"column": str(name), "null_rate": float(rate)}
        for name, rate in null_rates.head(20).items()
    ] if len(null_rates) else []
    return {
        "rows": int(len(features)),
        "feature_count": int(len(feature_cols)),
        "families": family_counts,
        "max_null_rate": float(null_rates.iloc[0]) if len(null_rates) else 0.0,
        "columns_with_nulls": int((features[feature_cols].isna().any()).sum()) if feature_cols else 0,
        "columns_null_rate_gt_50pct": int((null_rates > 0.50).sum()) if len(null_rates) else 0,
        "columns_null_rate_gt_95pct": int((null_rates > 0.95).sum()) if len(null_rates) else 0,
        "top_null_columns": top_null_columns,
    }
