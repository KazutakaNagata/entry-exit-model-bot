"""Feature leakage and manifest audit helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from typing import Iterable, Sequence

import pandas as pd

from swing_bot.data.schema import REQUIRED_COLUMNS
from swing_bot.features.manifest import FeatureSpec

FORBIDDEN_SUBSTRINGS = (
    "target",
    "future",
    "label",
    "hold_delta",
    "mfe",
    "mae",
    "diag_",
    "entry_price",
    "exit_price",
)

FORBIDDEN_EXACT_COLUMNS = set(REQUIRED_COLUMNS) - {"timestamp"}


@dataclass(frozen=True)
class FeatureAuditResult:
    ok: bool
    feature_count: int
    manifest_count: int
    violations: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _matches_any_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def audit_feature_columns(
    feature_columns: Sequence[str],
    specs: Sequence[FeatureSpec],
    *,
    exclude_patterns: Sequence[str] | None = None,
    forbidden_substrings: Sequence[str] = FORBIDDEN_SUBSTRINGS,
) -> FeatureAuditResult:
    """Audit feature names and manifest rows for obvious leakage hazards.

    This is not a proof of no leakage; it is a guardrail.  Human review of each
    feature formula is still required.
    """
    exclude_patterns = tuple(exclude_patterns or ())
    feature_columns = list(feature_columns)
    spec_by_name = {spec.name: spec for spec in specs}
    violations: list[str] = []
    warnings: list[str] = []

    if len(feature_columns) != len(set(feature_columns)):
        violations.append("duplicate feature columns found")
    if len(spec_by_name) != len(specs):
        violations.append("duplicate feature names found in manifest")

    missing_specs = sorted(set(feature_columns) - set(spec_by_name))
    extra_specs = sorted(set(spec_by_name) - set(feature_columns))
    if missing_specs:
        violations.append("features missing manifest specs: " + ", ".join(missing_specs[:20]))
    if extra_specs:
        warnings.append("manifest has specs not present in feature matrix: " + ", ".join(extra_specs[:20]))

    for name in feature_columns:
        lower = name.lower()
        if name in FORBIDDEN_EXACT_COLUMNS:
            violations.append(f"raw OHLCV column is not allowed as feature: {name}")
        if any(token in lower for token in forbidden_substrings):
            violations.append(f"forbidden leakage-like substring in feature name: {name}")
        if exclude_patterns and _matches_any_pattern(name, exclude_patterns):
            violations.append(f"feature matches excluded pattern: {name}")

    for spec in specs:
        if spec.uses_future:
            violations.append(f"manifest says uses_future=true: {spec.name}")
        if spec.lookback_minutes < 0:
            violations.append(f"negative lookback in manifest: {spec.name}")
        if not spec.family:
            violations.append(f"missing family in manifest: {spec.name}")

    return FeatureAuditResult(
        ok=not violations,
        feature_count=len(feature_columns),
        manifest_count=len(specs),
        violations=violations,
        warnings=warnings,
    )


def audit_feature_frame(
    features: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    *,
    timestamp_col: str = "timestamp",
    exclude_patterns: Sequence[str] | None = None,
) -> FeatureAuditResult:
    """Audit a feature matrix DataFrame and manifest."""
    feature_columns = [c for c in features.columns if c != timestamp_col]
    return audit_feature_columns(feature_columns, specs, exclude_patterns=exclude_patterns)
