"""Small LightGBM helpers.

The project intentionally keeps model wrappers thin.  Research policy belongs in
scripts/configs, not in a hidden framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swing_bot.artifacts.io import read_yaml


class ModelTrainingError(RuntimeError):
    """Raised when model training cannot proceed safely."""


def load_lgbm_params(config_path: Path | str) -> dict[str, Any]:
    """Load LightGBM params from a YAML config."""
    cfg = read_yaml(config_path)
    params = dict(cfg.get("params") or {})
    if not params:
        raise ModelTrainingError(f"model config has no params: {config_path}")
    return params


def make_lgbm_regressor(params: dict[str, Any]):
    """Create an ``LGBMRegressor`` lazily so importing tests stays lightweight."""
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModelTrainingError("lightgbm is required to train entry models") from exc
    return LGBMRegressor(**params)


def numeric_feature_columns(df: pd.DataFrame, *, exclude: set[str] | None = None) -> list[str]:
    """Return numeric feature columns excluding timestamp/targets/diagnostics."""
    exclude = set(exclude or set()) | {"timestamp"}
    cols: list[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if col in exclude:
            continue
        if lower.startswith("target_") or "hold_delta" in lower or lower.startswith("diag_"):
            continue
        if "mfe" in lower or "mae" in lower or "future" in lower or "label" in lower:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def assert_no_forbidden_feature_columns(feature_cols: list[str]) -> None:
    forbidden_tokens = ("target", "future", "label", "hold_delta", "mfe", "mae", "diag_", "entry_price", "exit_price")
    bad = [col for col in feature_cols if any(tok in col.lower() for tok in forbidden_tokens)]
    if bad:
        raise ModelTrainingError("forbidden leakage-like feature columns: " + ", ".join(bad[:20]))


def fit_regressor(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
):
    """Fit one LightGBM regressor on non-null target rows."""
    assert_no_forbidden_feature_columns(feature_cols)
    train = train_df.dropna(subset=[target_col]).copy()
    if train.empty:
        raise ModelTrainingError("no non-null target rows for training")
    x_train = train[feature_cols]
    y_train = pd.to_numeric(train[target_col], errors="coerce")
    ok = y_train.notna()
    if int(ok.sum()) == 0:
        raise ModelTrainingError("target is all NaN after numeric conversion")
    model = make_lgbm_regressor(params)
    model.fit(x_train.loc[ok], y_train.loc[ok])
    return model


def predict_regressor(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict with a fitted model."""
    assert_no_forbidden_feature_columns(feature_cols)
    return np.asarray(model.predict(df[feature_cols]), dtype="float64")


def feature_importance_frame(model, feature_cols: list[str], *, fold: str) -> pd.DataFrame:
    """Return split/gain importances for one fitted LightGBM model."""
    booster = getattr(model, "booster_", None)
    if booster is None:
        return pd.DataFrame(columns=["fold", "feature", "importance_split", "importance_gain"])
    split = booster.feature_importance(importance_type="split")
    gain = booster.feature_importance(importance_type="gain")
    return pd.DataFrame({
        "fold": fold,
        "feature": feature_cols,
        "importance_split": split,
        "importance_gain": gain,
    })


def save_lgbm_model(model, path: Path | str) -> Path:
    """Save the underlying LightGBM booster as text."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ModelTrainingError("model has no fitted booster_")
    booster.save_model(str(p))
    return p
