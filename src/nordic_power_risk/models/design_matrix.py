"""Shared feature-matrix prep for the ladder's model-backed rungs (LEAR, DNN).

Both rungs train on the same lag + calendar features from feature_day_ahead and
share the same missing-lag handling, so the design-matrix construction lives here
once rather than duplicated per rung.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = ("price_lag_24h", "price_lag_168h")
CATEGORICAL_FEATURES = ("hour_of_day", "day_of_week", "month")
BOOLEAN_FEATURES = ("is_weekend", "is_holiday")
MIN_TRAIN_ROWS = 10


def missing_mask(df: pd.DataFrame, columns: tuple[str, ...] = NUMERIC_FEATURES) -> pd.Series:
    return df[list(columns)].isna().any(axis=1)


def build_design_matrix(
    df: pd.DataFrame,
    columns: pd.Index | None = None,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
) -> pd.DataFrame:
    numeric = df[list(numeric_features)].astype(float)
    categorical = pd.get_dummies(
        df[list(CATEGORICAL_FEATURES)].astype("category"), prefix=list(CATEGORICAL_FEATURES)
    )
    boolean = df[list(BOOLEAN_FEATURES)].astype(float)
    design = pd.concat([numeric, categorical, boolean], axis=1)
    if columns is not None:
        design = design.reindex(columns=columns, fill_value=0.0)
    return design


__all__ = [
    "BOOLEAN_FEATURES",
    "CATEGORICAL_FEATURES",
    "MIN_TRAIN_ROWS",
    "NUMERIC_FEATURES",
    "build_design_matrix",
    "missing_mask",
]
