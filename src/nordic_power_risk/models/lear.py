"""LEAR (LASSO Estimated AutoRegressive) point forecaster (Phase 2 ticket 02).

Fits a LassoCV on feature_day_ahead's lag + calendar features per fold; quantile
extension reuses the ladder's existing residual-quantile machinery (models.baselines),
which is itself a QRA-style approach, so no separate quantile-regression path is needed.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ("price_lag_24h", "price_lag_168h")
CATEGORICAL_FEATURES = ("hour_of_day", "day_of_week", "month")
BOOLEAN_FEATURES = ("is_weekend", "is_holiday")
MIN_TRAIN_ROWS = 10


def _missing_mask(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return df[list(columns)].isna().any(axis=1)


def _design_matrix(df: pd.DataFrame, columns: pd.Index | None = None) -> pd.DataFrame:
    numeric = df[list(NUMERIC_FEATURES)].astype(float)
    categorical = pd.get_dummies(
        df[list(CATEGORICAL_FEATURES)].astype("category"), prefix=list(CATEGORICAL_FEATURES)
    )
    boolean = df[list(BOOLEAN_FEATURES)].astype(float)
    design = pd.concat([numeric, categorical, boolean], axis=1)
    if columns is not None:
        design = design.reindex(columns=columns, fill_value=0.0)
    return design


def lear_forecast(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit LASSO on train, return point forecasts for train (residual fitting) and test."""
    fit_rows = train.dropna(subset=[*NUMERIC_FEATURES, "price_eur_mwh"])
    if len(fit_rows) < MIN_TRAIN_ROWS:
        return (
            pd.Series(float("nan"), index=train.index),
            pd.Series(float("nan"), index=test.index),
        )

    x_train = _design_matrix(fit_rows)
    y_train = fit_rows["price_eur_mwh"].to_numpy()

    model = Pipeline(
        [("scale", StandardScaler()), ("lasso", LassoCV(cv=5, max_iter=10_000))]
    )
    model.fit(x_train, y_train)

    # Rows with missing lags can't be scored by the model; fill for predict() and
    # blank the result afterward rather than dropping (keeps the index aligned).
    x_train_full = _design_matrix(train, columns=x_train.columns).fillna(0.0)
    x_test_full = _design_matrix(test, columns=x_train.columns).fillna(0.0)
    train_point = pd.Series(model.predict(x_train_full), index=train.index)
    test_point = (
        pd.Series(model.predict(x_test_full), index=test.index)
        if len(x_test_full) > 0
        else pd.Series(dtype=float, index=test.index)
    )

    train_missing = _missing_mask(train, NUMERIC_FEATURES)
    test_missing = _missing_mask(test, NUMERIC_FEATURES)
    train_point[train_missing] = float("nan")
    test_point[test_missing] = float("nan")

    return train_point, test_point


__all__ = ["lear_forecast"]
