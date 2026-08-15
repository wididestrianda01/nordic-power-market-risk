"""Naive and seasonal-naive point forecasts, extended to quantile forecasts via
empirical in-sample residual quantiles (train-window only, no leakage).
"""

from __future__ import annotations

import pandas as pd

from nordic_power_risk.models.metrics import QUANTILE_GRID


def naive_forecast(features: pd.DataFrame) -> pd.Series:
    """Random-walk persistence: price(h, d) = price(h, d-1)."""
    return features["price_lag_24h"]


def seasonal_naive_forecast(features: pd.DataFrame) -> pd.Series:
    """Weekly persistence: price(h, d) = price(h, d-7)."""
    return features["price_lag_168h"]


def seasonal_naive_forecast_for(features: pd.DataFrame, value_column: str) -> pd.Series:
    """Weekly persistence for an arbitrary target column (secondary targets)."""
    return features[f"{value_column}_lag_168h"]


def residual_quantiles(
    actual: pd.Series, point_forecast: pd.Series, quantile_grid: tuple[float, ...] = QUANTILE_GRID
) -> dict[float, float]:
    residuals = (actual - point_forecast).dropna()
    return {q: float(residuals.quantile(q)) for q in quantile_grid}


def quantile_forecast(
    point_forecast: pd.Series, residual_q: dict[float, float]
) -> dict[float, pd.Series]:
    return {q: point_forecast + offset for q, offset in residual_q.items()}


__all__ = [
    "naive_forecast",
    "quantile_forecast",
    "residual_quantiles",
    "seasonal_naive_forecast",
    "seasonal_naive_forecast_for",
]
