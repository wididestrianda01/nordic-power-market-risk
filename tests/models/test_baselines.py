import pandas as pd
import pytest

from nordic_power_risk.models.baselines import (
    naive_forecast,
    quantile_forecast,
    residual_quantiles,
    seasonal_naive_forecast,
)


def test_naive_forecast_returns_lag_24h_column() -> None:
    df = pd.DataFrame({"price_lag_24h": [1.0, 2.0], "price_lag_168h": [9.0, 8.0]})
    pd.testing.assert_series_equal(naive_forecast(df), df["price_lag_24h"])


def test_seasonal_naive_forecast_returns_lag_168h_column() -> None:
    df = pd.DataFrame({"price_lag_24h": [1.0, 2.0], "price_lag_168h": [9.0, 8.0]})
    pd.testing.assert_series_equal(seasonal_naive_forecast(df), df["price_lag_168h"])


def test_residual_quantiles_median_zero_for_symmetric_errors() -> None:
    actual = pd.Series([9.0, 10.0, 11.0])
    point_forecast = pd.Series([10.0, 10.0, 10.0])
    q = residual_quantiles(actual, point_forecast, quantile_grid=(0.5,))
    assert q[0.5] == pytest.approx(0.0)


def test_quantile_forecast_shifts_point_forecast_by_residual_offset() -> None:
    point_forecast = pd.Series([100.0, 200.0])
    forecasts = quantile_forecast(point_forecast, {0.1: -5.0, 0.9: 5.0})
    assert list(forecasts[0.1]) == [95.0, 195.0]
    assert list(forecasts[0.9]) == [105.0, 205.0]
