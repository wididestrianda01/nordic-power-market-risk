import numpy as np
import pandas as pd

from nordic_power_risk.models.lgbm import SECONDARY_QUANTILE_GRID, lgbm_quantile_forecast


def _make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 50 + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2025-01-01", periods=n, freq="h"),
            "price": price,
            "price_lag_24h": price + rng.normal(0, 1, n),
            "price_lag_168h": price + rng.normal(0, 1, n),
            "hour_of_day": np.arange(n) % 24,
            "day_of_week": np.arange(n) % 7,
            "month": 1,
            "is_weekend": False,
            "is_holiday": False,
        }
    )


def test_lgbm_quantile_forecast_returns_one_series_per_quantile() -> None:
    train = _make_frame(300, seed=1)
    test = _make_frame(50, seed=2)

    forecasts = lgbm_quantile_forecast(train, test, "price")

    assert set(forecasts) == set(SECONDARY_QUANTILE_GRID)
    for series in forecasts.values():
        assert len(series) == len(test)


def test_lgbm_quantile_forecast_is_monotonic_across_quantiles() -> None:
    train = _make_frame(300, seed=3)
    test = _make_frame(50, seed=4)

    forecasts = lgbm_quantile_forecast(train, test, "price")

    low = forecasts[0.1].to_numpy()
    mid = forecasts[0.5].to_numpy()
    high = forecasts[0.9].to_numpy()
    assert (low <= mid + 1e-6).all()
    assert (mid <= high + 1e-6).all()


def test_lgbm_quantile_forecast_masks_missing_lag_rows() -> None:
    train = _make_frame(300, seed=5)
    test = _make_frame(20, seed=6)
    test.loc[test.index[0], "price_lag_24h"] = np.nan

    forecasts = lgbm_quantile_forecast(train, test, "price")

    assert forecasts[0.5].iloc[0] != forecasts[0.5].iloc[0]  # NaN


def test_lgbm_quantile_forecast_returns_nan_when_too_few_train_rows() -> None:
    train = _make_frame(5, seed=7)
    test = _make_frame(10, seed=8)

    forecasts = lgbm_quantile_forecast(train, test, "price")

    for series in forecasts.values():
        assert series.isna().all()


def test_lgbm_quantile_forecast_uses_target_specific_lag_columns() -> None:
    train = _make_frame(300, seed=9).rename(
        columns={
            "price": "imbalance_price_eur_mwh",
            "price_lag_24h": "imbalance_price_eur_mwh_lag_24h",
            "price_lag_168h": "imbalance_price_eur_mwh_lag_168h",
        }
    )
    test = _make_frame(30, seed=10).rename(
        columns={
            "price": "imbalance_price_eur_mwh",
            "price_lag_24h": "imbalance_price_eur_mwh_lag_24h",
            "price_lag_168h": "imbalance_price_eur_mwh_lag_168h",
        }
    )

    forecasts = lgbm_quantile_forecast(train, test, "imbalance_price_eur_mwh")

    assert set(forecasts) == set(SECONDARY_QUANTILE_GRID)
    assert forecasts[0.5].notna().any()
