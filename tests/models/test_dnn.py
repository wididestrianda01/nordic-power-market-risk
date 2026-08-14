from __future__ import annotations

import numpy as np
import pandas as pd

from nordic_power_risk.models.dnn import dnn_forecast


def _make_features(n_days: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    event_time = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    price_lag_24h = 30.0 + 5.0 * np.sin(np.arange(len(event_time)) / 24 * 2 * np.pi)
    price_lag_168h = price_lag_24h + rng.normal(0, 1.0, len(event_time))
    price = price_lag_24h + rng.normal(0, 0.5, len(event_time))
    df = pd.DataFrame(
        {
            "event_time": event_time,
            "price_eur_mwh": price,
            "price_lag_24h": price_lag_24h,
            "price_lag_168h": price_lag_168h,
            "hour_of_day": event_time.hour,
            "day_of_week": event_time.dayofweek,
            "month": event_time.month,
            "is_weekend": event_time.dayofweek >= 5,
            "is_holiday": False,
        }
    )
    df.loc[:7, "price_lag_168h"] = np.nan  # first week has no 168h history
    return df


class TestDnnForecast:
    def test_returns_series_indexed_like_inputs(self) -> None:
        df = _make_features()
        train, test = df.iloc[:800], df.iloc[800:]

        train_point, test_point = dnn_forecast(train, test)

        assert list(train_point.index) == list(train.index)
        assert list(test_point.index) == list(test.index)

    def test_predictions_track_price_lag_24h_signal(self) -> None:
        df = _make_features()
        train, test = df.iloc[:800], df.iloc[800:]

        _, test_point = dnn_forecast(train, test)

        valid = test_point.notna()
        correlation = np.corrcoef(test_point[valid], test.loc[valid, "price_eur_mwh"])[0, 1]
        assert correlation > 0.8

    def test_rows_with_missing_lags_are_nan(self) -> None:
        df = _make_features()
        train, test = df.iloc[:800], df.iloc[800:]

        train_point, _ = dnn_forecast(train, test)

        assert train_point.iloc[:7].isna().all()

    def test_below_min_train_rows_returns_all_nan(self) -> None:
        df = _make_features()
        train, test = df.iloc[:5], df.iloc[800:820]

        train_point, test_point = dnn_forecast(train, test)

        assert train_point.isna().all()
        assert test_point.isna().all()

    def test_empty_test_frame_returns_empty_series(self) -> None:
        df = _make_features()
        train, test = df.iloc[:800], df.iloc[0:0]

        _, test_point = dnn_forecast(train, test)

        assert test_point.empty
