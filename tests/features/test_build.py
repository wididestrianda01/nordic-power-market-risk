from datetime import datetime, timedelta

import pandas as pd
import pytest

from nordic_power_risk.facts.rules import day_ahead_issue_time
from nordic_power_risk.features.build import build_day_ahead_features


def _price_df(n_hours: int, start: datetime) -> pd.DataFrame:
    event_times = [start + timedelta(hours=h) for h in range(n_hours)]
    return pd.DataFrame(
        {
            "event_time": event_times,
            "issue_time": [day_ahead_issue_time(t) for t in event_times],
            "price_eur_mwh": [float(h) for h in range(n_hours)],
        }
    )


def test_lag_features_carry_correct_historical_value() -> None:
    df = _price_df(200, datetime(2025, 1, 1))
    features = build_day_ahead_features(df)

    row = features[features["event_time"] == datetime(2025, 1, 9)].iloc[0]
    assert row["price_lag_24h"] == pytest.approx(168.0)
    assert row["price_lag_168h"] == pytest.approx(24.0)


def test_lag_feature_missing_before_history_available() -> None:
    df = _price_df(10, datetime(2025, 1, 1))
    features = build_day_ahead_features(df)
    assert features["price_lag_24h"].isna().all()


def test_calendar_features_present() -> None:
    df = _price_df(48, datetime(2025, 1, 1))
    features = build_day_ahead_features(df)
    assert {"hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"} <= set(
        features.columns
    )
    new_years = features[features["event_time"] == datetime(2025, 1, 1)].iloc[0]
    assert bool(new_years["is_holiday"]) is True


def test_raises_on_look_ahead_leak() -> None:
    df = _price_df(200, datetime(2025, 1, 1))
    # Corrupt one row's issue_time so a lag source appears "known" after its target row.
    df.loc[df["event_time"] == datetime(2025, 1, 5), "issue_time"] = datetime(2030, 1, 1)
    with pytest.raises(ValueError, match="look-ahead leak"):
        build_day_ahead_features(df)
