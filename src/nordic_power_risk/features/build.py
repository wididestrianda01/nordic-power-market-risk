"""Day-ahead SE3 feature table (T09): lagged price + calendar/holiday, issue-time gated.

ENTSO-E day-ahead load/generation forecasts are named in T09 as an allowed feature
source but this repo has no ingest client for them yet (only day-ahead *price* is
ingested) -- ponytail: descoped, add load/gen features once that ingest source exists.
"""

from __future__ import annotations

import holidays
import pandas as pd

LAG_HOURS = (24, 168)


def build_day_ahead_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """price_df: columns event_time, issue_time, price_eur_mwh from fact_day_ahead_price."""
    df = price_df.sort_values("event_time").set_index("event_time")

    features = pd.DataFrame(index=df.index)
    features["issue_time"] = df["issue_time"]
    features["price_eur_mwh"] = df["price_eur_mwh"]

    for lag_h in LAG_HOURS:
        lagged = df[["issue_time", "price_eur_mwh"]].copy()
        lagged.index = lagged.index + pd.Timedelta(hours=lag_h)
        lagged = lagged.rename(
            columns={"price_eur_mwh": f"price_lag_{lag_h}h", "issue_time": "_lag_issue_time"}
        )
        features = features.join(lagged, how="left")
        leaked = features["_lag_issue_time"] > features["issue_time"]
        if leaked.any():
            raise ValueError(f"look-ahead leak building price_lag_{lag_h}h")
        features = features.drop(columns=["_lag_issue_time"])

    se_holidays = holidays.country_holidays("SE")
    event_index = pd.DatetimeIndex(features.index)
    features["hour_of_day"] = event_index.hour
    features["day_of_week"] = event_index.dayofweek
    features["month"] = event_index.month
    features["is_weekend"] = event_index.dayofweek >= 5
    features["is_holiday"] = [d in se_holidays for d in event_index.date]

    return features.reset_index().rename(columns={"index": "event_time"})


__all__ = ["LAG_HOURS", "build_day_ahead_features"]
