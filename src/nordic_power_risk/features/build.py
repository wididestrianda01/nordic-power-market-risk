"""Lagged value + calendar/holiday feature tables (T09), issue-time gated.

ENTSO-E day-ahead load/generation forecasts are named in T09 as an allowed feature
source but this repo has no ingest client for them yet (only day-ahead *price* is
ingested) -- ponytail: descoped, add load/gen features once that ingest source exists.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import holidays
import pandas as pd

LAG_HOURS = (24, 168)


def build_lag_calendar_features(
    df: pd.DataFrame,
    value_column: str,
    *,
    lag_prefix: str | None = None,
    issue_time_fn: Callable[[datetime], datetime] | None = None,
) -> pd.DataFrame:
    """df: columns event_time, issue_time, <value_column>.

    issue_time_fn, if given, overrides each row's own issue_time with a forecast
    decision cutoff (e.g. T-60min for imbalance) -- the look-ahead leak check below
    is run against this (possibly overridden) cutoff, since that's the time the
    forecast is actually issued and thus the correct leakage boundary.
    """
    lag_prefix = value_column if lag_prefix is None else lag_prefix
    frame = df.sort_values("event_time").set_index("event_time")

    features = pd.DataFrame(index=frame.index)
    if issue_time_fn is None:
        features["issue_time"] = frame["issue_time"]
    else:
        event_times = pd.DatetimeIndex(frame.index).to_pydatetime()
        features["issue_time"] = [issue_time_fn(t) for t in event_times]
    features[value_column] = frame[value_column]

    for lag_h in LAG_HOURS:
        lagged = frame[["issue_time", value_column]].copy()
        lagged.index = lagged.index + pd.Timedelta(hours=lag_h)
        lagged = lagged.rename(
            columns={value_column: f"{lag_prefix}_lag_{lag_h}h", "issue_time": "_lag_issue_time"}
        )
        features = features.join(lagged, how="left")
        leaked = features["_lag_issue_time"] > features["issue_time"]
        if leaked.any():
            raise ValueError(f"look-ahead leak building {lag_prefix}_lag_{lag_h}h")
        features = features.drop(columns=["_lag_issue_time"])

    se_holidays = holidays.country_holidays("SE")
    event_index = pd.DatetimeIndex(features.index)
    features["hour_of_day"] = event_index.hour
    features["day_of_week"] = event_index.dayofweek
    features["month"] = event_index.month
    features["is_weekend"] = event_index.dayofweek >= 5
    features["is_holiday"] = [d in se_holidays for d in event_index.date]

    return features.reset_index().rename(columns={"index": "event_time"})


def build_day_ahead_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """price_df: columns event_time, issue_time, price_eur_mwh from fact_day_ahead_price."""
    return build_lag_calendar_features(price_df, "price_eur_mwh", lag_prefix="price")


__all__ = ["LAG_HOURS", "build_day_ahead_features", "build_lag_calendar_features"]
