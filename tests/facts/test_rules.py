from datetime import date, datetime

from nordic_power_risk.facts.rules import (
    afrr_mfrr_capacity_issue_time,
    day_ahead_issue_time,
    delivery_day,
    delivery_day_hour_count,
    fcr_capacity_issue_time,
    imbalance_forecast_issue_time,
)


def test_day_ahead_issue_time_winter_cet() -> None:
    event_time = datetime(2024, 1, 15, 0, 0, 0)  # 01:00 CET local, delivery date 2024-01-15
    assert day_ahead_issue_time(event_time) == datetime(2024, 1, 14, 9, 0, 0)


def test_day_ahead_issue_time_summer_cest_dst() -> None:
    event_time = datetime(2024, 7, 15, 0, 0, 0)  # 02:00 CEST local, delivery date 2024-07-15
    assert day_ahead_issue_time(event_time) == datetime(2024, 7, 14, 8, 0, 0)


def test_day_ahead_issue_time_near_local_midnight() -> None:
    # 23:30 UTC on Jan 14 is already 00:30 CET on Jan 15 local -> D-1 gate is Jan 14.
    event_time = datetime(2024, 1, 14, 23, 30, 0)
    assert day_ahead_issue_time(event_time) == datetime(2024, 1, 14, 9, 0, 0)


def test_fcr_capacity_issue_time_winter_cet() -> None:
    event_time = datetime(2024, 1, 15, 0, 0, 0)
    assert fcr_capacity_issue_time(event_time) == datetime(2024, 1, 14, 16, 30, 0)


def test_afrr_mfrr_capacity_issue_time_winter_cet() -> None:
    event_time = datetime(2024, 1, 15, 0, 0, 0)
    assert afrr_mfrr_capacity_issue_time(event_time) == datetime(2024, 1, 14, 6, 0, 0)


def test_all_svk_issue_times_precede_event_time() -> None:
    event_time = datetime(2024, 6, 1, 12, 0, 0)
    assert day_ahead_issue_time(event_time) < event_time
    assert fcr_capacity_issue_time(event_time) < event_time


def test_imbalance_forecast_issue_time_is_60min_before_event() -> None:
    event_time = datetime(2024, 6, 1, 12, 0, 0)
    assert imbalance_forecast_issue_time(event_time) == datetime(2024, 6, 1, 11, 0, 0)


def test_imbalance_forecast_issue_time_precedes_event_time() -> None:
    event_time = datetime(2024, 6, 1, 12, 0, 0)
    assert afrr_mfrr_capacity_issue_time(event_time) < event_time


def test_delivery_day_winter_cet() -> None:
    assert delivery_day(datetime(2024, 1, 15, 0, 0, 0)) == date(2024, 1, 15)


def test_delivery_day_summer_cest_spans_utc_midnight() -> None:
    # 22:00 UTC on a CEST day is already 00:00 Stockholm on the next calendar day.
    assert delivery_day(datetime(2024, 7, 14, 22, 0, 0)) == date(2024, 7, 15)
    assert delivery_day(datetime(2024, 7, 14, 23, 0, 0)) == date(2024, 7, 15)


def test_delivery_day_hour_count_dst() -> None:
    assert delivery_day_hour_count(date(2024, 3, 31)) == 23  # spring forward
    assert delivery_day_hour_count(date(2024, 7, 1)) == 24
    assert delivery_day_hour_count(date(2024, 10, 27)) == 25  # fall back
