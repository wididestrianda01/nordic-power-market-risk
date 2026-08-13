"""Per-source issue_time derivation rules (T02/T03 publication cutoffs, frozen in spec.md).

Gate-closure-relative cutoffs are defined in Europe/Stockholm local time (CET/CEST,
DST-aware) and converted to naive UTC to match the raw-table timestamp convention.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

_STOCKHOLM = ZoneInfo("Europe/Stockholm")
_UTC = ZoneInfo("UTC")

# Imbalance settlement price is published twice per delivery interval.
IMBALANCE_ESTIMATED_LAG = timedelta(minutes=30)
IMBALANCE_FINAL_LAG = timedelta(minutes=45)


def _local_cutoff(event_time_utc: datetime, cutoff: time) -> datetime:
    """`cutoff` local Stockholm time, day before event_time's local delivery date, as naive UTC."""
    event_local = event_time_utc.replace(tzinfo=_UTC).astimezone(_STOCKHOLM)
    issue_date = event_local.date() - timedelta(days=1)
    issue_local = datetime.combine(issue_date, cutoff, tzinfo=_STOCKHOLM)
    return issue_local.astimezone(_UTC).replace(tzinfo=None)


def day_ahead_issue_time(event_time_utc: datetime) -> datetime:
    """SDAC/SvK day-ahead: issues 10:00 CET/CEST on D-1 (spec.md, 14-38h horizon)."""
    return _local_cutoff(event_time_utc, time(10, 0))


def fcr_capacity_issue_time(event_time_utc: datetime) -> datetime:
    """FCR-N/FCR-D: two D-1 auction gates (00:30, 18:00), each issues 30min before close.

    Per SvK's own auction design, the two gates are NOT a per-delivery-hour split: both
    auctions clear volume for the same 24 delivery hours of day D. The first (00:30) gate
    clears the majority; the second (18:00) gate clears/adjusts the residual for the same
    hours. So a delivery hour's capacity result isn't final until the later gate closes,
    regardless of which hour it is -- using the later cutoff (17:30 issue) for every hour
    is therefore correct, not a simplification of a two-branch rule.
    """
    return _local_cutoff(event_time_utc, time(17, 30))


def afrr_mfrr_capacity_issue_time(event_time_utc: datetime) -> datetime:
    """aFRR/mFRR capacity: single D-1 07:30 gate, issues 30min before close."""
    return _local_cutoff(event_time_utc, time(7, 0))


__all__ = [
    "IMBALANCE_ESTIMATED_LAG",
    "IMBALANCE_FINAL_LAG",
    "afrr_mfrr_capacity_issue_time",
    "day_ahead_issue_time",
    "fcr_capacity_issue_time",
]
