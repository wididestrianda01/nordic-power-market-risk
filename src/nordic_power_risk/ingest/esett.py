"""eSett Open Data: 15-minute single imbalance price. No auth.

https://opendata.esett.com/ — terms state data is "public for everyone"; no
formal open-licence grant, so treat as attribution-only, no redistribution claim.

The single imbalance price (Nordic single-price regime since Nov 2021) is served
by EXP14/Prices, keyed by metering-grid-area EIC code (not the short zone name)
and UTC datetimes. The API returns a sentinel ``-1.0`` where no imbalance price
was published for an interval; those intervals are dropped so settlement fails
closed rather than fabricating a price.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import requests

from nordic_power_risk.ingest.entsoe import ZONE_EIC

BASE_URL = "https://api.opendata.esett.com/EXP14/Prices"

# eSett publishes no imbalance price for some intervals, using -1.0 as a sentinel.
MISSING_PRICE_SENTINEL = -1.0


def _utc_datetime(value: date) -> str:
    """Render a naive window boundary as an eSett UTC datetime (end-exclusive)."""
    return f"{value.isoformat()}T00:00:00.000Z"


def fetch_imbalance_prices(zone: str, start: date, end: date, *, timeout: float = 30.0) -> bytes:
    """Raw JSON response for the single imbalance price over [start, end)."""
    params = {
        "start": _utc_datetime(start),
        "end": _utc_datetime(end),
        "mba": ZONE_EIC[zone],
    }
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_imbalance_prices(raw: bytes) -> list[dict[str, Any]]:
    """JSON rows -> [{"timestamp": naive-UTC iso str, "imbalance_price_eur_mwh": float}]."""
    payload = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for item in payload:
        price = float(item["imblPurchasePrice"])
        if price == MISSING_PRICE_SENTINEL:
            continue
        rows.append(
            {
                "timestamp": item["timestampUTC"].rstrip("Z"),
                "imbalance_price_eur_mwh": price,
            }
        )
    return rows
