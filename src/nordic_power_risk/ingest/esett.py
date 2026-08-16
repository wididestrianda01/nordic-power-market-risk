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

from nordic_power_risk.ingest.entsoe import ZONE_EIC, chunk_date_range

BASE_URL = "https://api.opendata.esett.com/EXP14/Prices"

# eSett publishes no imbalance price for some intervals, using -1.0 as a sentinel.
MISSING_PRICE_SENTINEL = -1.0

# Single-price imbalance regime began 1 Nov 2021; EXP14 holds no single imbalance
# price before that date. EXP14 also caps rows per request (~100k), so a 15-minute
# series longer than ~2.8 years overflows one call; fetch in 1-year chunks.
SINGLE_PRICE_START = date(2021, 11, 1)
ESETT_CHUNK_DAYS = 365


def _utc_datetime(value: date) -> str:
    """Render a naive window boundary as an eSett UTC datetime (end-exclusive)."""
    return f"{value.isoformat()}T00:00:00.000Z"


def fetch_imbalance_prices(
    zone: str, start: date, end: date, *, timeout: float = 30.0
) -> list[bytes]:
    """Raw JSON responses for the single imbalance price over [start, end), in 1-year chunks."""
    raw_chunks: list[bytes] = []
    for chunk_start, chunk_end in chunk_date_range(start, end, max_days=ESETT_CHUNK_DAYS):
        params = {
            "start": _utc_datetime(chunk_start),
            "end": _utc_datetime(chunk_end),
            "mba": ZONE_EIC[zone],
        }
        response = requests.get(BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        raw_chunks.append(response.content)
    return raw_chunks

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
