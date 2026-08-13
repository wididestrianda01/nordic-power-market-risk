"""eSett Open Data: 15-minute imbalance prices. No auth.

https://opendata.esett.com/ — terms state data is "public for everyone"; no
formal open-licence grant, so treat as attribution-only, no redistribution claim.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

BASE_URL = "https://api.opendata.esett.com/EXP16/PriceSingle"


def fetch_imbalance_prices(zone: str, start: date, end: date, *, timeout: float = 30.0) -> bytes:
    """Raw JSON response for single imbalance price over [start, end]."""
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "area": zone,
    }
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_imbalance_prices(raw: bytes) -> list[dict[str, Any]]:
    """JSON rows -> [{"timestamp": iso str, "imbalance_price_eur_mwh": float}]."""
    import json

    payload = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for item in payload:
        rows.append(
            {
                "timestamp": item["timestamp"],
                "imbalance_price_eur_mwh": float(item["value"]),
            }
        )
    return rows
