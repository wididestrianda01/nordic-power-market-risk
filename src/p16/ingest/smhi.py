"""SMHI Open Data: lagged weather observations. No auth.

https://opendata-download-metobs.smhi.se/ — station-based meteorological
observations, used as a lagged (not forecast) feature source.
"""

from __future__ import annotations

import json
from typing import Any

import requests

BASE_URL = "https://opendata-download-metobs.smhi.se/api/version/latest"


def fetch_observations(
    parameter: int, station: int, period: str = "latest-months", *, timeout: float = 30.0
) -> bytes:
    """Raw JSON response for one SMHI parameter/station/period combination."""
    url = f"{BASE_URL}/parameter/{parameter}/station/{station}/period/{period}/data.json"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_observations(raw: bytes) -> list[dict[str, Any]]:
    """JSON "value" rows -> [{"timestamp": iso str (ms epoch UTC), "value": float}]."""
    payload = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for item in payload.get("value", []):
        rows.append({"timestamp": item["date"], "value": float(item["value"])})
    return rows
