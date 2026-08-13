"""ENTSO-E Transparency Platform: day-ahead prices (document A44).

https://transparency.entsoe.eu/ — free token-gated REST API. Licence forbids
bulk redistribution; nothing pulled here is committed to the repo.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from xml.etree import ElementTree as ET

import requests

BASE_URL = "https://web-api.tp.entsoe.eu/api"
NAMESPACE = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

# ENTSO-E EIC area codes, Swedish bidding zones.
ZONE_EIC = {
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
}


def fetch_day_ahead_prices(
    token: str, zone: str, start: date, end: date, *, timeout: float = 30.0
) -> bytes:
    """Raw XML response for document A44 (day-ahead prices) over [start, end)."""
    eic = ZONE_EIC[zone]
    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": eic,
        "out_Domain": eic,
        "periodStart": start.strftime("%Y%m%d0000"),
        "periodEnd": end.strftime("%Y%m%d0000"),
    }
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_day_ahead_prices(raw: bytes) -> list[dict[str, Any]]:
    """XML Point/price.amount rows -> [{"timestamp": iso str, "price_eur_mwh": float}]."""
    root = ET.fromstring(raw)
    rows: list[dict[str, Any]] = []
    for series in root.findall("ns:TimeSeries", NAMESPACE):
        period = series.find("ns:Period", NAMESPACE)
        if period is None:
            continue
        start_str = period.findtext("ns:timeInterval/ns:start", namespaces=NAMESPACE)
        resolution = period.findtext("ns:resolution", namespaces=NAMESPACE)
        if start_str is None or resolution != "PT60M":
            continue
        period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")
        for point in period.findall("ns:Point", NAMESPACE):
            position = int(point.findtext("ns:position", namespaces=NAMESPACE, default="0"))
            price = float(point.findtext("ns:price.amount", namespaces=NAMESPACE, default="nan"))
            timestamp = period_start + timedelta(hours=position - 1)
            rows.append({"timestamp": timestamp.isoformat(), "price_eur_mwh": price})
    return rows
