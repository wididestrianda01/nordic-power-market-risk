"""ENTSO-E Transparency Platform: day-ahead prices (document A44) and balancing data.

https://transparency.entsoe.eu/ — free token-gated REST API. Licence forbids
bulk redistribution; nothing pulled here is committed to the repo.
"""

from __future__ import annotations

import io
import zipfile
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


MAX_CHUNK_DAYS = 365


def chunk_date_range(
    start: date, end: date, *, max_days: int = MAX_CHUNK_DAYS
) -> list[tuple[date, date]]:
    """Split [start, end) into consecutive [chunk_start, chunk_end) spans of at most max_days."""
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=max_days), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


BALANCING_CHUNK_DAYS = 14


def _fetch_chunks(
    token: str, start: date, end: date, fixed: dict[str, str], *, timeout: float
) -> list[bytes]:
    """Fetch an ENTSO-E document over [start, end) in two-week chunks.

    Balancing documents (A84/A81/A86) return far more data than day-ahead A44
    (15-minute prices and volumes across several products); a single request over
    the full window read-times-out, so split into <=14-day chunks.
    """
    raw_chunks = []
    for chunk_start, chunk_end in chunk_date_range(start, end, max_days=BALANCING_CHUNK_DAYS):
        params = {
            "securityToken": token,
            "periodStart": chunk_start.strftime("%Y%m%d0000"),
            "periodEnd": chunk_end.strftime("%Y%m%d0000"),
            **fixed,
        }
        response = requests.get(BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        raw_chunks.append(response.content)
    return raw_chunks


def fetch_day_ahead_prices(
    token: str, zone: str, start: date, end: date, *, timeout: float = 30.0
) -> list[bytes]:
    """Raw XML responses for document A44 (day-ahead prices) over [start, end).

    ENTSO-E rejects A44 requests spanning more than ~1 year, so [start, end) is
    split into <=1-year chunks and fetched with one request per chunk.
    """
    eic = ZONE_EIC[zone]
    raw_chunks = []
    for chunk_start, chunk_end in chunk_date_range(start, end):
        params = {
            "securityToken": token,
            "documentType": "A44",
            "in_Domain": eic,
            "out_Domain": eic,
            "periodStart": chunk_start.strftime("%Y%m%d0000"),
            "periodEnd": chunk_end.strftime("%Y%m%d0000"),
        }
        response = requests.get(BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        raw_chunks.append(response.content)
    return raw_chunks


def parse_day_ahead_prices(raw: bytes) -> list[dict[str, Any]]:
    """XML Point/price.amount rows -> [{"timestamp": iso str, "price_eur_mwh": float}].

    Day-ahead moved to 15-minute resolution on 2025-10-01 (SDAC 15-min MTU), so
    PT15M series are aggregated to the hourly spine (arithmetic mean of the four
    quarter-hour prices); PT60M series pass through unchanged.
    """
    root = ET.fromstring(raw)
    rows: list[dict[str, Any]] = []
    for series in root.findall("ns:TimeSeries", NAMESPACE):
        period = series.find("ns:Period", NAMESPACE)
        if period is None:
            continue
        start_str = period.findtext("ns:timeInterval/ns:start", namespaces=NAMESPACE)
        resolution = period.findtext("ns:resolution", namespaces=NAMESPACE)
        if start_str is None or resolution not in {"PT60M", "PT15M"}:
            continue
        period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")
        if resolution == "PT60M":
            for point in period.findall("ns:Point", NAMESPACE):
                position = int(point.findtext("ns:position", namespaces=NAMESPACE, default="0"))
                price = float(
                    point.findtext("ns:price.amount", namespaces=NAMESPACE, default="nan")
                )
                timestamp = period_start + timedelta(hours=position - 1)
                rows.append({"timestamp": timestamp.isoformat(), "price_eur_mwh": price})
        else:
            hourly: dict[datetime, list[float]] = {}
            for point in period.findall("ns:Point", NAMESPACE):
                position = int(point.findtext("ns:position", namespaces=NAMESPACE, default="0"))
                price = float(
                    point.findtext("ns:price.amount", namespaces=NAMESPACE, default="nan")
                )
                timestamp = period_start + timedelta(hours=(position - 1) // 4)
                hourly.setdefault(timestamp, []).append(price)
            for timestamp, prices in hourly.items():
                rows.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "price_eur_mwh": sum(prices) / len(prices),
                    }
                )
    return rows


# ENTSO-E process types for balancing reserves. Values verified against the
# Transparency API guide / entsoe-py mappings: A51 = aFRR, A52 = FCR, A47 = mFRR.
ACTIVATION_PROCESS_TYPES = {
    "FCR_N": "A52",  # Frequency Containment Reserve
    "AFRR": "A51",  # Automatic Frequency Restoration Reserve
    "MFRR": "A47",  # Manual Frequency Restoration Reserve
}

# businessType codes carried inside balancing documents (A84 activated prices).
_BUSINESS_TYPE_PRODUCT = {
    "A95": "FCR_N",  # Frequency containment reserve
    "A96": "AFRR",  # Automatic frequency restoration reserve
    "A97": "MFRR",  # Manual frequency restoration reserve
}

_DIRECTION_CODES = {"A01": "up", "A02": "down", "A03": "symmetric"}


def _direction_name(code: str | None) -> str:
    return _DIRECTION_CODES.get(code or "A01", "up")


def fetch_activated_energy_prices(
    token: str, zone: str, start: date, end: date, *, timeout: float = 120.0
) -> list[bytes]:
    """Raw XML chunks for document A84 (activated balancing energy prices)."""
    return _fetch_chunks(
        token,
        start,
        end,
        {
            "documentType": "A84",
            "processType": "A16",  # realised
            "controlArea_Domain": ZONE_EIC[zone],
        },
        timeout=timeout,
    )


def parse_activated_energy_prices(raw: bytes) -> list[dict[str, Any]]:
    """A84 XML -> [{"timestamp", "product", "direction", "activation_price_eur_mwh"}].

    Each TimeSeries carries its own businessType (A95/A96/A97) and flowDirection;
    prices are 15-minute `activation_Price.amount` points (curveType A03).
    """
    root = ET.fromstring(raw)
    rows: list[dict[str, Any]] = []
    for series in root.findall("{*}TimeSeries"):
        product = _BUSINESS_TYPE_PRODUCT.get(series.findtext("{*}businessType", default="") or "")
        if product is None:
            continue
        direction = _direction_name(series.findtext("{*}flowDirection.direction", default="A01"))
        period = series.find("{*}Period")
        if period is None:
            continue
        start_str = period.findtext("{*}timeInterval/{*}start")
        resolution = period.findtext("{*}resolution")
        if start_str is None or resolution not in ("PT60M", "PT15M"):
            continue
        period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")
        step = timedelta(minutes=60 if resolution == "PT60M" else 15)
        for point in period.findall("{*}Point"):
            position = int(point.findtext("{*}position", default="0"))
            price = float(point.findtext("{*}activation_Price.amount", default="nan"))
            timestamp = period_start + step * (position - 1)
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "product": product,
                    "direction": direction,
                    "activation_price_eur_mwh": price,
                }
            )
    return rows


def fetch_procured_volume(
    token: str, zone: str, process_type: str, start: date, end: date, *, timeout: float = 120.0
) -> list[bytes]:
    """Raw ZIP chunks for document A81 (contracted reserves: procured volume + price)."""
    return _fetch_chunks(
        token,
        start,
        end,
        {
            "documentType": "A81",
            "businessType": "B95",  # procured capacity
            "processType": process_type,
            "controlArea_Domain": ZONE_EIC[zone],
            "type_MarketAgreement.Type": "A01",  # daily
        },
        timeout=timeout,
    )


def parse_procured_volume(raw: bytes, product: str) -> list[dict[str, Any]]:
    """A81 response -> [{"timestamp", "product", "direction", "procured_mw"}].

    aFRR/mFRR return a ZIP archive of XML; FCR returns plain XML (or an empty
    acknowledgement with no TimeSeries, which parses to no rows).
    """
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            xml_docs = [archive.read(n) for n in archive.namelist() if n.endswith(".xml")]
    else:
        xml_docs = [raw]

    rows: list[dict[str, Any]] = []
    for xml in xml_docs:
        root = ET.fromstring(xml)
        for series in root.findall("{*}TimeSeries"):
            direction = _direction_name(
                series.findtext("{*}flowDirection.direction", default="A01")
            )
            for period in series.findall("{*}Period"):
                start_str = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution")
                if start_str is None or resolution != "PT60M":
                    continue
                period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")
                for point in period.findall("{*}Point"):
                    position = int(point.findtext("{*}position", default="0"))
                    volume = float(point.findtext("{*}quantity", default="nan"))
                    timestamp = period_start + timedelta(hours=position - 1)
                    rows.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "product": product,
                            "direction": direction,
                            "procured_mw": volume,
                        }
                    )
    return rows


def fetch_imbalance_volumes(
    token: str, zone: str, start: date, end: date, *, timeout: float = 120.0
) -> list[bytes]:
    """Raw ZIP chunks for document A86 (imbalance volume = mFRR activation volume)."""
    return _fetch_chunks(
        token,
        start,
        end,
        {
            "documentType": "A86",
            "controlArea_Domain": ZONE_EIC[zone],
        },
        timeout=timeout,
    )


def parse_imbalance_volumes(raw: bytes) -> list[dict[str, Any]]:
    """A86 ZIP -> [{"timestamp", "product", "direction", "activated_mw"}].

    A86 "imbalance volume" is the control area's mFRR activation volume, published
    at 15 minutes in up (A01) and down (A02) TimeSeries (businessType A19). Each
    point is MWh; aggregated to hourly MWh (numerically the hourly average MW) to
    match the hourly dispatch spine.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml_docs = [archive.read(n) for n in archive.namelist() if n.endswith(".xml")]

    up: dict[datetime, float] = {}
    down: dict[datetime, float] = {}
    for xml in xml_docs:
        root = ET.fromstring(xml)
        for series in root.findall("{*}TimeSeries"):
            direction = _direction_name(
                series.findtext("{*}flowDirection.direction", default="A01")
            )
            for period in series.findall("{*}Period"):
                start_str = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution")
                if start_str is None or resolution != "PT15M":
                    continue
                period_start = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ")
                for point in period.findall("{*}Point"):
                    position = int(point.findtext("{*}position", default="0"))
                    quantity = float(point.findtext("{*}quantity", default="nan"))
                    timestamp = period_start + timedelta(minutes=15 * (position - 1))
                    (up if direction == "up" else down)[timestamp] = quantity

    hourly_up: dict[datetime, float] = {}
    hourly_down: dict[datetime, float] = {}
    for timestamp, quantity in up.items():
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        hourly_up[hour] = hourly_up.get(hour, 0.0) + quantity
    for timestamp, quantity in down.items():
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        hourly_down[hour] = hourly_down.get(hour, 0.0) + quantity

    rows: list[dict[str, Any]] = []
    for hour in sorted(hourly_up):
        rows.append(
            {
                "timestamp": hour.isoformat(),
                "product": "MFRR",
                "direction": "up",
                "activated_mw": hourly_up[hour],
            }
        )
    for hour in sorted(hourly_down):
        rows.append(
            {
                "timestamp": hour.isoformat(),
                "product": "MFRR",
                "direction": "down",
                "activated_mw": hourly_down[hour],
            }
        )
    return rows
