"""Svenska kraftnat (SvK) Data Service / Mimer: CKAN datastore API.

https://data.svk.se — historical day-ahead + FCR/aFRR/mFRR capacity series,
CC BY 4.0. SvK stopped updating day-ahead 1 July 2026 (historical-only).
"""

from __future__ import annotations

import json
from typing import Any

import requests

BASE_URL = "https://data.svk.se/api/3/action/datastore_search"

# CKAN resource_id per series (T03 research corrected: the two capacity
# resource_ids were swapped — verified against live CKAN field/records).
RESOURCE_IDS = {
    "day_ahead_price": "0c56e30d-8fce-4c27-afc8-621c230ae34d",
    "fcr_capacity": "72ef5ec0-d0d7-4d22-95e9-4f22b3048af4",
    "afrr_mfrr_capacity": "6351d2cc-1657-43eb-b112-b8408c700529",
}


def fetch_resource(series: str, *, limit: int = 100_000, timeout: float = 30.0) -> bytes:
    """Raw JSON response from the CKAN datastore_search endpoint for `series`."""
    resource_id = RESOURCE_IDS[series]
    params: dict[str, str | int] = {"resource_id": resource_id, "limit": limit}
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_resource(raw: bytes) -> list[dict[str, Any]]:
    """CKAN datastore records -> list of row dicts (schema is series-specific)."""
    payload = json.loads(raw)
    result = payload.get("result", {})
    records = result.get("records", [])
    return [dict(record) for record in records]


# Activated balancing energy (60-min, discontinued 2025-03-04) — the only public
# aggregate activation VOLUME source for Swedish bidding zones. mFRR is absent
# (never published by SvK); aFRR, FCR-N and FCR-D activation are here.
ACTIVATION_RESOURCE_IDS = {
    "afrr": "f40313df-c68c-420a-a655-1ea8ffda2859",
    "fcr": "c0a75ee4-2cf3-43f2-9168-2c8627aef0ef",
}

# SvK reserve_product -> canonical product code used across the pipeline.
_ACTIVATION_PRODUCTS = {
    "aFRRCapacityMarket": "AFRR",
    "FCRN": "FCR_N",
    "FCRD": "FCR_D",
}


def fetch_activated_energy(series: str, *, limit: int = 100_000, timeout: float = 30.0) -> bytes:
    """Raw JSON for SvK activated balancing energy (`afrr` or `fcr`)."""
    resource_id = ACTIVATION_RESOURCE_IDS[series]
    params: dict[str, str | int] = {"resource_id": resource_id, "limit": limit}
    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_activated_energy(raw: bytes, zone: str) -> list[dict[str, Any]]:
    """SvK activated-energy records -> raw_activation rows for `zone`.

    `volume` is signed MWh over a 60-min interval (up > 0, down < 0); we store the
    magnitude as `activated_mw` (numerically equal to average MW over that hour)
    and keep `direction` for the sign.
    """
    payload = json.loads(raw)
    records = payload.get("result", {}).get("records", [])
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("bidding_zone") != zone:
            continue
        product = _ACTIVATION_PRODUCTS.get(rec.get("reserve_product"))
        if product is None:
            continue
        rows.append(
            {
                "timestamp": rec["start_time_utc"],
                "product": product,
                "direction": rec["reserve_direction"],
                "activated_mw": abs(float(rec["volume"])),
            }
        )
    return rows
