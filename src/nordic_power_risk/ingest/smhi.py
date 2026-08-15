"""SMHI Open Data: lagged weather observations. No auth.

https://opendata-download-metobs.smhi.se/ — station-based meteorological
observations, used as a lagged (not forecast) feature source.

The corrected-archive period is the quality-controlled historical series (it
excludes the most recent ~3 months, which are only available via the
``latest-*`` periods). It is served as a semicolon-delimited CSV, unlike the
JSON used by the near-real-time periods.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

import requests

BASE_URL = "https://opendata-download-metobs.smhi.se/api/version/latest"


def fetch_observations(parameter: int, station: int, *, timeout: float = 30.0) -> bytes:
    """Raw CSV for one SMHI parameter/station over the corrected historical archive."""
    url = (
        f"{BASE_URL}/parameter/{parameter}/station/{station}"
        "/period/corrected-archive/data.csv"
    )
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_observations(raw: bytes) -> list[dict[str, Any]]:
    """CSV rows -> [{"timestamp": ms epoch (UTC), "value": float}]."""
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";")
    rows: list[dict[str, Any]] = []
    for record in reader:
        if len(record) < 3:
            continue
        date_str, time_str, value_str = record[0], record[1], record[2]
        try:
            value = float(value_str)
        except ValueError:
            continue  # header/metadata line
        try:
            moment = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=UTC)
        except ValueError:
            continue  # period-metadata line, not a data row
        rows.append({"timestamp": int(moment.timestamp() * 1000), "value": value})
    return rows
