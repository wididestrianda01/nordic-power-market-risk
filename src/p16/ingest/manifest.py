"""Source manifest: what was pulled, from where, under what licence, when."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel


class ManifestEntry(BaseModel):
    name: str
    licence: str
    coverage_start: date
    coverage_end: date
    endpoint: str
    pulled_at: datetime
    checksum: str
    row_count: int


def checksum_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_entry(
    *,
    name: str,
    licence: str,
    coverage_start: date,
    coverage_end: date,
    endpoint: str,
    raw: bytes,
    row_count: int,
) -> ManifestEntry:
    return ManifestEntry(
        name=name,
        licence=licence,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        endpoint=endpoint,
        pulled_at=datetime.now(UTC),
        checksum=checksum_of(raw),
        row_count=row_count,
    )


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(e.model_dump_json()) for e in entries]
    path.write_text(json.dumps(payload, indent=2) + "\n")
