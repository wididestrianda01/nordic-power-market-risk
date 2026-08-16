"""DuckDB storage helpers shared by every source client."""

from __future__ import annotations

from datetime import datetime
from math import nan
from pathlib import Path
from typing import Any

import duckdb

from nordic_power_risk.config import PipelineConfig


def get_connection(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def write_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[dict[str, Any]],
    columns: dict[str, str] | None = None,
) -> int:
    """Replace `table` with `rows`. Raw ingest tables are always full-refresh.

    `columns` (name -> DuckDB type) fixes the schema when `rows` is empty, so
    an empty result still exposes the columns downstream queries expect.
    """
    if not rows:
        col_defs = (
            ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
            if columns
            else "empty BOOLEAN"
        )
        conn.execute(f"CREATE OR REPLACE TABLE {table} ({col_defs})")
        return 0
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT unnest(?, recursive := true)", [rows])
    return len(rows)


def coerce_datetime(value: object) -> datetime:
    """Return `value` as a naive datetime, accepting ISO-8601 strings."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def coerce_float(value: Any) -> float:
    """Return `value` as a float; a missing (None) cell coerces to NaN, not zero."""
    return nan if value is None else float(value)


def fetch_scalar(
    conn: duckdb.DuckDBPyConnection, query: str, params: list[Any] | None = None
) -> Any:
    """Return the first column of the first row, or None if the query returns no rows."""
    row = conn.execute(query, params or []).fetchone()
    return row[0] if row is not None else None


def read_table(config: PipelineConfig, table: str) -> list[dict[str, Any]]:
    """Read every row of `table` as a list of column-keyed dicts."""
    conn = get_connection(config.duckdb_path)
    try:
        cursor = conn.execute(f"SELECT * FROM {table}")
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]
    finally:
        conn.close()
