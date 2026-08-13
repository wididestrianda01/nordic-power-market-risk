"""DuckDB storage helpers shared by every source client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def get_connection(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def write_table(
    conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict[str, Any]]
) -> int:
    """Replace `table` with `rows`. Raw ingest tables are always full-refresh."""
    if not rows:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (empty BOOLEAN)")
        return 0
    conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT unnest(?, recursive := true)", [rows])
    return len(rows)
