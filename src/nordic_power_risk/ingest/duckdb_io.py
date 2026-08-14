"""DuckDB storage helpers shared by every source client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


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
