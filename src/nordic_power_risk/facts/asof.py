"""As-of query helper: enforce the issue_time <= as_of_time look-ahead cutoff (T03)."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd


def as_of(conn: duckdb.DuckDBPyConnection, table: str, as_of_time: datetime) -> pd.DataFrame:
    """Return rows from a fact_* table as they would have been known at as_of_time."""
    return conn.execute(f"SELECT * FROM {table} WHERE issue_time <= ?", [as_of_time]).fetchdf()


__all__ = ["as_of"]
