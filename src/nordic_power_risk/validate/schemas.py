"""Pandera schemas for raw_* ingest tables.

Every raw table is a single series keyed on `timestamp`, with one float value
column (name varies by source). Bounds are injected at call time from the
frozen primary window (config.yaml, T08).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# table -> its value column name
RAW_TABLE_VALUE_COLUMNS: dict[str, str] = {
    "raw_entsoe_day_ahead_price": "price_eur_mwh",
    "raw_esett_imbalance_price": "imbalance_price_eur_mwh",
    "raw_svk_fcr_capacity": "price",
    "raw_svk_afrr_mfrr_capacity": "price",
    "raw_svk_mfrr_capacity": "price",
    "raw_smhi_observations": "value",
}

# table -> its timestamp column name, for tables that don't use the "timestamp" default.
RAW_TABLE_TIMESTAMP_COLUMNS: dict[str, str] = {
    "raw_svk_fcr_capacity": "start_time_utc",
    "raw_svk_afrr_mfrr_capacity": "start_time_utc",
    "raw_svk_mfrr_capacity": "start_time_utc",
}

# Tables with more than one row per timestamp (e.g. multiple reserve products/
# directions/bidding zones per interval) can't enforce timestamp uniqueness.
RAW_TABLES_WITH_NON_UNIQUE_TIMESTAMP: set[str] = {
    "raw_svk_fcr_capacity",
    "raw_svk_afrr_mfrr_capacity",
    "raw_svk_mfrr_capacity",
}


def build_schema(table: str, window_start: date, window_end: date) -> DataFrameSchema:
    """Schema for `table`: typed value column, non-null, unique+in-window timestamp."""
    value_column = RAW_TABLE_VALUE_COLUMNS[table]
    timestamp_column = RAW_TABLE_TIMESTAMP_COLUMNS.get(table, "timestamp")
    is_unique = table not in RAW_TABLES_WITH_NON_UNIQUE_TIMESTAMP
    lower = pd.Timestamp(window_start)
    upper = pd.Timestamp(window_end) + pd.Timedelta(days=1)
    return DataFrameSchema(
        {
            timestamp_column: Column(
                "datetime64[ns]",
                nullable=False,
                unique=is_unique,
                checks=Check.in_range(lower, upper, include_max=False),
            ),
            value_column: Column(float, nullable=False, coerce=False),
        },
        strict=False,
        unique_column_names=True,
    )


__all__ = ["RAW_TABLE_VALUE_COLUMNS", "build_schema"]
