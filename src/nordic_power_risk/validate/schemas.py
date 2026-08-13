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
    "raw_svk_day_ahead_price": "value",
    "raw_svk_fcr_capacity": "value",
    "raw_svk_afrr_mfrr_capacity": "value",
    "raw_smhi_observations": "value",
}


def build_schema(table: str, window_start: date, window_end: date) -> DataFrameSchema:
    """Schema for `table`: typed value column, non-null, unique+in-window timestamp."""
    value_column = RAW_TABLE_VALUE_COLUMNS[table]
    lower = pd.Timestamp(window_start)
    upper = pd.Timestamp(window_end) + pd.Timedelta(days=1)
    return DataFrameSchema(
        {
            "timestamp": Column(
                "datetime64[ns]",
                nullable=False,
                unique=True,
                checks=Check.in_range(lower, upper, include_max=False),
            ),
            value_column: Column(float, nullable=False, coerce=False),
        },
        strict=False,
        unique_column_names=True,
    )


__all__ = ["RAW_TABLE_VALUE_COLUMNS", "build_schema"]
