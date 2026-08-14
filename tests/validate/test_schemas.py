from datetime import date

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from nordic_power_risk.validate.schemas import (
    RAW_TABLE_TIMESTAMP_COLUMNS,
    RAW_TABLE_VALUE_COLUMNS,
    RAW_TABLES_WITH_NON_UNIQUE_TIMESTAMP,
    build_schema,
)

WINDOW_START = date(2020, 1, 1)
WINDOW_END = date(2020, 1, 31)


def _timestamp_column(table: str) -> str:
    return RAW_TABLE_TIMESTAMP_COLUMNS.get(table, "timestamp")


def _good_df(table: str, value_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            _timestamp_column(table): pd.to_datetime(
                ["2020-01-05T00:00:00", "2020-01-06T00:00:00"]
            ),
            value_column: [1.0, 2.0],
        }
    )


@pytest.mark.parametrize("table,value_column", RAW_TABLE_VALUE_COLUMNS.items())
def test_valid_rows_pass(table: str, value_column: str) -> None:
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    schema.validate(_good_df(table, value_column), lazy=True)


@pytest.mark.parametrize("table,value_column", RAW_TABLE_VALUE_COLUMNS.items())
def test_null_value_fails(table: str, value_column: str) -> None:
    df = _good_df(table, value_column)
    df.loc[0, value_column] = None
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


@pytest.mark.parametrize(
    "table,value_column",
    [
        (t, v)
        for t, v in RAW_TABLE_VALUE_COLUMNS.items()
        if t not in RAW_TABLES_WITH_NON_UNIQUE_TIMESTAMP
    ],
)
def test_duplicate_timestamp_fails(table: str, value_column: str) -> None:
    df = _good_df(table, value_column)
    timestamp_column = _timestamp_column(table)
    df.loc[1, timestamp_column] = df.loc[0, timestamp_column]
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


@pytest.mark.parametrize(
    "table,value_column",
    [
        (t, v)
        for t, v in RAW_TABLE_VALUE_COLUMNS.items()
        if t in RAW_TABLES_WITH_NON_UNIQUE_TIMESTAMP
    ],
)
def test_duplicate_timestamp_allowed_for_multi_row_tables(table: str, value_column: str) -> None:
    df = _good_df(table, value_column)
    timestamp_column = _timestamp_column(table)
    df.loc[1, timestamp_column] = df.loc[0, timestamp_column]
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    schema.validate(df, lazy=True)


@pytest.mark.parametrize("table,value_column", RAW_TABLE_VALUE_COLUMNS.items())
def test_out_of_window_timestamp_fails(table: str, value_column: str) -> None:
    df = _good_df(table, value_column)
    df.loc[0, _timestamp_column(table)] = pd.Timestamp("2019-01-01T00:00:00")
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


@pytest.mark.parametrize("table,value_column", RAW_TABLE_VALUE_COLUMNS.items())
def test_wrong_value_type_fails(table: str, value_column: str) -> None:
    df = _good_df(table, value_column)
    df[value_column] = df[value_column].astype(object)
    df.loc[0, value_column] = "not-a-number"
    schema = build_schema(table, WINDOW_START, WINDOW_END)
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)
