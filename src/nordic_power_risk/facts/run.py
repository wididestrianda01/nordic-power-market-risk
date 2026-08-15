"""Build event_time/issue_time fact tables from the validated raw_* layer (T02/T03)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.facts.rules import (
    ACTIVATION_PUBLICATION_LAG,
    IMBALANCE_ESTIMATED_LAG,
    IMBALANCE_FINAL_LAG,
    afrr_mfrr_capacity_issue_time,
    day_ahead_issue_time,
    fcr_capacity_issue_time,
    reserve_volume_issue_time,
)
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.validate.schemas import RAW_TABLE_TIMESTAMP_COLUMNS

# smhi stores epoch-millisecond ints; every other raw table stores ISO 8601 strings.
_EPOCH_MS_TABLES = {"raw_smhi_observations"}


@dataclass(frozen=True)
class FactBuildResult:
    table: str
    row_count: int


def _read_raw(conn: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    df = conn.execute(f"SELECT * FROM {table}").fetchdf()
    source_column = RAW_TABLE_TIMESTAMP_COLUMNS.get(table, "timestamp")
    if table in _EPOCH_MS_TABLES:
        df["timestamp"] = pd.to_datetime(df[source_column], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df[source_column])
    return df


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0]
    return count > 0


def _price_rows(
    df: pd.DataFrame, value_column: str, issue_time_fn: Callable[[datetime], datetime]
) -> list[dict[str, Any]]:
    event_times = df["timestamp"].dt.to_pydatetime()
    values = df[value_column].tolist()
    return [
        {"event_time": event_time, "issue_time": issue_time_fn(event_time), value_column: value}
        for event_time, value in zip(event_times, values, strict=True)
    ]


# FCR raw rows mix product (FCRD/FCRN), direction (up/down/symmetric), and zone in
# one table; each (product, direction) pair is its own T09 secondary target.
_FCR_TARGETS = [
    ("fact_svk_fcr_d_up", "FCRD", "up"),
    ("fact_svk_fcr_d_down", "FCRD", "down"),
    ("fact_svk_fcr_n", "FCRN", "symmetric"),
]


def _fcr_rows(
    df: pd.DataFrame, zone: str, reserve_product: str, reserve_direction: str
) -> list[dict[str, Any]]:
    subset = df[
        (df["bidding_zone"] == zone)
        & (df["reserve_product"] == reserve_product)
        & (df["reserve_direction"] == reserve_direction)
    ]
    return _price_rows(subset, "price", fcr_capacity_issue_time)


def _imbalance_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    event_times = df["timestamp"].dt.to_pydatetime()
    values = df["imbalance_price_eur_mwh"].tolist()
    rows: list[dict[str, Any]] = []
    for event_time, value in zip(event_times, values, strict=True):
        rows.append(
            {
                "event_time": event_time,
                "issue_time": event_time + IMBALANCE_ESTIMATED_LAG,
                "imbalance_price_eur_mwh": value,
                "price_type": "estimated",
            }
        )
        rows.append(
            {
                "event_time": event_time,
                "issue_time": event_time + IMBALANCE_FINAL_LAG,
                "imbalance_price_eur_mwh": value,
                "price_type": "final",
            }
        )
    return rows


def _smhi_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    event_times = df["timestamp"].dt.to_pydatetime()
    values = df["value"].tolist()
    return [
        {"event_time": event_time, "issue_time": event_time, "value": value}
        for event_time, value in zip(event_times, values, strict=True)
    ]


def _activation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate reserve activation volumes -> fact rows keyed by product/direction."""
    event_times = df["timestamp"].dt.to_pydatetime()
    products = df["product"].tolist()
    directions = df["direction"].tolist()
    values = df["activated_mw"].tolist()
    return [
        {
            "event_time": event_time,
            "issue_time": event_time + ACTIVATION_PUBLICATION_LAG,
            "product": str(product),
            "direction": str(direction),
            "activated_mw": float(value),
        }
        for event_time, product, direction, value in zip(
            event_times, products, directions, values, strict=True
        )
    ]


def _activation_price_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """A84 activated balancing energy prices -> fact rows keyed by product/direction."""
    event_times = df["timestamp"].dt.to_pydatetime()
    products = df["product"].tolist()
    directions = df["direction"].tolist()
    values = df["activation_price_eur_mwh"].tolist()
    return [
        {
            "event_time": event_time,
            "issue_time": event_time + ACTIVATION_PUBLICATION_LAG,
            "product": str(product),
            "direction": str(direction),
            "activation_price_eur_mwh": float(value),
        }
        for event_time, product, direction, value in zip(
            event_times, products, directions, values, strict=True
        )
    ]


def _reserve_volume_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """A75 procured reserve volumes -> fact rows keyed by product/direction."""
    event_times = df["timestamp"].dt.to_pydatetime()
    products = df["product"].tolist()
    directions = df["direction"].tolist()
    values = df["procured_mw"].tolist()
    return [
        {
            "event_time": event_time,
            "issue_time": reserve_volume_issue_time(str(product), event_time),
            "product": str(product),
            "direction": str(direction),
            "procured_mw": float(value),
        }
        for event_time, product, direction, value in zip(
            event_times, products, directions, values, strict=True
        )
    ]


def build_all_facts(config: PipelineConfig) -> list[FactBuildResult]:
    conn = get_connection(config.duckdb_path)
    results: list[FactBuildResult] = []
    try:
        source_specs = [
            (
                "raw_entsoe_day_ahead_price",
                "fact_day_ahead_price",
                "price_eur_mwh",
                day_ahead_issue_time,
            ),
            (
                "raw_svk_afrr_mfrr_capacity",
                "fact_svk_afrr_mfrr_capacity",
                "price",
                afrr_mfrr_capacity_issue_time,
            ),
        ]
        for raw_table, fact_table, value_column, issue_time_fn in source_specs:
            df = _read_raw(conn, raw_table)
            rows = _price_rows(df, value_column, issue_time_fn)
            price_columns = {
                "event_time": "TIMESTAMP",
                "issue_time": "TIMESTAMP",
                value_column: "DOUBLE",
            }
            count = write_table(conn, fact_table, rows, columns=price_columns)
            results.append(FactBuildResult(table=fact_table, row_count=count))

        fcr_df = _read_raw(conn, "raw_svk_fcr_capacity")
        fcr_columns = {"event_time": "TIMESTAMP", "issue_time": "TIMESTAMP", "price": "DOUBLE"}
        for fact_table, reserve_product, reserve_direction in _FCR_TARGETS:
            rows = _fcr_rows(fcr_df, config.zone, reserve_product, reserve_direction)
            count = write_table(conn, fact_table, rows, columns=fcr_columns)
            results.append(FactBuildResult(table=fact_table, row_count=count))

        imbalance_df = _read_raw(conn, "raw_esett_imbalance_price")
        imbalance_rows = _imbalance_rows(imbalance_df)
        imbalance_columns = {
            "event_time": "TIMESTAMP",
            "issue_time": "TIMESTAMP",
            "imbalance_price_eur_mwh": "DOUBLE",
            "price_type": "VARCHAR",
        }
        count = write_table(conn, "fact_imbalance_price", imbalance_rows, columns=imbalance_columns)
        results.append(FactBuildResult(table="fact_imbalance_price", row_count=count))

        smhi_df = _read_raw(conn, "raw_smhi_observations")
        smhi_rows = _smhi_rows(smhi_df)
        smhi_columns = {"event_time": "TIMESTAMP", "issue_time": "TIMESTAMP", "value": "DOUBLE"}
        count = write_table(conn, "fact_smhi_observations", smhi_rows, columns=smhi_columns)
        results.append(FactBuildResult(table="fact_smhi_observations", row_count=count))

        activation_rows: list[dict[str, Any]] = []
        if _table_exists(conn, "raw_activation"):
            activation_df = _read_raw(conn, "raw_activation")
            activation_rows = _activation_rows(activation_df)
        activation_columns = {
            "event_time": "TIMESTAMP",
            "issue_time": "TIMESTAMP",
            "product": "VARCHAR",
            "direction": "VARCHAR",
            "activated_mw": "DOUBLE",
        }
        count = write_table(conn, "fact_activation", activation_rows, columns=activation_columns)
        results.append(FactBuildResult(table="fact_activation", row_count=count))

        activation_price_rows: list[dict[str, Any]] = []
        if _table_exists(conn, "raw_activation_price"):
            activation_price_df = _read_raw(conn, "raw_activation_price")
            activation_price_rows = _activation_price_rows(activation_price_df)
        activation_price_columns = {
            "event_time": "TIMESTAMP",
            "issue_time": "TIMESTAMP",
            "product": "VARCHAR",
            "direction": "VARCHAR",
            "activation_price_eur_mwh": "DOUBLE",
        }
        count = write_table(
            conn, "fact_activation_price", activation_price_rows, columns=activation_price_columns
        )
        results.append(FactBuildResult(table="fact_activation_price", row_count=count))

        reserve_volume_rows: list[dict[str, Any]] = []
        if _table_exists(conn, "raw_reserve_volume"):
            reserve_volume_df = _read_raw(conn, "raw_reserve_volume")
            reserve_volume_rows = _reserve_volume_rows(reserve_volume_df)
        reserve_volume_columns = {
            "event_time": "TIMESTAMP",
            "issue_time": "TIMESTAMP",
            "product": "VARCHAR",
            "direction": "VARCHAR",
            "procured_mw": "DOUBLE",
        }
        count = write_table(
            conn, "fact_reserve_volume", reserve_volume_rows, columns=reserve_volume_columns
        )
        results.append(FactBuildResult(table="fact_reserve_volume", row_count=count))
    finally:
        conn.close()
    return results


__all__ = ["FactBuildResult", "build_all_facts"]
