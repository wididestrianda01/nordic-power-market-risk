"""Run pandera schemas against every raw_* table; report per-table pass/fail."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandera.errors import SchemaErrors

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.ingest.duckdb_io import get_connection
from nordic_power_risk.validate.schemas import (
    RAW_TABLE_TIMESTAMP_COLUMNS,
    RAW_TABLE_VALUE_COLUMNS,
    build_schema,
)

SUPPORTED_ZONE = "SE3"

# smhi stores epoch-millisecond ints; every other source stores ISO 8601 strings.
_EPOCH_MS_TABLES = {"raw_smhi_observations"}


@dataclass(frozen=True)
class TableValidationResult:
    table: str
    passed: bool
    failure_cases: pd.DataFrame | None = None


def validate_all(config: PipelineConfig) -> list[TableValidationResult]:
    if config.zone != SUPPORTED_ZONE:
        raise ValueError(f"unsupported zone {config.zone!r}; only {SUPPORTED_ZONE} is frozen (T08)")

    window = config.windows["primary"]
    conn = get_connection(config.duckdb_path)
    results: list[TableValidationResult] = []
    try:
        for table in RAW_TABLE_VALUE_COLUMNS:
            df = conn.execute(f"SELECT * FROM {table}").fetchdf()
            source_column = RAW_TABLE_TIMESTAMP_COLUMNS.get(table, "timestamp")
            if table in _EPOCH_MS_TABLES:
                df[source_column] = pd.to_datetime(df[source_column], unit="ms")
            else:
                df[source_column] = pd.to_datetime(df[source_column])
            schema = build_schema(table, window.start, window.end)
            try:
                schema.validate(df, lazy=True)
                results.append(TableValidationResult(table=table, passed=True))
            except SchemaErrors as exc:
                results.append(
                    TableValidationResult(
                        table=table, passed=False, failure_cases=exc.failure_cases
                    )
                )
    finally:
        conn.close()
    return results
