"""Build feature_day_ahead from fact_day_ahead_price (Phase 2 ticket 01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.facts.rules import imbalance_forecast_issue_time
from nordic_power_risk.features.build import build_day_ahead_features, build_lag_calendar_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table


@dataclass(frozen=True)
class FeatureBuildResult:
    table: str
    row_count: int


# Secondary targets (T09 ticket 04): FCR-N/D reuse their fact table's own gate-closure
# issue_time as-is; imbalance overrides it to the T-60min forecast decision cutoff.
_SECONDARY_TARGETS = [
    ("fact_svk_fcr_d_up", "feature_fcr_d_up", "price", None),
    ("fact_svk_fcr_d_down", "feature_fcr_d_down", "price", None),
    ("fact_svk_fcr_n", "feature_fcr_n", "price", None),
    (
        "fact_imbalance_price",
        "feature_imbalance",
        "imbalance_price_eur_mwh",
        imbalance_forecast_issue_time,
    ),
]


def build_all_features(config: PipelineConfig) -> FeatureBuildResult:
    conn = get_connection(config.duckdb_path)
    try:
        price_df = conn.execute(
            "SELECT event_time, issue_time, price_eur_mwh FROM fact_day_ahead_price"
        ).fetchdf()
        features = build_day_ahead_features(price_df)
        rows = cast(list[dict[str, Any]], features.to_dict("records"))
        count = write_table(conn, "feature_day_ahead", rows)
        return FeatureBuildResult(table="feature_day_ahead", row_count=count)
    finally:
        conn.close()


def build_secondary_features(config: PipelineConfig) -> list[FeatureBuildResult]:
    conn = get_connection(config.duckdb_path)
    results: list[FeatureBuildResult] = []
    try:
        for fact_table, feature_table, value_column, issue_time_fn in _SECONDARY_TARGETS:
            where = " WHERE price_type = 'final'" if fact_table == "fact_imbalance_price" else ""
            df = conn.execute(
                f"SELECT event_time, issue_time, {value_column} FROM {fact_table}{where}"
            ).fetchdf()
            features = build_lag_calendar_features(df, value_column, issue_time_fn=issue_time_fn)
            rows = cast(list[dict[str, Any]], features.to_dict("records"))
            count = write_table(conn, feature_table, rows)
            results.append(FeatureBuildResult(table=feature_table, row_count=count))
    finally:
        conn.close()
    return results


__all__ = ["FeatureBuildResult", "build_all_features", "build_secondary_features"]
