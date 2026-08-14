"""Build feature_day_ahead from fact_day_ahead_price (Phase 2 ticket 01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.features.build import build_day_ahead_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table


@dataclass(frozen=True)
class FeatureBuildResult:
    table: str
    row_count: int


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


__all__ = ["FeatureBuildResult", "build_all_features"]
