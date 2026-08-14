from datetime import date

import pandas as pd

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.features.run import build_all_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table


def _make_config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2025, 1, 1), end=date(2025, 3, 1))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed(config: PipelineConfig) -> None:
    hours = pd.date_range("2025-01-01", "2025-03-01", freq="h", inclusive="left")
    iso_rows = [t.isoformat() for t in hours]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "raw_entsoe_day_ahead_price",
            [{"timestamp": t, "price_eur_mwh": 10.0} for t in iso_rows],
        )
        write_table(
            conn,
            "raw_esett_imbalance_price",
            [{"timestamp": t, "imbalance_price_eur_mwh": 15.0} for t in iso_rows],
        )
        write_table(
            conn, "raw_svk_day_ahead_price", [{"timestamp": t, "value": 10.0} for t in iso_rows]
        )
        write_table(
            conn, "raw_svk_fcr_capacity", [{"timestamp": t, "value": 5.0} for t in iso_rows]
        )
        write_table(
            conn, "raw_svk_afrr_mfrr_capacity", [{"timestamp": t, "value": 3.0} for t in iso_rows]
        )
        write_table(
            conn,
            "raw_smhi_observations",
            [{"timestamp": int(t.timestamp() * 1000), "value": -2.5} for t in hours],
        )
    finally:
        conn.close()


def test_feature_table_never_uses_a_lag_source_issued_after_its_own_issue_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    _seed(config)
    build_all_facts(config)
    result = build_all_features(config)
    assert result.row_count > 0

    conn = get_connection(config.duckdb_path)
    try:
        feature_df = conn.execute("SELECT * FROM feature_day_ahead").fetchdf()
        price_df = conn.execute("SELECT event_time, issue_time FROM fact_day_ahead_price").fetchdf()
    finally:
        conn.close()

    lookup = price_df.set_index("event_time")["issue_time"]
    for lag_h, col in ((24, "price_lag_24h"), (168, "price_lag_168h")):
        available = feature_df[feature_df[col].notna()]
        source_event = available["event_time"] - pd.Timedelta(hours=lag_h)
        source_issue_time = lookup.reindex(source_event).to_numpy()
        assert (source_issue_time <= available["issue_time"].to_numpy()).all()
