from datetime import date

import pandas as pd

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.features.run import build_all_features, build_secondary_features
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table


def _make_config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2025, 1, 1), end=date(2025, 3, 1))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _fcr_rows(iso_rows: list[str], zone: str) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    products = (("FCRD", "up", 5.0), ("FCRD", "down", 4.0), ("FCRN", "symmetric", 6.0))
    for product, direction, price in products:
        rows.extend(
            {
                "start_time_utc": t,
                "price": price,
                "bidding_zone": zone,
                "reserve_product": product,
                "reserve_direction": direction,
            }
            for t in iso_rows
        )
    return rows


def _afrr_mfrr_rows(iso_rows: list[str], zone: str, reserve_product: str) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    for direction, price in (("up", 3.0), ("down", 3.5)):
        rows.extend(
            {
                "start_time_utc": t,
                "price": price,
                "bidding_zone": zone,
                "reserve_product": reserve_product,
                "reserve_direction": direction,
            }
            for t in iso_rows
        )
    return rows


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
        write_table(conn, "raw_svk_fcr_capacity", _fcr_rows(iso_rows, config.zone))
        write_table(
            conn,
            "raw_svk_afrr_mfrr_capacity",
            _afrr_mfrr_rows(iso_rows, config.zone, "aFRRCapacityMarket"),
        )
        write_table(
            conn,
            "raw_svk_mfrr_capacity",
            _afrr_mfrr_rows(iso_rows, config.zone, "mFRRCapacityMarket"),
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


def test_build_secondary_features_writes_one_table_per_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    _seed(config)
    build_all_facts(config)
    results = build_secondary_features(config)

    tables = {r.table for r in results}
    assert tables == {
        "feature_fcr_d_up",
        "feature_fcr_d_down",
        "feature_fcr_n",
        "feature_afrr_up",
        "feature_afrr_down",
        "feature_mfrr_up",
        "feature_mfrr_down",
        "feature_imbalance",
    }
    assert all(r.row_count > 0 for r in results)


def test_secondary_features_never_use_a_lag_source_issued_after_forecast_issue_time(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    _seed(config)
    build_all_facts(config)
    build_secondary_features(config)

    conn = get_connection(config.duckdb_path)
    try:
        feature_df = conn.execute("SELECT * FROM feature_imbalance").fetchdf()
        fact_df = conn.execute(
            "SELECT event_time, issue_time FROM fact_imbalance_price WHERE price_type = 'final'"
        ).fetchdf()
    finally:
        conn.close()

    lookup = fact_df.set_index("event_time")["issue_time"]
    lag_columns = (
        (24, "imbalance_price_eur_mwh_lag_24h"),
        (168, "imbalance_price_eur_mwh_lag_168h"),
    )
    for lag_h, col in lag_columns:
        available = feature_df[feature_df[col].notna()]
        source_event = available["event_time"] - pd.Timedelta(hours=lag_h)
        source_issue_time = lookup.reindex(source_event).to_numpy()
        assert (source_issue_time <= available["issue_time"].to_numpy()).all()


def test_secondary_imbalance_feature_issue_time_is_forecast_cutoff_not_settlement_time(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _make_config(tmp_path)
    _seed(config)
    build_all_facts(config)
    build_secondary_features(config)

    conn = get_connection(config.duckdb_path)
    try:
        feature_df = conn.execute("SELECT event_time, issue_time FROM feature_imbalance").fetchdf()
    finally:
        conn.close()

    assert (feature_df["issue_time"] == feature_df["event_time"] - pd.Timedelta(minutes=60)).all()
