from datetime import date, datetime
from pathlib import Path

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table

RAW_ROWS: dict[str, list[dict[str, object]]] = {
    "raw_entsoe_day_ahead_price": [
        {"timestamp": "2024-01-15T00:00:00", "price_eur_mwh": 10.0},
        {"timestamp": "2024-01-15T01:00:00", "price_eur_mwh": 12.0},
    ],
    "raw_esett_imbalance_price": [
        {"timestamp": "2024-01-15T00:00:00", "imbalance_price_eur_mwh": 15.0},
    ],
    "raw_svk_day_ahead_price": [
        {"timestamp": "2024-01-15T00:00:00", "value": 10.0},
    ],
    "raw_svk_fcr_capacity": [
        {
            "start_time_utc": "2024-01-15T00:00:00",
            "price": 5.0,
            "reserve_product": "FCRD",
            "reserve_direction": "up",
            "bidding_zone": "SE3",
        },
        {
            "start_time_utc": "2024-01-15T00:00:00",
            "price": 6.0,
            "reserve_product": "FCRD",
            "reserve_direction": "down",
            "bidding_zone": "SE3",
        },
        {
            "start_time_utc": "2024-01-15T00:00:00",
            "price": 7.0,
            "reserve_product": "FCRN",
            "reserve_direction": "symmetric",
            "bidding_zone": "SE3",
        },
        {
            "start_time_utc": "2024-01-15T00:00:00",
            "price": 99.0,
            "reserve_product": "FCRD",
            "reserve_direction": "up",
            "bidding_zone": "SE4",
        },
    ],
    "raw_svk_afrr_mfrr_capacity": [
        {"start_time_utc": "2024-01-15T00:00:00", "price": 3.0},
    ],
    "raw_activation": [
        {
            "timestamp": "2024-01-15T00:00:00",
            "product": "FCR_N",
            "direction": "up",
            "activated_mw": 0.4,
        },
        {
            "timestamp": "2024-01-15T00:00:00",
            "product": "AFRR",
            "direction": "down",
            "activated_mw": 0.6,
        },
    ],
    "raw_smhi_observations": [
        {"timestamp": 1705276800000, "value": -2.5},  # 2024-01-15T00:00:00
    ],
}


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2026, 12, 31))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed_raw_tables(config: PipelineConfig) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        for table, rows in RAW_ROWS.items():
            write_table(conn, table, rows)
    finally:
        conn.close()


def test_build_all_facts_writes_every_fact_table(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_raw_tables(config)

    results = build_all_facts(config)

    tables = {result.table: result.row_count for result in results}
    assert tables == {
        "fact_day_ahead_price": 2,
        "fact_svk_day_ahead_price": 1,
        "fact_svk_fcr_d_up": 1,
        "fact_svk_fcr_d_down": 1,
        "fact_svk_fcr_n": 1,
        "fact_svk_afrr_mfrr_capacity": 1,
        "fact_imbalance_price": 2,  # estimated + final per raw row
        "fact_smhi_observations": 1,
        "fact_activation": 2,
    }


def test_day_ahead_issue_time_lands_on_d_minus_1_10_00_cet(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_raw_tables(config)
    build_all_facts(config)

    conn = get_connection(config.duckdb_path)
    try:
        row = conn.execute(
            "SELECT event_time, issue_time FROM fact_day_ahead_price WHERE event_time = ?",
            [datetime(2024, 1, 15, 0, 0, 0)],
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    event_time, issue_time = row
    # 2024-01-15T00:00 UTC delivery -> D-1 = 2024-01-14, 10:00 CET local = 09:00 UTC.
    assert issue_time == datetime(2024, 1, 14, 9, 0, 0)
    assert issue_time < event_time


def test_imbalance_price_tags_estimated_and_final(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_raw_tables(config)
    build_all_facts(config)

    conn = get_connection(config.duckdb_path)
    try:
        rows = conn.execute(
            "SELECT price_type, issue_time FROM fact_imbalance_price ORDER BY price_type"
        ).fetchall()
    finally:
        conn.close()

    by_type = dict(rows)
    assert by_type["estimated"] == datetime(2024, 1, 15, 0, 30, 0)
    assert by_type["final"] == datetime(2024, 1, 15, 0, 45, 0)


def test_smhi_issue_time_equals_event_time(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_raw_tables(config)
    build_all_facts(config)

    conn = get_connection(config.duckdb_path)
    try:
        row = conn.execute("SELECT event_time, issue_time FROM fact_smhi_observations").fetchone()
    finally:
        conn.close()

    assert row is not None
    event_time, issue_time = row
    assert event_time == issue_time
