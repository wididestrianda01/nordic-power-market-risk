from datetime import date, datetime
from pathlib import Path

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.facts.asof import as_of
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
    "raw_svk_fcr_capacity": [
        {
            "start_time_utc": "2024-01-15T00:00:00",
            "price": 5.0,
            "reserve_product": "FCRD",
            "reserve_direction": "up",
            "bidding_zone": "SE3",
        },
    ],
    "raw_svk_afrr_mfrr_capacity": [
        {"start_time_utc": "2024-01-15T00:00:00", "price": 3.0},
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


def _seed_and_build(tmp_path: Path) -> PipelineConfig:
    config = _make_config(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        for table, rows in RAW_ROWS.items():
            write_table(conn, table, rows)
    finally:
        conn.close()
    build_all_facts(config)
    return config


def test_as_of_excludes_rows_issued_after_cutoff(tmp_path: Path) -> None:
    config = _seed_and_build(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        # day-ahead issue_time for both rows is 2024-01-14T09:00 UTC (10:00 CET D-1).
        before = as_of(conn, "fact_day_ahead_price", datetime(2024, 1, 14, 8, 59, 0))
        after = as_of(conn, "fact_day_ahead_price", datetime(2024, 1, 14, 9, 0, 0))
    finally:
        conn.close()

    assert len(before) == 0
    assert len(after) == 2
    assert (after["issue_time"] <= datetime(2024, 1, 14, 9, 0, 0)).all()


def test_as_of_imbalance_swap_at_t_plus_45(tmp_path: Path) -> None:
    config = _seed_and_build(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        # event_time 2024-01-15T00:00: estimated issues at 00:30, final at 00:45.
        only_estimated = as_of(conn, "fact_imbalance_price", datetime(2024, 1, 15, 0, 30, 0))
        both = as_of(conn, "fact_imbalance_price", datetime(2024, 1, 15, 0, 45, 0))
    finally:
        conn.close()

    assert sorted(only_estimated["price_type"]) == ["estimated"]
    assert sorted(both["price_type"]) == ["estimated", "final"]
