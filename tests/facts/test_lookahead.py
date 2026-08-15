"""Phase 1 exit criterion: no as_of() query may return data issued after as_of_time.

Samples as_of_time across the frozen T08 primary spine (config.yaml, 2019-01-01
to 2026-06-30) and asserts the cutoff holds for every fact_* table.
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nordic_power_risk.config import PipelineConfig, Window, get_config
from nordic_power_risk.facts.asof import as_of
from nordic_power_risk.facts.run import build_all_facts
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table

FACT_TABLES = [
    "fact_day_ahead_price",
    "fact_svk_fcr_d_up",
    "fact_svk_fcr_d_down",
    "fact_svk_fcr_n",
    "fact_svk_afrr_up",
    "fact_svk_afrr_down",
    "fact_svk_mfrr_up",
    "fact_svk_mfrr_down",
    "fact_imbalance_price",
    "fact_smhi_observations",
    "fact_activation",
]


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2026, 12, 31))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed_across_spine(config: PipelineConfig, event_times: pd.DatetimeIndex) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        iso_rows = [t.isoformat() for t in event_times]
        write_table(
            conn,
            "raw_entsoe_day_ahead_price",
            [{"timestamp": t, "price_eur_mwh": 10.0} for t in iso_rows],
        )
        imbalance_rows = [{"timestamp": t, "imbalance_price_eur_mwh": 15.0} for t in iso_rows]
        write_table(conn, "raw_esett_imbalance_price", imbalance_rows)
        write_table(
            conn,
            "raw_svk_fcr_capacity",
            [
                {
                    "start_time_utc": t,
                    "price": 5.0,
                    "reserve_product": "FCRD",
                    "reserve_direction": "up",
                    "bidding_zone": "SE3",
                }
                for t in iso_rows
            ],
        )
        write_table(
            conn,
            "raw_svk_afrr_mfrr_capacity",
            [
                {
                    "start_time_utc": t,
                    "price": 3.0,
                    "bidding_zone": "SE3",
                    "reserve_product": "aFRRCapacityMarket",
                    "reserve_direction": "up",
                }
                for t in iso_rows
            ],
        )
        write_table(
            conn,
            "raw_svk_mfrr_capacity",
            [
                {
                    "start_time_utc": t,
                    "price": 4.0,
                    "bidding_zone": "SE3",
                    "reserve_product": "mFRRCapacityMarket",
                    "reserve_direction": "up",
                }
                for t in iso_rows
            ],
        )
        write_table(
            conn,
            "raw_activation",
            [
                {"timestamp": t, "product": "FCR_N", "direction": "up", "activated_mw": 0.5}
                for t in iso_rows
            ],
        )
        write_table(
            conn,
            "raw_smhi_observations",
            [{"timestamp": int(t.timestamp() * 1000), "value": -2.5} for t in event_times],
        )
    finally:
        conn.close()


def test_as_of_never_leaks_future_data_across_primary_spine(tmp_path: Path) -> None:
    primary = get_config().windows["primary"]
    spine = pd.date_range(primary.start, primary.end, periods=6)

    config = _make_config(tmp_path)
    _seed_across_spine(config, spine)
    build_all_facts(config)

    as_of_samples = pd.date_range(primary.start, primary.end, periods=9)

    conn = get_connection(config.duckdb_path)
    try:
        for as_of_time in as_of_samples:
            for table in FACT_TABLES:
                rows = as_of(conn, table, as_of_time.to_pydatetime())
                if rows.empty:
                    continue
                assert (rows["issue_time"] <= as_of_time.to_pydatetime()).all(), (
                    f"{table} leaked a row with issue_time > as_of_time={as_of_time}"
                )
    finally:
        conn.close()


def test_imbalance_estimated_final_swap_resolves_at_t_plus_45(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        ts = "2024-01-15T00:00:00"
        write_table(
            conn, "raw_esett_imbalance_price", [{"timestamp": ts, "imbalance_price_eur_mwh": 15.0}]
        )
        write_table(conn, "raw_entsoe_day_ahead_price", [{"timestamp": ts, "price_eur_mwh": 10.0}])
        write_table(
            conn,
            "raw_svk_fcr_capacity",
            [
                {
                    "start_time_utc": ts,
                    "price": 5.0,
                    "reserve_product": "FCRD",
                    "reserve_direction": "up",
                    "bidding_zone": "SE3",
                }
            ],
        )
        write_table(
            conn,
            "raw_svk_afrr_mfrr_capacity",
            [
                {
                    "start_time_utc": ts,
                    "price": 3.0,
                    "bidding_zone": "SE3",
                    "reserve_product": "aFRRCapacityMarket",
                    "reserve_direction": "up",
                }
            ],
        )
        write_table(
            conn,
            "raw_activation",
            [{"timestamp": ts, "product": "FCR_N", "direction": "up", "activated_mw": 0.5}],
        )
        write_table(conn, "raw_smhi_observations", [{"timestamp": 1705276800000, "value": -2.5}])
    finally:
        conn.close()
    build_all_facts(config)

    conn = get_connection(config.duckdb_path)
    try:
        before_estimated = as_of(conn, "fact_imbalance_price", datetime(2024, 1, 15, 0, 29, 59))
        at_estimated = as_of(conn, "fact_imbalance_price", datetime(2024, 1, 15, 0, 30, 0))
        at_final = as_of(conn, "fact_imbalance_price", datetime(2024, 1, 15, 0, 45, 0))
    finally:
        conn.close()

    assert before_estimated.empty
    assert sorted(at_estimated["price_type"]) == ["estimated"]
    assert sorted(at_final["price_type"]) == ["estimated", "final"]
