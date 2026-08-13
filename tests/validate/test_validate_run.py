from datetime import date
from pathlib import Path

import pytest

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.validate.run import validate_all

VALID_ROWS: dict[str, list[dict[str, object]]] = {
    "raw_entsoe_day_ahead_price": [
        {"timestamp": "2020-01-05T00:00:00", "price_eur_mwh": 1.0},
        {"timestamp": "2020-01-06T00:00:00", "price_eur_mwh": 2.0},
    ],
    "raw_esett_imbalance_price": [
        {"timestamp": "2020-01-05T00:00:00", "imbalance_price_eur_mwh": 1.0},
        {"timestamp": "2020-01-06T00:00:00", "imbalance_price_eur_mwh": 2.0},
    ],
    "raw_svk_day_ahead_price": [
        {"timestamp": "2020-01-05T00:00:00", "value": 1.0},
        {"timestamp": "2020-01-06T00:00:00", "value": 2.0},
    ],
    "raw_svk_fcr_capacity": [
        {"timestamp": "2020-01-05T00:00:00", "value": 1.0},
        {"timestamp": "2020-01-06T00:00:00", "value": 2.0},
    ],
    "raw_svk_afrr_mfrr_capacity": [
        {"timestamp": "2020-01-05T00:00:00", "value": 1.0},
        {"timestamp": "2020-01-06T00:00:00", "value": 2.0},
    ],
    "raw_smhi_observations": [
        {"timestamp": 1578182400000, "value": 1.0},
        {"timestamp": 1578268800000, "value": 2.0},
    ],
}


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 31))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed_valid_tables(config: PipelineConfig) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        for table, rows in VALID_ROWS.items():
            write_table(conn, table, rows)
    finally:
        conn.close()


def test_validate_all_passes_on_clean_data(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_valid_tables(config)

    results = validate_all(config)

    assert len(results) == len(VALID_ROWS)
    assert all(result.passed for result in results)
    assert all(result.failure_cases is None for result in results)


def test_validate_all_fails_on_duplicate_key(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _seed_valid_tables(config)

    conn = get_connection(config.duckdb_path)
    try:
        duplicated = [
            {"timestamp": "2020-01-05T00:00:00", "price_eur_mwh": 1.0},
            {"timestamp": "2020-01-05T00:00:00", "price_eur_mwh": 2.0},
        ]
        write_table(conn, "raw_entsoe_day_ahead_price", duplicated)
    finally:
        conn.close()

    results = validate_all(config)

    by_table = {result.table: result for result in results}
    assert by_table["raw_entsoe_day_ahead_price"].passed is False
    assert by_table["raw_entsoe_day_ahead_price"].failure_cases is not None
    assert by_table["raw_svk_day_ahead_price"].passed is True


def test_validate_all_rejects_unsupported_zone(tmp_path: Path) -> None:
    config = _make_config(tmp_path).model_copy(update={"zone": "SE4"})

    with pytest.raises(ValueError, match="SE3"):
        validate_all(config)
