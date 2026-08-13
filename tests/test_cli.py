from datetime import date

from typer.testing import CliRunner

from nordic_power_risk import cli
from nordic_power_risk.cli import app
from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table

runner = CliRunner()


_VALID_ROWS = {
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


def _seeded_config(tmp_path):
    config = PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 31))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )
    conn = get_connection(config.duckdb_path)
    try:
        for table, rows in _VALID_ROWS.items():
            write_table(conn, table, rows)
    finally:
        conn.close()
    return config


def test_validate_passes_on_clean_data(tmp_path, monkeypatch):
    config = _seeded_config(tmp_path)
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0
    assert "raw_entsoe_day_ahead_price: PASS" in result.output


def test_validate_fails_loud_on_duplicate_key(tmp_path, monkeypatch):
    config = _seeded_config(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "raw_entsoe_day_ahead_price",
            [
                {"timestamp": "2020-01-05T00:00:00", "price_eur_mwh": 1.0},
                {"timestamp": "2020-01-05T00:00:00", "price_eur_mwh": 2.0},
            ],
        )
    finally:
        conn.close()
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 1
    assert "raw_entsoe_day_ahead_price: FAIL" in result.output


def test_validate_fails_loud_on_null_value(tmp_path, monkeypatch):
    config = _seeded_config(tmp_path)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "raw_esett_imbalance_price",
            [
                {"timestamp": "2020-01-05T00:00:00", "imbalance_price_eur_mwh": None},
                {"timestamp": "2020-01-06T00:00:00", "imbalance_price_eur_mwh": 2.0},
            ],
        )
    finally:
        conn.close()
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 1
    assert "raw_esett_imbalance_price: FAIL" in result.output


def test_ingest_without_token_exits_nonzero(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    from nordic_power_risk import config as config_module

    config_module.get_settings.cache_clear()
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "ENTSOE_API_TOKEN" in result.output
