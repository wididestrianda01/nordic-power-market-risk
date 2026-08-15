from datetime import date
from types import SimpleNamespace

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
        {"start_time_utc": "2020-01-05T00:00:00", "price": 1.0},
        {"start_time_utc": "2020-01-06T00:00:00", "price": 2.0},
    ],
    "raw_svk_afrr_mfrr_capacity": [
        {
            "start_time_utc": "2020-01-05T00:00:00",
            "price": 1.0,
            "bidding_zone": "SE3",
            "reserve_product": "aFRRCapacityMarket",
            "reserve_direction": "up",
        },
        {
            "start_time_utc": "2020-01-06T00:00:00",
            "price": 2.0,
            "bidding_zone": "SE3",
            "reserve_product": "aFRRCapacityMarket",
            "reserve_direction": "up",
        },
    ],
    "raw_svk_mfrr_capacity": [
        {
            "start_time_utc": "2020-01-05T00:00:00",
            "price": 1.0,
            "bidding_zone": "SE3",
            "reserve_product": "mFRRCapacityMarket",
            "reserve_direction": "up",
        },
        {
            "start_time_utc": "2020-01-06T00:00:00",
            "price": 2.0,
            "bidding_zone": "SE3",
            "reserve_product": "mFRRCapacityMarket",
            "reserve_direction": "up",
        },
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
    from nordic_power_risk.config import Settings

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(entsoe_api_token=None))
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "ENTSOE_API_TOKEN" in result.output


def test_models_runs_primary_secondary_and_tertiary_forecasts(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    from nordic_power_risk.models import run as primary_module
    from nordic_power_risk.models import secondary_run

    config = object()
    calls: list[str] = []
    primary = SimpleNamespace(
        rung="naive",
        n_obs=2,
        pinball_loss=1.0,
        crps=2.0,
        coverage_80=0.8,
        winkler_80=3.0,
        pit_mean=0.5,
        dm_stat=None,
        dm_pvalue=None,
        dm_stat_vs_seasonal_naive=None,
        dm_pvalue_vs_seasonal_naive=None,
    )
    secondary = SimpleNamespace(
        target="imbalance",
        rung="seasonal_naive",
        n_obs=2,
        pinball_loss=1.5,
        crps=2.5,
        coverage_80=0.8,
        winkler_80=3.5,
        pit_mean=0.5,
        dm_stat=None,
        dm_pvalue=None,
    )
    tertiary = SimpleNamespace(
        target="afrr_up",
        source="seasonal_naive",
        n_obs=2,
        mae=4.0,
    )
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(
        primary_module,
        "run_benchmark_ladder",
        lambda actual: calls.append("primary") or [primary],
    )
    monkeypatch.setattr(primary_module, "select_best_rung", lambda results: primary)
    monkeypatch.setattr(
        secondary_run,
        "run_secondary_benchmark",
        lambda actual: calls.append("secondary") or [secondary],
    )
    monkeypatch.setattr(
        secondary_run,
        "run_tertiary_forecast",
        lambda actual: calls.append("tertiary") or [tertiary],
    )

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert calls == ["primary", "secondary", "tertiary"]
    assert "promoted: naive" in result.output
    assert "imbalance/seasonal_naive" in result.output
    assert "afrr_up/seasonal_naive" in result.output
