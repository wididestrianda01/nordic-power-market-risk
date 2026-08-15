from datetime import date, datetime

import pytest

from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.settle.stress import run_stresses


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "stress.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(terminal_value_eur_mwh=0.0),
    )


def _seed(config: PipelineConfig) -> None:
    delivery = datetime(2025, 1, 1, 23)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "dispatch_energy",
            [
                {
                    "delivery_time": delivery,
                    "duration_hours": 1.0,
                    "charge_mw": 0.0,
                    "discharge_mw": 1.0,
                    "degradation_cost_eur": 1.0,
                }
            ],
        )
        write_table(
            conn,
            "fact_day_ahead_price",
            [{"event_time": delivery, "price_eur_mwh": 50.0}],
        )
        write_table(
            conn,
            "settlement",
            [{"delivery_time": delivery, "component": "day_ahead_revenue", "value_eur": 49.0}],
        )
    finally:
        conn.close()


def test_stresses_report_scenario_deltas(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    result = run_stresses(config)

    assert result.table == "stress"
    assert result.baseline_eur == pytest.approx(49.0)
    assert set(result.scenarios) == {
        "negative_price",
        "price_spike",
        "forecast_outage",
        "reduced_capacity",
        "efficiency_loss",
        "correlated_reserve",
    }
    # flat schedule -> zero P&L, so the delta is the full baseline loss.
    assert result.scenarios["forecast_outage"] == pytest.approx(-49.0)
    # negative price: (1 x -50 - 1) - 49 = -100
    assert result.scenarios["negative_price"] == pytest.approx(-100.0)
