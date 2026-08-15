from datetime import date, datetime, timedelta

import pytest

from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.settle.attribution import attribute


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "attribution.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(terminal_value_eur_mwh=0.0),
    )


def _seed(config: PipelineConfig) -> None:
    day = datetime(2025, 1, 1, 23)
    prices = [0.0, 10.0, 90.0, 100.0]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "fact_day_ahead_price",
            [
                {"event_time": day + timedelta(hours=i), "price_eur_mwh": price}
                for i, price in enumerate(prices)
            ],
        )
        write_table(
            conn,
            "settlement",
            [],
            columns={
                "delivery_time": "TIMESTAMP",
                "component": "VARCHAR",
                "value_eur": "DOUBLE",
            },
        )
        dispatch_columns = {"objective_eur": "DOUBLE", "terminal_value_eur": "DOUBLE"}
        write_table(conn, "dispatch_energy", [], columns=dispatch_columns)
        write_table(conn, "dispatch_imbalance", [], columns=dispatch_columns)
        write_table(conn, "dispatch_reserve", [], columns={"capacity_value_eur": "DOUBLE"})
    finally:
        conn.close()


def test_attribution_components_sum_to_gap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    result = attribute(config)

    assert result.table == "attribution"
    assert sum(result.components.values()) == pytest.approx(result.gap_eur)
    # On an empty settlement, the whole gap is constraint cost (residual).
    assert result.components["constraint_cost"] == pytest.approx(result.gap_eur)
    assert result.components["forecast_error"] == pytest.approx(0.0)
