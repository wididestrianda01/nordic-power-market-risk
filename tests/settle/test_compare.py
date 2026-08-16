from datetime import date, datetime, timedelta

import pytest

from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.settle.compare import compare_policies


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "compare.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(terminal_value_eur_mwh=0.0),
    )


def _seed(config: PipelineConfig) -> None:
    # One Stockholm delivery day, four hourly prices with a wide spread.
    day = datetime(2025, 1, 1, 23)  # 23:00 UTC = 00:00 Stockholm next day
    prices = [0.0, 10.0, 90.0, 100.0]
    rows = [
        {"event_time": day + timedelta(hours=i), "price_eur_mwh": price}
        for i, price in enumerate(prices)
    ]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "fact_day_ahead_price", rows)
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
    finally:
        conn.close()


def test_compare_policies_orders_paper_above_no_trade_and_heuristic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    result = compare_policies(config)

    assert result.table == "comparison"
    eff = config.optimizer.one_way_efficiency**2
    assert result.policies["no_trade"] == pytest.approx(0.0)
    # heuristic: buy 0, sell 100 -> eff*100 - 0 - 2*15
    assert result.policies["heuristic"] == pytest.approx(eff * 100.0 - 30.0)
    # perfect foresight: buy [0,10], sell [90,100] -> eff*190 - 10 - 4*15
    assert result.policies["perfect_foresight"] == pytest.approx(eff * 190.0 - 70.0)
    assert result.policies["perfect_foresight"] >= result.policies["heuristic"]


def test_perfect_foresight_stays_above_optimized_with_reserve_revenue(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "settlement",
            [
                {
                    "delivery_time": datetime(2025, 1, 1, 0),
                    "component": "reserve_capacity",
                    "value_eur": 500.0,
                },
                {
                    "delivery_time": datetime(2025, 1, 1, 0),
                    "component": "reserve_activation",
                    "value_eur": 300.0,
                },
            ],
            columns={
                "delivery_time": "TIMESTAMP",
                "component": "VARCHAR",
                "value_eur": "DOUBLE",
            },
        )
    finally:
        conn.close()

    result = compare_policies(config)

    # Optimized books the realized reserve; perfect foresight must be an upper
    # bound on top of it, not an energy-only figure that dips below it.
    assert result.policies["optimized"] == pytest.approx(800.0)
    assert result.policies["perfect_foresight"] > result.policies["optimized"]
