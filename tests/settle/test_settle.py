from datetime import date, datetime

import pytest
from typer.testing import CliRunner

from nordic_power_risk import cli
from nordic_power_risk.cli import app
from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.settle.run import reconcile, run_settlement

runner = CliRunner()


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "settle.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(terminal_value_eur_mwh=0.0),
    )


def _seed_empty_reserve_tables(config: PipelineConfig) -> None:
    """Create empty reserve dispatch/fact tables so settlement reads cleanly."""
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "dispatch_reserve",
            [],
            columns={
                "delivery_time": "TIMESTAMP",
                "duration_hours": "DOUBLE",
                "capacity_mw": "DOUBLE",
                "capacity_value_eur": "DOUBLE",
                "product": "VARCHAR",
                "direction": "VARCHAR",
                "conditional_acceptance": "BOOLEAN",
            },
        )
        for table in (
            "fact_svk_fcr_d_up",
            "fact_svk_fcr_d_down",
            "fact_svk_fcr_n",
            "fact_svk_afrr_up",
            "fact_svk_afrr_down",
            "fact_svk_mfrr_up",
            "fact_svk_mfrr_down",
        ):
            write_table(
                conn,
                table,
                [],
                columns={"event_time": "TIMESTAMP", "price": "DOUBLE"},
            )
        write_table(
            conn,
            "fact_activation",
            [],
            columns={
                "event_time": "TIMESTAMP",
                "product": "VARCHAR",
                "direction": "VARCHAR",
                "activated_mw": "DOUBLE",
            },
        )
        for table, value_column in (
            ("fact_activation_price", "activation_price_eur_mwh"),
            ("fact_reserve_volume", "procured_mw"),
        ):
            write_table(
                conn,
                table,
                [],
                columns={
                    "event_time": "TIMESTAMP",
                    "product": "VARCHAR",
                    "direction": "VARCHAR",
                    value_column: "DOUBLE",
                },
            )
    finally:
        conn.close()


def _seed_energy(config: PipelineConfig) -> None:
    energy = [
        {
            "delivery_time": datetime(2025, 1, 1, 0),
            "duration_hours": 1.0,
            "charge_mw": 1.0,
            "discharge_mw": 0.0,
            "degradation_cost_eur": 1.0,
            "objective_eur": 0.0,
            "terminal_value_eur": 0.0,
        },
        {
            "delivery_time": datetime(2025, 1, 1, 1),
            "duration_hours": 1.0,
            "charge_mw": 0.0,
            "discharge_mw": 1.0,
            "degradation_cost_eur": 1.5,
            "objective_eur": 0.0,
            "terminal_value_eur": 0.0,
        },
    ]
    prices = [
        {"event_time": datetime(2025, 1, 1, 0), "price_eur_mwh": 50.0},
        {"event_time": datetime(2025, 1, 1, 1), "price_eur_mwh": 80.0},
    ]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "dispatch_energy", energy)
        write_table(conn, "fact_day_ahead_price", prices)
        write_table(
            conn,
            "dispatch_imbalance",
            [],
            columns={
                "delivery_time": "TIMESTAMP",
                "duration_hours": "DOUBLE",
                "objective_eur": "DOUBLE",
                "terminal_value_eur": "DOUBLE",
            },
        )
        write_table(
            conn,
            "fact_imbalance_price",
            [],
            columns={"event_time": "TIMESTAMP", "imbalance_price_eur_mwh": "DOUBLE"},
        )
    finally:
        conn.close()
    _seed_empty_reserve_tables(config)


def _seed_imbalance(config: PipelineConfig) -> None:
    imbalance = [
        {
            "delivery_time": datetime(2025, 1, 1, 0),
            "duration_hours": 1.0,
            "imbalance_position_mw": 0.5,
            "degradation_cost_eur": 0.2,
        },
        {
            "delivery_time": datetime(2025, 1, 1, 1),
            "duration_hours": 1.0,
            "imbalance_position_mw": -1.0,
            "degradation_cost_eur": 0.3,
        },
    ]
    prices = [
        {
            "event_time": datetime(2025, 1, 1, 0),
            "imbalance_price_eur_mwh": 60.0,
            "price_type": "final",
        },
        {
            "event_time": datetime(2025, 1, 1, 0),
            "imbalance_price_eur_mwh": 55.0,
            "price_type": "estimated",
        },
        {
            "event_time": datetime(2025, 1, 1, 1),
            "imbalance_price_eur_mwh": 70.0,
            "price_type": "final",
        },
    ]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "dispatch_imbalance", imbalance)
        write_table(conn, "fact_imbalance_price", prices)
        write_table(
            conn,
            "dispatch_energy",
            [],
            columns={"delivery_time": "TIMESTAMP", "duration_hours": "DOUBLE"},
        )
        write_table(
            conn,
            "fact_day_ahead_price",
            [],
            columns={"event_time": "TIMESTAMP", "price_eur_mwh": "DOUBLE"},
        )
    finally:
        conn.close()
    _seed_empty_reserve_tables(config)


def _seed_reserve(config: PipelineConfig) -> None:
    _seed_empty_reserve_tables(config)
    reserve = [
        {
            "delivery_time": datetime(2025, 1, 1, 0),
            "duration_hours": 1.0,
            "capacity_mw": 0.5,
            "product": "FCR_D",
            "direction": "up",
            "conditional_acceptance": True,
        },
        {
            "delivery_time": datetime(2025, 1, 1, 1),
            "duration_hours": 1.0,
            "capacity_mw": 1.0,
            "product": "FCR_N",
            "direction": "symmetric",
            "conditional_acceptance": False,
        },
        {
            "delivery_time": datetime(2025, 1, 1, 2),
            "duration_hours": 1.0,
            "capacity_mw": 0.25,
            "product": "AFRR",
            "direction": "up",
            "conditional_acceptance": True,
        },
    ]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "dispatch_reserve", reserve)
        write_table(
            conn, "fact_svk_fcr_d_up", [{"event_time": datetime(2025, 1, 1, 0), "price": 10.0}]
        )
        write_table(
            conn, "fact_svk_fcr_n", [{"event_time": datetime(2025, 1, 1, 1), "price": 20.0}]
        )
        write_table(
            conn,
            "fact_svk_afrr_up",
            [{"event_time": datetime(2025, 1, 1, 2), "price": 5.0}],
        )
        write_table(
            conn,
            "dispatch_energy",
            [],
            columns={"delivery_time": "TIMESTAMP", "duration_hours": "DOUBLE"},
        )
        write_table(
            conn,
            "fact_day_ahead_price",
            [],
            columns={"event_time": "TIMESTAMP", "price_eur_mwh": "DOUBLE"},
        )
        write_table(
            conn,
            "dispatch_imbalance",
            [],
            columns={"delivery_time": "TIMESTAMP", "duration_hours": "DOUBLE"},
        )
        write_table(
            conn,
            "fact_imbalance_price",
            [],
            columns={"event_time": "TIMESTAMP", "imbalance_price_eur_mwh": "DOUBLE"},
        )
    finally:
        conn.close()


def _components(config: PipelineConfig) -> dict[str, float]:
    conn = get_connection(config.duckdb_path)
    try:
        rows = conn.execute(
            "SELECT component, SUM(value_eur) AS total FROM settlement GROUP BY component"
        ).fetchdf()
    finally:
        conn.close()
    return dict(zip(rows["component"], rows["total"], strict=True))


def test_energy_settlement_reconciles_to_observed_prices(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_energy(config)

    result = run_settlement(config)

    assert result.table == "settlement"
    assert result.row_count == 4  # 2 components x 2 intervals
    assert result.total_pnl_eur == pytest.approx(80.0 - 50.0)

    components = _components(config)
    assert components["day_ahead_revenue"] == pytest.approx(80.0)
    assert components["day_ahead_purchase"] == pytest.approx(-50.0)
    assert "degradation" not in components


def test_missing_observed_price_leaves_interval_unsettled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_energy(config)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "dispatch_energy",
            [
                {
                    "delivery_time": datetime(2025, 1, 1, 2),
                    "duration_hours": 1.0,
                    "charge_mw": 1.0,
                    "discharge_mw": 0.0,
                    "degradation_cost_eur": 0.0,
                }
            ],
        )
    finally:
        conn.close()

    result = run_settlement(config)

    assert result.row_count == 0
    assert result.total_pnl_eur == pytest.approx(0.0)


def test_imbalance_settlement_uses_final_price(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_imbalance(config)

    result = run_settlement(config)

    # interval 1: +0.5 MW x 60.0 = +30.0, interval 2: -1.0 MW x 70.0 = -70.0
    assert result.total_pnl_eur == pytest.approx(30.0 - 70.0 - 0.5)

    components = _components(config)
    assert components["imbalance"] == pytest.approx(-40.0)
    assert components["degradation"] == pytest.approx(-0.5)


def test_imbalance_falls_back_to_estimated_price(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_imbalance(config)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "fact_imbalance_price",
            [
                {
                    "event_time": datetime(2025, 1, 1, 0),
                    "imbalance_price_eur_mwh": 55.0,
                    "price_type": "estimated",
                },
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "imbalance_price_eur_mwh": 70.0,
                    "price_type": "final",
                },
            ],
        )
    finally:
        conn.close()

    run_settlement(config)
    components = _components(config)

    assert components["imbalance"] == pytest.approx(0.5 * 55.0 - 1.0 * 70.0)


def test_reserve_capacity_settlement_is_conditional(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_reserve(config)

    result = run_settlement(config)

    components = _components(config)
    # FCR_D up accepted: 0.5 x 10.0 = 5.0; AFRR up accepted: 0.25 x 5.0 = 1.25;
    # FCR_N not accepted -> no revenue.
    assert components["reserve_capacity"] == pytest.approx(5.0 + 1.25)
    assert result.total_pnl_eur == pytest.approx(6.25)


def test_settle_cli_reports_total_pnl(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_energy(config)
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["settle"])

    assert result.exit_code == 0
    assert "settlement: 4 rows" in result.output
    assert "total-pnl=30.00 EUR" in result.output


def test_reconcile_groups_settlement_into_components(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_energy(config)
    run_settlement(config)

    result = reconcile(config)

    assert result.table == "reconciliation"
    assert result.total_pnl_eur == pytest.approx(30.0)
    assert result.residual_eur == pytest.approx(0.0)
    assert result.components["day_ahead_revenue"] == pytest.approx(80.0)
    assert result.components["day_ahead_purchase"] == pytest.approx(-50.0)
    assert "degradation" not in result.components


def test_reserve_activation_allocates_pro_rata_and_settles_at_activation_price(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_empty_reserve_tables(config)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "dispatch_reserve",
            [
                {
                    "delivery_time": datetime(2025, 1, 1, 0),
                    "duration_hours": 1.0,
                    "capacity_mw": 0.5,
                    "product": "FCR_D",
                    "direction": "up",
                    "conditional_acceptance": True,
                },
                {
                    "delivery_time": datetime(2025, 1, 1, 1),
                    "duration_hours": 1.0,
                    "capacity_mw": 1.0,
                    "product": "AFRR",
                    "direction": "down",
                    "conditional_acceptance": True,
                },
                {
                    "delivery_time": datetime(2025, 1, 1, 2),
                    "duration_hours": 1.0,
                    "capacity_mw": 0.5,
                    "product": "MFRR",
                    "direction": "up",
                    "conditional_acceptance": True,
                },
            ],
        )
        write_table(
            conn,
            "fact_activation",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "product": "AFRR",
                    "direction": "down",
                    "activated_mw": 0.6,
                },
                {
                    "event_time": datetime(2025, 1, 1, 2),
                    "product": "MFRR",
                    "direction": "up",
                    "activated_mw": 0.4,
                },
            ],
        )
        write_table(
            conn,
            "fact_reserve_volume",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "product": "AFRR",
                    "direction": "down",
                    "procured_mw": 2.0,
                },
                {
                    "event_time": datetime(2025, 1, 1, 2),
                    "product": "MFRR",
                    "direction": "up",
                    "procured_mw": 0.5,
                },
            ],
        )
        write_table(
            conn,
            "fact_activation_price",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "product": "AFRR",
                    "direction": "down",
                    "activation_price_eur_mwh": 70.0,
                },
                {
                    "event_time": datetime(2025, 1, 1, 2),
                    "product": "MFRR",
                    "direction": "up",
                    "activation_price_eur_mwh": 60.0,
                },
            ],
        )
        write_table(
            conn,
            "fact_imbalance_price",
            [],
            columns={
                "event_time": "TIMESTAMP",
                "imbalance_price_eur_mwh": "DOUBLE",
                "price_type": "VARCHAR",
            },
        )
        for table in ("dispatch_energy", "fact_day_ahead_price", "dispatch_imbalance"):
            write_table(conn, table, [], columns={"event_time": "TIMESTAMP"})
    finally:
        conn.close()

    run_settlement(config)
    components = _components(config)

    # AFRR down: share 1.0/2.0 = 0.5, energy 0.5*0.6 = 0.3, value -0.3*70 = -21.0
    # MFRR up:   share 0.5/0.5 = 1.0, energy 1.0*0.4 = 0.4, value +0.4*60 = +24.0
    # FCR_D up: not published -> no activation.
    assert components["reserve_activation"] == pytest.approx(-21.0 + 24.0)


def test_reserve_activation_falls_back_to_imbalance_price(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_empty_reserve_tables(config)
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "dispatch_reserve",
            [
                {
                    "delivery_time": datetime(2025, 1, 1, 1),
                    "duration_hours": 1.0,
                    "capacity_mw": 1.0,
                    "product": "AFRR",
                    "direction": "down",
                    "conditional_acceptance": True,
                },
            ],
        )
        write_table(
            conn,
            "fact_activation",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "product": "AFRR",
                    "direction": "down",
                    "activated_mw": 0.6,
                },
            ],
        )
        write_table(
            conn,
            "fact_reserve_volume",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "product": "AFRR",
                    "direction": "down",
                    "procured_mw": 2.0,
                },
            ],
        )
        # No fact_activation_price -> falls back to final imbalance price.
        write_table(
            conn,
            "fact_imbalance_price",
            [
                {
                    "event_time": datetime(2025, 1, 1, 1),
                    "imbalance_price_eur_mwh": 70.0,
                    "price_type": "final",
                },
            ],
        )
        for table in ("dispatch_energy", "fact_day_ahead_price", "dispatch_imbalance"):
            write_table(conn, table, [], columns={"event_time": "TIMESTAMP"})
    finally:
        conn.close()

    run_settlement(config)
    components = _components(config)

    # share 0.5, energy 0.3, value -0.3*70 = -21.0 at the fallback imbalance price.
    assert components["reserve_activation"] == pytest.approx(-21.0)
