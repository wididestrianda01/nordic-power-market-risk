from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from typer.testing import CliRunner

from nordic_power_risk import cli
from nordic_power_risk.cli import app
from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.facts.rules import day_ahead_issue_time
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.optimize import run as run_module
from nordic_power_risk.optimize.dispatch import DispatchForecast, solve_energy_dispatch
from nordic_power_risk.optimize.run import run_energy_dispatch


runner = CliRunner()
ISSUE_TIME = datetime(2025, 1, 1, 9)
DELIVERY_TIME = datetime(2025, 1, 2)
STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc


def _config(
    tmp_path,  # type: ignore[no-untyped-def]
    *,
    horizon_days: int = 1,
    initial_soc_mwh: float = 1.0,
    terminal_value_eur_mwh: float = 30.0,
    degradation_cost_eur_mwh: float = 15.0,
) -> PipelineConfig:
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2025, 1, 1), end=date(2025, 1, 6))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(
            horizon_days=horizon_days,
            initial_soc_mwh=initial_soc_mwh,
            terminal_value_eur_mwh=terminal_value_eur_mwh,
            degradation_cost_eur_mwh=degradation_cost_eur_mwh,
        ),
    )


def _write_forecasts(config: PipelineConfig, rows: list[dict[str, object]]) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "forecast_day_ahead", rows)
    finally:
        conn.close()


def _write_imbalance_forecasts(
    config: PipelineConfig, rows: list[dict[str, object]]
) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "forecast_imbalance", rows)
    finally:
        conn.close()


def _write_reserve_forecasts(
    config: PipelineConfig, rows: list[dict[str, object]]
) -> None:
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "forecast_reserve", rows)
    finally:
        conn.close()


def _seed_derived_forecasts(config: PipelineConfig) -> None:
    rows = [
        {"event_time": DELIVERY_TIME + timedelta(hours=hour), "q0_5": price}
        for hour, price in enumerate((-100.0, 200.0))
    ]
    _write_forecasts(config, rows)


def _fetch_dispatch(config: PipelineConfig) -> pd.DataFrame:
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute("SELECT * FROM dispatch_energy ORDER BY delivery_time").fetchdf()
    finally:
        conn.close()



def _fetch_imbalance_dispatch(config: PipelineConfig) -> pd.DataFrame:
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute(
            "SELECT * FROM dispatch_imbalance ORDER BY delivery_time"
        ).fetchdf()
    finally:
        conn.close()


def _fetch_reserve_dispatch(config: PipelineConfig) -> pd.DataFrame:
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute(
            "SELECT * FROM dispatch_reserve ORDER BY delivery_time, product, direction"
        ).fetchdf()
    finally:
        conn.close()


def _dispatch_table_names(config: PipelineConfig) -> set[str]:
    conn = get_connection(config.duckdb_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'dispatch_%'"
            ).fetchall()
        }
    finally:
        conn.close()

def _forecast_row(
    issue_time: datetime,
    delivery_time: datetime,
    price: float,
    duration_hours: float = 1.0,
    **extra: object,
) -> dict[str, object]:
    return {
        "issue_time": issue_time,
        "event_time": delivery_time,
        "q0_5": price,
        "duration_hours": duration_hours,
        **extra,
    }


def _utc_hours_for_local_day(delivery_date: date) -> list[datetime]:
    start_local = datetime.combine(delivery_date, time(), tzinfo=STOCKHOLM)
    end_local = datetime.combine(delivery_date + timedelta(days=1), time(), tzinfo=STOCKHOLM)
    current = start_local.astimezone(UTC)
    end = end_local.astimezone(UTC)
    hours = []
    while current < end:
        hours.append(current.replace(tzinfo=None))
        current += timedelta(hours=1)
    return hours


def test_run_persists_reproducible_schedule_and_objective_breakdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)

    result = run_energy_dispatch(config)
    rows = _fetch_dispatch(config)

    assert result.table == "dispatch_energy"
    assert result.row_count == 2
    assert list(rows.columns) == [
        "issue_time",
        "delivery_time",
        "charge_mw",
        "discharge_mw",
        "soc_mwh",
        "energy_revenue_eur",
        "degradation_cost_eur",
        "terminal_value_eur",
        "objective_eur",
        "solver_status",
    ]
    assert rows["issue_time"].nunique() == 1
    assert rows["issue_time"].tolist() == [
        day_ahead_issue_time(DELIVERY_TIME),
        day_ahead_issue_time(DELIVERY_TIME + timedelta(hours=1)),
    ]
    assert rows["delivery_time"].tolist() == [DELIVERY_TIME, DELIVERY_TIME + timedelta(hours=1)]
    assert rows.iloc[-1]["terminal_value_eur"] == result.terminal_value_eur
    assert set(rows["solver_status"]) == {"optimal"}


def test_derived_forecasts_reject_unsupported_multiday_horizon(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, horizon_days=2)
    _seed_derived_forecasts(config)

    with pytest.raises(ValueError, match="explicit issue_time"):
        run_energy_dispatch(config)


def test_two_vintages_solve_calendar_windows_commit_once_and_handoff_soc(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, horizon_days=2, terminal_value_eur_mwh=0.0)
    issue_two = datetime(2025, 1, 2, 9)
    rows = [
        _forecast_row(ISSUE_TIME, datetime(2025, 1, 2), -1_000.0, 0.5),
        _forecast_row(ISSUE_TIME, datetime(2025, 1, 2, 0, 30), -1_000.0, 0.5),
        _forecast_row(ISSUE_TIME, datetime(2025, 1, 3), 1_000.0),
        _forecast_row(ISSUE_TIME, datetime(2025, 1, 4), 0.0),
        _forecast_row(issue_two, datetime(2025, 1, 3), 1_000.0, 0.5),
        _forecast_row(issue_two, datetime(2025, 1, 3, 0, 30), 1_000.0, 0.5),
        _forecast_row(issue_two, datetime(2025, 1, 4), 0.0),
        _forecast_row(issue_two, datetime(2025, 1, 5), 0.0),
    ]
    _write_forecasts(config, rows)
    observed_windows: list[list[datetime]] = []
    real_solve = run_module.solve_energy_dispatch

    def capture_window(forecasts, dispatch_config):  # type: ignore[no-untyped-def]
        observed_windows.append([forecast.delivery_time for forecast in forecasts])
        return real_solve(forecasts, dispatch_config)

    monkeypatch.setattr(run_module, "solve_energy_dispatch", capture_window)

    result = run_energy_dispatch(config)
    persisted = _fetch_dispatch(config)

    assert observed_windows == [
        [datetime(2025, 1, 2), datetime(2025, 1, 2, 0, 30), datetime(2025, 1, 3)],
        [datetime(2025, 1, 3), datetime(2025, 1, 3, 0, 30), datetime(2025, 1, 4)],
    ]
    assert result.row_count == 4
    assert persisted["delivery_time"].is_unique
    assert persisted["delivery_time"].tolist() == [
        datetime(2025, 1, 2),
        datetime(2025, 1, 2, 0, 30),
        datetime(2025, 1, 3),
        datetime(2025, 1, 3, 0, 30),
    ]
    assert persisted["issue_time"].tolist() == [
        ISSUE_TIME,
        ISSUE_TIME,
        issue_two,
        issue_two,
    ]
    previous_soc = config.optimizer.initial_soc_mwh
    for row in persisted.itertuples(index=False):
        expected_soc = (
            previous_soc
            + 0.9487 * row.charge_mw * 0.5
            - row.discharge_mw * 0.5 / 0.9487
        )
        assert row.soc_mwh == pytest.approx(expected_soc)
        previous_soc = row.soc_mwh


@pytest.mark.parametrize(
    ("delivery_date", "expected_hours"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)],
)
def test_stockholm_delivery_day_accepts_dst_hour_counts(
    tmp_path, delivery_date: date, expected_hours: int
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, initial_soc_mwh=0.0, terminal_value_eur_mwh=0.0)
    delivery_hours = _utc_hours_for_local_day(delivery_date)
    issue_time = day_ahead_issue_time(delivery_hours[0])
    _write_forecasts(
        config,
        [_forecast_row(issue_time, delivery, 0.0) for delivery in delivery_hours],
    )

    result = run_energy_dispatch(config)

    assert len(delivery_hours) == expected_hours
    assert result.row_count == expected_hours


def test_terminal_inventory_is_counted_once_at_final_committed_boundary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(
        tmp_path, horizon_days=2, initial_soc_mwh=2.0, terminal_value_eur_mwh=30.0
    )
    issue_two = datetime(2025, 1, 2, 9)
    _write_forecasts(
        config,
        [
            _forecast_row(ISSUE_TIME, datetime(2025, 1, 2), 0.0),
            _forecast_row(ISSUE_TIME, datetime(2025, 1, 3), 0.0),
            _forecast_row(issue_two, datetime(2025, 1, 3), 0.0),
            _forecast_row(issue_two, datetime(2025, 1, 4), 0.0),
        ],
    )

    result = run_energy_dispatch(config)
    persisted = _fetch_dispatch(config)

    assert result.row_count == 2
    assert result.energy_revenue_eur == pytest.approx(0.0)
    assert result.degradation_cost_eur == pytest.approx(0.0)
    assert result.terminal_value_eur == pytest.approx(60.0)
    assert result.objective_eur == pytest.approx(60.0)
    assert persisted["terminal_value_eur"].tolist() == [0.0, 60.0]


def test_rejects_forecast_vintage_later_than_day_ahead_cutoff(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    late_issue = day_ahead_issue_time(DELIVERY_TIME) + timedelta(minutes=1)
    _write_forecasts(config, [_forecast_row(late_issue, DELIVERY_TIME, 10.0)])

    with pytest.raises(ValueError, match="later than.*cutoff"):
        run_energy_dispatch(config)


def test_realized_price_columns_cannot_change_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    rows = [
        {
            "event_time": DELIVERY_TIME + timedelta(hours=hour),
            "q0_5": price,
            "realized_price_eur_mwh": realized,
        }
        for hour, price, realized in [(0, -100.0, 9_999.0), (1, 200.0, -9_999.0)]
    ]
    _write_forecasts(config, rows)
    run_energy_dispatch(config)
    first = _fetch_dispatch(config)
    rows[0]["realized_price_eur_mwh"] = -9_999.0
    rows[1]["realized_price_eur_mwh"] = 9_999.0
    _write_forecasts(config, rows)

    run_energy_dispatch(config)
    second = _fetch_dispatch(config)

    pd.testing.assert_frame_equal(first, second)


def test_persisted_values_match_solution_sql_totals_and_replace_deterministically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)
    forecasts = [
        DispatchForecast(ISSUE_TIME, DELIVERY_TIME, -100.0),
        DispatchForecast(ISSUE_TIME, DELIVERY_TIME + timedelta(hours=1), 200.0),
    ]
    solved = solve_energy_dispatch(forecasts, config.optimizer)

    first_result = run_energy_dispatch(config)
    first = _fetch_dispatch(config)
    second_result = run_energy_dispatch(config)
    second = _fetch_dispatch(config)

    for row, interval in zip(first.itertuples(index=False), solved.intervals, strict=True):
        assert row.charge_mw == pytest.approx(interval.charge_mw)
        assert row.discharge_mw == pytest.approx(interval.discharge_mw)
        assert row.soc_mwh == pytest.approx(interval.soc_mwh)
        assert row.energy_revenue_eur == pytest.approx(interval.energy_revenue_eur)
        assert row.degradation_cost_eur == pytest.approx(interval.degradation_cost_eur)
        assert row.objective_eur == pytest.approx(
            row.energy_revenue_eur - row.degradation_cost_eur + row.terminal_value_eur
        )
    assert first["terminal_value_eur"].iloc[:-1].eq(0.0).all()
    assert first["terminal_value_eur"].iloc[-1] == pytest.approx(
        config.optimizer.terminal_value_eur_mwh * first["soc_mwh"].iloc[-1]
    )
    assert first["energy_revenue_eur"].sum() == pytest.approx(first_result.energy_revenue_eur)
    assert first["degradation_cost_eur"].sum() == pytest.approx(first_result.degradation_cost_eur)
    assert first["terminal_value_eur"].sum() == pytest.approx(first_result.terminal_value_eur)
    assert first["objective_eur"].sum() == pytest.approx(first_result.objective_eur)
    assert first_result == second_result
    pd.testing.assert_frame_equal(first, second)


def test_solver_failure_does_not_persist_partial_schedule(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("HiGHS failed: termination=infeasible")

    monkeypatch.setattr(run_module, "solve_energy_dispatch", fail)

    with pytest.raises(RuntimeError, match="infeasible"):
        run_energy_dispatch(config)
    conn = get_connection(config.duckdb_path)
    try:
        table_count = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'dispatch_energy'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert table_count == 0


def test_optimize_cli_runs_dispatch_and_reports_persisted_rows(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 0
    assert "dispatch_energy: 2 rows" in result.output
    assert str(config.duckdb_path) in result.output
    assert "dispatch_imbalance: 2 rows" in result.output
    assert "dispatch_reserve: 0 rows" in result.output


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("empty", "forecast_day_ahead is empty"),
        ("malformed", "missing promoted median"),
        ("runtime", "solver unavailable"),
    ],
)
def test_optimize_cli_fails_concisely_for_invalid_or_runtime_input(
    tmp_path, monkeypatch, scenario: str, message: str
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    if scenario == "empty":
        conn = get_connection(config.duckdb_path)
        try:
            write_table(
                conn,
                "forecast_day_ahead",
                [],
                columns={"event_time": "TIMESTAMP", "q0_5": "DOUBLE"},
            )
        finally:
            conn.close()
    elif scenario == "malformed":
        _write_forecasts(config, [{"event_time": DELIVERY_TIME, "realized_price": 1.0}])
    else:
        _seed_derived_forecasts(config)
        monkeypatch.setattr(
            run_module,
            "run_energy_dispatch",
            lambda config: (_ for _ in ()).throw(RuntimeError("solver unavailable")),
        )
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 1
    assert message in result.output


def test_rejects_duplicate_delivery_within_forecast_vintage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    duplicate = _forecast_row(ISSUE_TIME, DELIVERY_TIME, 10.0)
    _write_forecasts(config, [duplicate, duplicate.copy()])

    with pytest.raises(ValueError, match="duplicate delivery_time"):
        run_energy_dispatch(config)


def test_seven_day_vintage_excludes_day_eight_and_commits_day_one(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    config = _config(
        tmp_path, horizon_days=7, initial_soc_mwh=0.0, terminal_value_eur_mwh=0.0
    )
    deliveries = [DELIVERY_TIME + timedelta(days=offset) for offset in range(8)]
    _write_forecasts(
        config,
        [_forecast_row(ISSUE_TIME, delivery, 0.0) for delivery in deliveries],
    )
    observed_windows: list[list[datetime]] = []
    real_solve = run_module.solve_energy_dispatch

    def capture_window(forecasts, dispatch_config):  # type: ignore[no-untyped-def]
        observed_windows.append([forecast.delivery_time for forecast in forecasts])
        return real_solve(forecasts, dispatch_config)

    monkeypatch.setattr(run_module, "solve_energy_dispatch", capture_window)

    result = run_energy_dispatch(config)
    persisted = _fetch_dispatch(config)

    assert observed_windows == [deliveries[:7]]
    assert result.row_count == 1
    assert persisted["delivery_time"].tolist() == deliveries[:1]


def test_eligible_imbalance_forecasts_change_actual_setpoints_with_soc_handoff(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    deliveries = [DELIVERY_TIME, DELIVERY_TIME + timedelta(minutes=30)]
    _write_forecasts(
        config,
        [_forecast_row(ISSUE_TIME, delivery, 0.0, 0.5) for delivery in deliveries],
    )
    _write_imbalance_forecasts(
        config,
        [
            {
                "issue_time": delivery - timedelta(minutes=60),
                "event_time": delivery,
                "q0_5": price,
            }
            for delivery, price in zip(deliveries, (1_000.0, -1_000.0), strict=True)
        ],
    )

    result = run_energy_dispatch(config)
    energy = _fetch_dispatch(config)
    imbalance = _fetch_imbalance_dispatch(config)

    assert result.imbalance.table == "dispatch_imbalance"
    assert result.imbalance.row_count == 2
    assert energy["charge_mw"].eq(0.0).all()
    assert energy["discharge_mw"].eq(0.0).all()
    assert imbalance.iloc[0]["actual_discharge_mw"] > 0.0
    assert imbalance.iloc[1]["actual_charge_mw"] > 0.0
    previous_soc = config.optimizer.initial_soc_mwh
    for row in imbalance.itertuples(index=False):
        expected_position = (
            row.actual_discharge_mw
            - row.actual_charge_mw
            - (row.day_ahead_discharge_mw - row.day_ahead_charge_mw)
        )
        expected_soc = (
            previous_soc
            + config.optimizer.one_way_efficiency * row.actual_charge_mw * row.duration_hours
            - row.actual_discharge_mw
            * row.duration_hours
            / config.optimizer.one_way_efficiency
        )
        assert row.imbalance_position_mw == pytest.approx(expected_position)
        assert row.forecast_value_eur == pytest.approx(
            row.forecast_price_eur_mwh
            * row.imbalance_position_mw
            * row.duration_hours
        )
        assert row.forecast_value_eur > 0.0
        assert row.objective_eur == pytest.approx(
            row.forecast_value_eur
            - row.degradation_cost_eur
            + row.terminal_value_eur
        )
        assert 0.0 <= row.actual_charge_mw <= config.optimizer.power_limit_mw
        assert 0.0 <= row.actual_discharge_mw <= config.optimizer.power_limit_mw
        assert row.actual_charge_mw * row.actual_discharge_mw == pytest.approx(
            0.0, abs=1e-8
        )
        assert 0.0 <= row.soc_mwh <= config.optimizer.energy_capacity_mwh
        assert row.soc_mwh == pytest.approx(expected_soc, abs=1e-8)
        previous_soc = row.soc_mwh


@pytest.mark.parametrize("scenario", ["missing", "stale", "late"])
def test_unavailable_imbalance_forecast_keeps_day_ahead_schedule(
    tmp_path, scenario: str
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _seed_derived_forecasts(config)
    if scenario != "missing":
        offset = -timedelta(minutes=61) if scenario == "stale" else -timedelta(minutes=59)
        _write_imbalance_forecasts(
            config,
            [
                {
                    "issue_time": delivery + offset,
                    "event_time": delivery,
                    "q0_5": 10_000.0,
                }
                for delivery in (DELIVERY_TIME, DELIVERY_TIME + timedelta(hours=1))
            ],
        )

    run_energy_dispatch(config)
    energy = _fetch_dispatch(config)
    imbalance = _fetch_imbalance_dispatch(config)

    assert imbalance["imbalance_position_mw"].eq(0.0).all()
    assert imbalance["forecast_issue_time"].isna().all()
    assert imbalance["forecast_price_eur_mwh"].isna().all()
    assert imbalance["forecast_value_eur"].eq(0.0).all()
    assert imbalance["actual_charge_mw"].tolist() == pytest.approx(
        energy["charge_mw"].tolist()
    )
    assert imbalance["actual_discharge_mw"].tolist() == pytest.approx(
        energy["discharge_mw"].tolist()
    )


def test_late_forecast_and_realized_data_cannot_change_recourse(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(config, [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)])
    cutoff = DELIVERY_TIME - timedelta(minutes=60)
    rows = [
        {
            "issue_time": cutoff,
            "event_time": DELIVERY_TIME,
            "q0_5": 500.0,
            "realized_price_eur_mwh": -99_999.0,
        },
        {
            "issue_time": cutoff + timedelta(minutes=1),
            "event_time": DELIVERY_TIME,
            "q0_5": 99_999.0,
            "realized_price_eur_mwh": 99_999.0,
        },
    ]
    _write_imbalance_forecasts(config, rows)
    run_energy_dispatch(config)
    first_energy = _fetch_dispatch(config)
    first_imbalance = _fetch_imbalance_dispatch(config)
    rows[1]["q0_5"] = -99_999.0
    rows[0]["realized_price_eur_mwh"] = 99_999.0
    rows[1]["realized_price_eur_mwh"] = -99_999.0
    _write_imbalance_forecasts(config, rows)

    run_energy_dispatch(config)
    second_energy = _fetch_dispatch(config)
    second_imbalance = _fetch_imbalance_dispatch(config)

    pd.testing.assert_frame_equal(first_energy, second_energy)
    pd.testing.assert_frame_equal(first_imbalance, second_imbalance)
    assert first_imbalance.iloc[0]["imbalance_position_mw"] > 0.0


@pytest.mark.parametrize("preexisting", [False, True], ids=["empty", "replace"])
def test_dispatch_tables_replace_atomically(
    tmp_path, monkeypatch, preexisting: bool
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    old_energy = [{"marker": "old-energy"}]
    old_imbalance = [{"marker": "old-imbalance"}]
    if preexisting:
        conn = get_connection(config.duckdb_path)
        try:
            write_table(conn, "dispatch_energy", old_energy)
            write_table(conn, "dispatch_imbalance", old_imbalance)
        finally:
            conn.close()
    real_write = run_module.write_table

    def fail_second(conn, table, rows):  # type: ignore[no-untyped-def]
        if table == "dispatch_imbalance":
            raise RuntimeError("second write failed")
        return real_write(conn, table, rows)

    monkeypatch.setattr(run_module, "write_table", fail_second)
    with pytest.raises(RuntimeError, match="second write failed"):
        run_module._persist_rows(
            config,
            [{"marker": "new-energy"}],
            [{"marker": "new-imbalance"}],
            [],
        )

    if not preexisting:
        assert _dispatch_table_names(config) == set()
        return
    conn = get_connection(config.duckdb_path)
    try:
        energy = conn.execute("SELECT * FROM dispatch_energy").fetchall()
        imbalance = conn.execute("SELECT * FROM dispatch_imbalance").fetchall()
    finally:
        conn.close()
    assert energy == [("old-energy",)]
    assert imbalance == [("old-imbalance",)]


def test_stage_two_failure_persists_neither_table_and_cli_fails(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("imbalance solver unavailable")

    monkeypatch.setattr(run_module, "solve_imbalance_dispatch", fail)
    with pytest.raises(RuntimeError, match="imbalance solver unavailable"):
        run_energy_dispatch(config)
    assert _dispatch_table_names(config) == set()
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 1
    assert "imbalance solver unavailable" in result.output
    assert _dispatch_table_names(config) == set()


@pytest.mark.parametrize(
    "median",
    [None, float("nan"), float("inf"), float("-inf")],
    ids=["null", "nan", "positive-infinity", "negative-infinity"],
)
def test_unavailable_median_persists_flat_imbalance(tmp_path, median) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _seed_derived_forecasts(config)
    _write_imbalance_forecasts(
        config,
        [
            {
                "issue_time": DELIVERY_TIME - timedelta(minutes=60),
                "event_time": DELIVERY_TIME,
                "q0_5": median,
            }
        ],
    )

    run_energy_dispatch(config)
    imbalance = _fetch_imbalance_dispatch(config)

    assert imbalance["imbalance_position_mw"].eq(0.0).all()
    assert imbalance["forecast_issue_time"].isna().all()
    assert imbalance["forecast_price_eur_mwh"].isna().all()
    assert imbalance["forecast_value_eur"].eq(0.0).all()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"event_time": DELIVERY_TIME, "q0_5": 1.0}],
            "missing event_time, issue_time, or q0_5",
        ),
        (
            [
                {
                    "issue_time": DELIVERY_TIME - timedelta(minutes=60),
                    "event_time": DELIVERY_TIME,
                    "q0_5": "not-a-number",
                }
            ],
            "q0_5 must be numeric",
        ),
        (
            [
                {
                    "issue_time": DELIVERY_TIME - timedelta(minutes=60),
                    "event_time": DELIVERY_TIME,
                    "q0_5": 1.0,
                },
                {
                    "issue_time": DELIVERY_TIME - timedelta(minutes=60),
                    "event_time": DELIVERY_TIME,
                    "q0_5": 2.0,
                },
            ],
            "duplicate eligible delivery_time",
        ),
    ],
    ids=["missing-schema", "nonnumeric", "duplicate-eligible"],
)
def test_malformed_imbalance_forecast_fails_clearly(
    tmp_path, rows: list[dict[str, object]], message: str
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed_derived_forecasts(config)
    _write_imbalance_forecasts(config, rows)

    with pytest.raises(ValueError, match=message):
        run_energy_dispatch(config)


def test_nonzero_day_ahead_commitment_is_fixed_during_recourse(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(
        config,
        [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 1_000.0)],
    )
    run_energy_dispatch(config)
    baseline_energy = _fetch_dispatch(config)
    cutoff = DELIVERY_TIME - timedelta(minutes=60)
    _write_imbalance_forecasts(
        config,
        [{"issue_time": cutoff, "event_time": DELIVERY_TIME, "q0_5": -1_000.0}],
    )

    run_energy_dispatch(config)
    energy = _fetch_dispatch(config)
    imbalance = _fetch_imbalance_dispatch(config)

    pd.testing.assert_frame_equal(baseline_energy, energy)
    assert imbalance["day_ahead_charge_mw"].tolist() == pytest.approx(
        energy["charge_mw"].tolist()
    )
    assert imbalance["day_ahead_discharge_mw"].tolist() == pytest.approx(
        energy["discharge_mw"].tolist()
    )
    assert imbalance.iloc[0]["day_ahead_issue_time"] == energy.iloc[0]["issue_time"]
    assert imbalance.iloc[0]["forecast_issue_time"] == cutoff
    assert imbalance.iloc[0]["decision_time"] == cutoff
    assert imbalance.iloc[0]["day_ahead_issue_time"] != cutoff
    assert imbalance.iloc[0]["actual_charge_mw"] != pytest.approx(
        energy.iloc[0]["charge_mw"]
    )


def test_degradation_threshold_changes_persisted_recourse_decision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    positions: dict[float, float] = {}
    for degradation_cost in (15.0, 40.0):
        config = _config(
            tmp_path / str(int(degradation_cost)),
            initial_soc_mwh=1.0,
            terminal_value_eur_mwh=0.0,
            degradation_cost_eur_mwh=degradation_cost,
        )
        _write_forecasts(
            config,
            [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)],
        )
        _write_imbalance_forecasts(
            config,
            [
                {
                    "issue_time": DELIVERY_TIME - timedelta(minutes=60),
                    "event_time": DELIVERY_TIME,
                    "q0_5": 25.0,
                }
            ],
        )
        run_energy_dispatch(config)
        positions[degradation_cost] = _fetch_imbalance_dispatch(config).iloc[0][
            "imbalance_position_mw"
        ]

    assert positions[15.0] > 0.0
    assert positions[40.0] == pytest.approx(0.0)


def _reserve_row(
    delivery_time: datetime,
    *,
    product: str = "FCR_D",
    direction: str = "up",
    price: object = 100.0,
    issue_offset: timedelta = timedelta(),
    **extra: object,
) -> dict[str, object]:
    issue_time = (
        datetime.combine(
            delivery_time.replace(tzinfo=UTC).astimezone(STOCKHOLM).date() - timedelta(days=1),
            time(17, 30),
            tzinfo=STOCKHOLM,
        ).astimezone(UTC).replace(tzinfo=None)
        + issue_offset
    )
    return {
        "product": product,
        "direction": direction,
        "issue_time": issue_time,
        "delivery_time": delivery_time,
        "q0_1": 10.0,
        "q0_5": price,
        "q0_9": 190.0,
        "forecast_source": "lgbm",
        **extra,
    }


def _reserve_row_with(field: str, value: object) -> dict[str, object]:
    row = _reserve_row(DELIVERY_TIME)
    row[field] = value
    return row


def test_run_persists_normalized_conditional_reserve_decisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(config, [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)])
    _write_reserve_forecasts(
        config,
        [
            _reserve_row(DELIVERY_TIME, product="FCR_N", direction="symmetric"),
            _reserve_row(DELIVERY_TIME, product="FCR_D", direction="up"),
            _reserve_row(DELIVERY_TIME, product="FCR_D", direction="down"),
        ],
    )

    result = run_energy_dispatch(config)
    rows = _fetch_reserve_dispatch(config)
    imbalance = _fetch_imbalance_dispatch(config).iloc[0]

    assert result.reserve.table == "dispatch_reserve"
    assert result.reserve.row_count == 3
    assert list(rows.columns) == [
        "product",
        "direction",
        "issue_time",
        "delivery_time",
        "duration_hours",
        "forecast_value_eur_mw_h",
        "capacity_mw",
        "reserved_up_mw",
        "reserved_down_mw",
        "minimum_soc_mwh",
        "maximum_soc_mwh",
        "conditional_acceptance",
        "capacity_value_eur",
        "solver_status",
    ]
    assert rows["conditional_acceptance"].tolist() == (
        rows["capacity_mw"] > 1e-8
    ).tolist()
    assert (rows["capacity_mw"] > 1e-8).sum() == 1
    assert rows["capacity_value_eur"].tolist() == pytest.approx(
        (
            rows["forecast_value_eur_mw_h"]
            * rows["capacity_mw"]
            * rows["duration_hours"]
        ).tolist()
    )
    assert rows["capacity_value_eur"].sum() == pytest.approx(
        result.reserve.capacity_value_eur
    )
    assert imbalance["reserved_up_mw"] == pytest.approx(rows["reserved_up_mw"].sum())
    assert imbalance["reserved_down_mw"] == pytest.approx(
        rows["reserved_down_mw"].sum()
    )
    assert imbalance["actual_discharge_mw"] <= 1.0 - imbalance["reserved_up_mw"] + 1e-8
    assert imbalance["actual_charge_mw"] <= 1.0 - imbalance["reserved_down_mw"] + 1e-8
    assert imbalance["minimum_soc_mwh"] - 1e-8 <= imbalance["soc_mwh"]
    assert imbalance["soc_mwh"] <= imbalance["maximum_soc_mwh"] + 1e-8


def test_reserve_dispatch_ignores_realized_prices(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(config, [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)])
    base = _reserve_row(DELIVERY_TIME, realized_price_eur_mw_h=-1_000_000.0)
    _write_reserve_forecasts(config, [base])
    first = run_energy_dispatch(config).reserve
    first_rows = _fetch_reserve_dispatch(config)

    base["realized_price_eur_mw_h"] = 1_000_000.0
    _write_reserve_forecasts(config, [base])
    second = run_energy_dispatch(config).reserve
    second_rows = _fetch_reserve_dispatch(config)

    assert second == first
    pd.testing.assert_frame_equal(second_rows, first_rows)


def test_missing_reserve_forecasts_fall_back_flat_with_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _seed_derived_forecasts(config)

    result = run_energy_dispatch(config)
    rows = _fetch_reserve_dispatch(config)

    assert result.reserve.row_count == 0
    assert rows.empty
    assert "capacity_mw" in rows.columns
    assert "conditional_acceptance" in rows.columns


@pytest.mark.parametrize(
    "rows,message",
    [
        ([{"product": "FCR_D"}], "missing required columns"),
        ([_reserve_row(DELIVERY_TIME, price="bad")], "q0_5 must be numeric"),
        ([_reserve_row(DELIVERY_TIME, product="aFRR")], "unsupported FCR"),
        ([_reserve_row(DELIVERY_TIME, direction="sideways")], "unsupported FCR"),
        ([_reserve_row_with("forecast_source", "seasonal_naive")], "forecast_source"),
        ([_reserve_row(DELIVERY_TIME), _reserve_row(DELIVERY_TIME)], "duplicate"),
        ([_reserve_row_with("issue_time", "not-a-time")], "issue_time must be a valid timestamp"),
        ([_reserve_row_with("delivery_time", None)], "delivery_time must be a valid timestamp"),
    ],
    ids=[
        "missing-columns", "nonnumeric-price", "unsupported-product",
        "unsupported-direction", "unsupported-source", "duplicate-exact-key",
        "invalid-issue-time", "null-delivery-time",
    ],
)
def test_malformed_reserve_forecasts_fail_before_persistence(
    tmp_path, rows: list[dict[str, object]], message: str
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _seed_derived_forecasts(config)
    _write_reserve_forecasts(config, rows)

    with pytest.raises(ValueError, match=message):
        run_energy_dispatch(config)

    assert _dispatch_table_names(config) == set()


def test_three_dispatch_tables_replace_atomically(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _seed_derived_forecasts(config)
    _write_reserve_forecasts(config, [_reserve_row(DELIVERY_TIME)])
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "dispatch_energy", [{"sentinel": "old-energy"}])
        write_table(conn, "dispatch_imbalance", [{"sentinel": "old-imbalance"}])
        write_table(conn, "dispatch_reserve", [{"sentinel": "old-reserve"}])
    finally:
        conn.close()
    real_write = run_module.write_table

    def fail_third(conn, table, rows, columns=None):  # type: ignore[no-untyped-def]
        if table == "dispatch_reserve":
            raise RuntimeError("reserve write failed")
        return real_write(conn, table, rows, columns=columns)

    monkeypatch.setattr(run_module, "write_table", fail_third)
    with pytest.raises(RuntimeError, match="reserve write failed"):
        run_energy_dispatch(config)

    conn = get_connection(config.duckdb_path)
    try:
        assert conn.execute("SELECT * FROM dispatch_energy").fetchall() == [("old-energy",)]
        assert conn.execute("SELECT * FROM dispatch_imbalance").fetchall() == [("old-imbalance",)]
        assert conn.execute("SELECT * FROM dispatch_reserve").fetchall() == [("old-reserve",)]
    finally:
        conn.close()


def test_optimize_cli_reports_nonzero_reserve_value(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(config, [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)])
    _write_reserve_forecasts(config, [_reserve_row(DELIVERY_TIME, price=100.0)])
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 0
    assert "dispatch_reserve: 1 rows" in result.output
    assert "capacity-value=100.00 EUR" in result.output


def test_optimize_cli_fails_for_invalid_reserve_source(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path, terminal_value_eur_mwh=0.0)
    _write_forecasts(config, [_forecast_row(ISSUE_TIME, DELIVERY_TIME, 0.0)])
    _write_reserve_forecasts(
        config, [_reserve_row_with("forecast_source", "seasonal_naive")]
    )
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 1
    assert "forecast_source" in result.output
    assert _dispatch_table_names(config) == set()
