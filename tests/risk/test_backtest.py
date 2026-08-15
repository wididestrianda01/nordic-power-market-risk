import json
from datetime import date, datetime, timedelta

import pytest
from typer.testing import CliRunner

from nordic_power_risk import cli
from nordic_power_risk.cli import app
from nordic_power_risk.config import DispatchConfig, PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.optimize import run as optimize_module
from nordic_power_risk.optimize.dispatch import DispatchInterval
from nordic_power_risk.risk import backtest as backtest_module
from nordic_power_risk.risk.backtest import run_risk_backtest

runner = CliRunner()


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "backtest.duckdb",
        manifest_path=tmp_path / "manifest.json",
        optimizer=DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0),
    )


def _seed(config: PipelineConfig, *, tail_breach: bool = False) -> None:
    fact_rows = []
    training_paths = ([-100.0, 200.0], [100.0, -100.0], [-50.0, 100.0], [0.0, 50.0])
    for day_offset, prices in enumerate(training_paths):
        day = datetime(2024, 12, 1, 23) + timedelta(days=day_offset)
        for hour, price in enumerate(prices):
            event_time = day + timedelta(hours=hour)
            fact_rows.append(
                {
                    "event_time": event_time,
                    "issue_time": event_time - timedelta(days=1),
                    "price_eur_mwh": price,
                }
            )
    delivery_times = [datetime(2025, 1, 4, 23), datetime(2025, 1, 5)]
    for event_time, price in zip(delivery_times, [-100.0, 200.0], strict=True):
        fact_rows.append(
            {
                "event_time": event_time,
                "issue_time": event_time - timedelta(days=1),
                "price_eur_mwh": price,
            }
        )
    forecast_rows = []
    for index, event_time in enumerate(delivery_times):
        median = [-100.0, 200.0][index]
        row = {
            "event_time": event_time,
            "issue_time": datetime(2025, 1, 4, 9),
            "duration_hours": 1.0,
            "q0_01": median,
            "q0_05": median,
            "q0_5": median,
            "q0_95": median,
            "q0_99": median,
        }
        forecast_rows.append(row)
    if tail_breach:
        forecast_rows[0]["q0_01"] = 300.0
        forecast_rows[1]["q0_01"] = -300.0
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "fact_day_ahead_price", fact_rows)
        write_table(conn, "forecast_day_ahead", forecast_rows)
    finally:
        conn.close()


def _dispatch(config: PipelineConfig):  # type: ignore[no-untyped-def]
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute("SELECT * FROM dispatch_energy ORDER BY delivery_time").fetchdf()
    finally:
        conn.close()


def test_risk_backtest_accepts_schedule_and_appends_decisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    result = run_risk_backtest(config)
    dispatch = _dispatch(config)

    assert result.dispatch.row_count == 2
    assert result.blocked_decisions == 0
    assert result.decision_count == 1
    assert dispatch["charge_mw"].iloc[0] == pytest.approx(1.0)
    assert dispatch["discharge_mw"].iloc[1] == pytest.approx(1.0)
    assert result.gate_state == "open"
    assert result.decision_log.exists()


def test_cvar_breach_replaces_every_market_leg_with_flat_decision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config, tail_breach=True)

    result = run_risk_backtest(config)
    dispatch = _dispatch(config)

    assert result.blocked_decisions == 1
    assert dispatch["charge_mw"].eq(0.0).all()
    assert dispatch["discharge_mw"].eq(0.0).all()
    assert dispatch["solver_status"].eq("risk_blocked").all()
    assert result.gate_state == "blocked"


def test_optimizer_failure_fails_closed_to_flat_schedule(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(optimize_module, "solve_energy_dispatch", fail)
    result = run_risk_backtest(config)
    dispatch = _dispatch(config)

    assert result.blocked_decisions == 1
    assert dispatch["charge_mw"].eq(0.0).all()
    assert dispatch["discharge_mw"].eq(0.0).all()
    assert dispatch["solver_status"].eq("optimizer_failure").all()
    assert result.fallback_reason == "optimizer_failure:solver unavailable"


def test_risk_gate_internal_error_fails_closed_to_flat_schedule(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    def fail(self, intervals):  # type: ignore[no-untyped-def]
        raise ValueError("bad risk input")

    monkeypatch.setattr(backtest_module.RiskEvaluator, "evaluate", fail)
    result = run_risk_backtest(config)
    dispatch = _dispatch(config)

    assert dispatch["charge_mw"].eq(0.0).all()
    assert dispatch["discharge_mw"].eq(0.0).all()
    assert result.fallback_reason == "risk_input:bad risk input"


def test_late_optimizer_failure_preserves_prior_decision_records(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)

    def fail_after_gate(config, *, energy_gate):  # type: ignore[no-untyped-def]
        intervals = tuple(
            DispatchInterval(
                issue_time=datetime(2025, 1, 4, 9),
                delivery_time=delivery_time,
                duration_hours=1.0,
                charge_mw=charge,
                discharge_mw=discharge,
                soc_mwh=1.0,
                energy_revenue_eur=0.0,
                degradation_cost_eur=0.0,
                terminal_value_eur=0.0,
                objective_eur=0.0,
            )
            for delivery_time, charge, discharge in (
                (datetime(2025, 1, 4, 23), 1.0, 0.0),
                (datetime(2025, 1, 5), 0.0, 1.0),
            )
        )
        energy_gate(intervals)
        raise RuntimeError("late solver failure")

    monkeypatch.setattr(backtest_module, "run_energy_dispatch", fail_after_gate)
    result = run_risk_backtest(config)
    records = [
        json.loads(line) for line in result.decision_log.read_text().splitlines() if line.strip()
    ]

    assert result.decision_count == 2
    assert [record["fallback_reason"] for record in records] == [
        None,
        None,
        "optimizer_failure:late solver failure",
        "optimizer_failure:late solver failure",
    ]


def test_missing_input_record_carries_resolved_versions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
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

    result = run_risk_backtest(config)
    record = json.loads(result.decision_log.read_text().strip())
    assert record["model_version"] != "unknown"
    assert record["git_version"] != "unknown"


def test_optimize_and_risk_cli_report_integrated_gate_state(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)
    monkeypatch.setattr(cli, "get_config", lambda: config)

    optimize = runner.invoke(app, ["optimize"])
    risk = runner.invoke(app, ["risk"])

    assert optimize.exit_code == 0
    assert "risk=open" in optimize.output
    assert "decisions=1" in optimize.output
    assert risk.exit_code == 0
    assert "gate=open" in risk.output
    assert "records=2" in risk.output


@pytest.mark.parametrize(
    ("product", "issue_time", "forecast_source"),
    [
        ("FCR_D", datetime(2025, 1, 4, 16, 30), "lgbm"),
        ("MFRR", datetime(2025, 1, 4, 6), "seasonal_naive"),
    ],
)
def test_optimize_cli_reports_nonzero_reserve_value_when_risk_gate_is_open(
    tmp_path,
    monkeypatch,
    product: str,
    issue_time: datetime,
    forecast_source: str,
) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    _seed(config)
    conn = get_connection(config.duckdb_path)
    try:
        conn.execute(
            "UPDATE forecast_day_ahead SET q0_01 = 0, q0_05 = 0, q0_5 = 0, q0_95 = 0, q0_99 = 0"
        )
        write_table(
            conn,
            "forecast_reserve",
            [
                {
                    "product": product,
                    "direction": "up",
                    "issue_time": issue_time,
                    "delivery_time": datetime(2025, 1, 4, 23),
                    "q0_1": 10.0,
                    "q0_5": 100.0,
                    "q0_9": 190.0,
                    "forecast_source": forecast_source,
                }
            ],
        )
    finally:
        conn.close()
    monkeypatch.setattr(cli, "get_config", lambda: config)

    result = runner.invoke(app, ["optimize"])

    assert result.exit_code == 0
    assert "dispatch_reserve: 1 rows" in result.output
    assert "capacity-value=100.00 EUR" in result.output
    assert "risk=open" in result.output
