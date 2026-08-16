"""Monitoring report tests: metrics computed from seeded tables + decision log."""

from __future__ import annotations

import json
from datetime import date

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.monitor.run import run_monitoring
from nordic_power_risk.risk.run import decision_log_path


def _seed(tmp_path) -> PipelineConfig:
    config = PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 2))},
        duckdb_path=tmp_path / "nordic_power_risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )
    conn = get_connection(config.duckdb_path)
    try:
        write_table(
            conn,
            "forecast_day_ahead",
            [
                {"event_time": "2020-01-01T00:00:00", "q0_1": 8.0, "q0_5": 10.0, "q0_9": 12.0},
                {"event_time": "2020-01-01T01:00:00", "q0_1": 18.0, "q0_5": 20.0, "q0_9": 22.0},
            ],
        )
        write_table(
            conn,
            "fact_day_ahead_price",
            [
                {"event_time": "2020-01-01T00:00:00", "price_eur_mwh": 11.0},
                {"event_time": "2020-01-01T01:00:00", "price_eur_mwh": 19.0},
            ],
        )
        write_table(
            conn,
            "settlement",
            [
                {"value_eur": 100.0},
                {"value_eur": -40.0},
            ],
        )
        features = []
        for hour in range(24):
            features.append(
                {
                    "event_time": f"2020-01-01T{hour:02d}:00:00",
                    "issue_time": "2019-12-31T10:00:00",
                    "price_eur_mwh": 10.0 + hour,
                    "price_lag_24h": 9.0 + hour,
                    "price_lag_168h": 8.0 + hour,
                    "hour_of_day": hour,
                    "day_of_week": 0,
                    "month": 1,
                    "is_weekend": False,
                    "is_holiday": False,
                }
            )
        write_table(conn, "feature_day_ahead", features)
    finally:
        conn.close()

    log = decision_log_path(config)
    log.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"drawdown_eur": 10.0, "breach": False, "fallback_reason": None},
                {"drawdown_eur": 25.0, "breach": True, "fallback_reason": "breach_gate"},
                {"drawdown_eur": 15.0, "breach": False, "fallback_reason": "missing_input"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_run_monitoring_computes_metrics(tmp_path) -> None:
    config = _seed(tmp_path)
    result = run_monitoring(config)

    assert result.missingness["forecast_day_ahead"] == 0.0
    assert result.missingness["fact_day_ahead_price"] == 0.0
    assert result.forecast_mae == 1.0
    assert result.interval_coverage_80 == 1.0
    assert result.realized_pnl_eur == 60.0
    assert result.max_drawdown_eur == 25.0
    assert result.breach_count == 1
    assert result.optimizer_failures == 2
    assert result.data_latency_hours is not None and result.data_latency_hours > 0.0
    assert result.drift_share is None or isinstance(result.drift_share, float)

def test_run_monitoring_writes_summary_artifact(tmp_path) -> None:
    from pathlib import Path

    config = _seed(tmp_path)
    result = run_monitoring(config)

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["breach_count"] == 1
    assert summary["forecast_mae"] == 1.0


def test_run_monitoring_tolerates_missing_tables(tmp_path) -> None:
    config = PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2020, 1, 1), end=date(2020, 1, 2))},
        duckdb_path=tmp_path / "empty.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )
    result = run_monitoring(config)

    assert result.forecast_mae is None
    assert result.realized_pnl_eur is None
    assert result.max_drawdown_eur is None
    assert result.breach_count == 0
    assert result.missingness["forecast_day_ahead"] == 1.0
