from datetime import date, datetime, timedelta

import pytest

from nordic_power_risk.config import PipelineConfig, Window
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.risk.run import (
    RiskEvaluator,
    RiskScheduleInterval,
    _price_paths_by_day,
    append_decision_log,
    read_risk_status,
)


def _config(tmp_path) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig(
        zone="SE3",
        windows={"primary": Window(start=date(2024, 12, 1), end=date(2025, 2, 1))},
        duckdb_path=tmp_path / "risk.duckdb",
        manifest_path=tmp_path / "manifest.json",
    )


def _seed_risk_inputs(config: PipelineConfig, *, realized=(-100.0, 100.0)) -> list[datetime]:
    training_prices = ([10.0, 20.0], [15.0, 25.0], [-10.0, 30.0], [5.0, 35.0])
    fact_rows = []
    for day_offset, prices in enumerate(training_prices):
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
    for event_time, price in zip(delivery_times, realized, strict=True):
        fact_rows.append(
            {
                "event_time": event_time,
                "issue_time": event_time - timedelta(days=1),
                "price_eur_mwh": price,
            }
        )
    forecast_rows = [
        {
            "event_time": delivery_times[0],
            "issue_time": datetime(2025, 1, 4, 9),
            "q0_01": 0.0,
            "q0_05": 1.0,
            "q0_5": 10.0,
            "q0_95": 15.0,
            "q0_99": 20.0,
        },
        {
            "event_time": delivery_times[1],
            "issue_time": datetime(2025, 1, 4, 9),
            "q0_01": 20.0,
            "q0_05": 15.0,
            "q0_5": 10.0,
            "q0_95": 1.0,
            "q0_99": 0.0,
        },
    ]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "fact_day_ahead_price", fact_rows)
        write_table(conn, "forecast_day_ahead", forecast_rows)
    finally:
        conn.close()
    return delivery_times


def _schedule(delivery_times: list[datetime]) -> list[RiskScheduleInterval]:
    issue_time = datetime(2025, 1, 4, 9)
    return [
        RiskScheduleInterval(issue_time, delivery_times[0], 1.0, 0.0, 1.0, 1.0, 0.0),
        RiskScheduleInterval(issue_time, delivery_times[1], 1.0, 1.0, 0.0, 1.0, 0.0),
    ]


def test_evaluator_computes_tail_metrics_and_training_limit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    delivery_times = _seed_risk_inputs(config)
    evaluator = RiskEvaluator(config, model_version="model", git_version="git")

    outcome = evaluator.evaluate(_schedule(delivery_times))

    assert outcome.blocked is False
    assert outcome.loss_limit_99_eur == pytest.approx(40.0)
    assert outcome.var_95_eur == pytest.approx(20.0)
    assert outcome.cvar_95_eur == pytest.approx(20.0)
    assert outcome.var_99_eur == pytest.approx(20.0)
    assert outcome.cvar_99_eur == pytest.approx(20.0)
    assert len(evaluator.records) == 2
    assert evaluator.records[0]["forecast_quantiles"]["0.01"] == pytest.approx(0.0)


def test_realized_loss_updates_drawdown_only_after_delivery_is_observable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    delivery_times = _seed_risk_inputs(config, realized=(-100.0, 100.0))
    evaluator = RiskEvaluator(config, model_version="model", git_version="git")
    evaluator.evaluate(_schedule(delivery_times))

    evaluator.advance(datetime(2025, 1, 5, 0, 30))
    assert evaluator.state.last_realized_loss_eur is None

    evaluator.advance(datetime(2025, 1, 6, 7))
    assert evaluator.state.last_realized_loss_eur == pytest.approx(200.0)
    assert evaluator.state.drawdown_eur == pytest.approx(200.0)
    assert evaluator.state.gate_reason(date(2025, 1, 6), 40.0) == "cooldown"


def test_bid_sanity_and_missing_tail_fail_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    delivery_times = _seed_risk_inputs(config)
    conn = get_connection(config.duckdb_path)
    try:
        conn.execute("UPDATE forecast_day_ahead SET q0_5 = 1000.0")
    finally:
        conn.close()

    sanity = RiskEvaluator(config).evaluate(_schedule(delivery_times))
    assert sanity.blocked is True
    assert sanity.fallback_reason == "bid_sanity"

    conn = get_connection(config.duckdb_path)
    try:
        conn.execute("ALTER TABLE forecast_day_ahead DROP COLUMN q0_99")
    finally:
        conn.close()
    missing = RiskEvaluator(config).evaluate(_schedule(delivery_times))
    assert missing.blocked is True
    assert missing.fallback_reason == "missing_quantile_0.99"


def test_decision_log_appends_and_status_reads_latest_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    delivery_times = _seed_risk_inputs(config)
    evaluator = RiskEvaluator(config, model_version="model", git_version="git")
    outcome = evaluator.evaluate(_schedule(delivery_times))

    append_decision_log(config, evaluator.records)
    append_decision_log(config, evaluator.records[:1])
    status = read_risk_status(config)

    assert outcome.blocked is False
    assert status.record_count == 3
    assert status.gate_state == "open"
    assert status.last_delivery_time == delivery_times[0]


def test_null_forecast_and_realized_values_fail_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config(tmp_path)
    delivery_times = _seed_risk_inputs(config)
    conn = get_connection(config.duckdb_path)
    try:
        conn.execute(
            "UPDATE forecast_day_ahead SET q0_99 = NULL WHERE event_time = ?",
            [delivery_times[0]],
        )
    finally:
        conn.close()

    forecast = RiskEvaluator(config).evaluate(_schedule(delivery_times))
    assert forecast.blocked is True
    assert forecast.fallback_reason == "nonfinite_forecast_quantile"

    conn = get_connection(config.duckdb_path)
    try:
        conn.execute("UPDATE forecast_day_ahead SET q0_99 = q0_5")
        conn.execute(
            "UPDATE fact_day_ahead_price SET price_eur_mwh = NULL WHERE event_time = ?",
            [delivery_times[0]],
        )
    finally:
        conn.close()
    realized = RiskEvaluator(config).evaluate(_schedule(delivery_times))
    assert realized.blocked is True
    assert realized.fallback_reason == "nonfinite_realized_prices"


def test_training_paths_normalize_valid_stockholm_dst_days() -> None:
    rows = []
    for start, count in (
        (datetime(2025, 3, 29, 23), 23),
        (datetime(2025, 10, 25, 22), 25),
        (datetime(2025, 12, 1, 23), 23),
    ):
        rows.extend(
            {
                "event_time": start + timedelta(hours=hour),
                "price_eur_mwh": float(hour),
            }
            for hour in range(count)
        )

    paths = _price_paths_by_day(rows, 24)

    assert len(paths) == 2
    assert all(len(path) == 24 for path in paths)
