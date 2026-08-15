"""Load promoted forecasts, solve rolling energy windows, and persist commitments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

from nordic_power_risk.config import DispatchConfig, PipelineConfig
from nordic_power_risk.facts.rules import day_ahead_issue_time
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.optimize.dispatch import (
    DispatchForecast,
    DispatchInterval,
    DispatchResult,
    _delivery_day,
    solve_energy_dispatch,
)


@dataclass(frozen=True)
class DispatchRunResult:
    table: str
    row_count: int
    energy_revenue_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float


@dataclass(frozen=True)
class _Commitment:
    interval: DispatchInterval
    solver_status: str


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))

def _validate_unique_vintage_deliveries(forecasts: list[DispatchForecast]) -> None:
    seen: set[tuple[datetime, datetime]] = set()
    for forecast in forecasts:
        key = (forecast.issue_time, forecast.delivery_time)
        if key in seen:
            raise ValueError("forecast vintage contains duplicate delivery_time")
        seen.add(key)



def _load_promoted_forecasts(
    config: PipelineConfig,
) -> tuple[list[DispatchForecast], bool]:
    conn = get_connection(config.duckdb_path)
    try:
        cursor = conn.execute("SELECT * FROM forecast_day_ahead ORDER BY event_time")
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]
    finally:
        conn.close()
    if not rows:
        raise ValueError("forecast_day_ahead is empty")
    if "q0_5" not in columns:
        raise ValueError("forecast_day_ahead is missing promoted median q0_5")
    explicit_issue_time = "issue_time" in columns
    forecasts = [_to_forecast(row) for row in rows]
    _validate_unique_vintage_deliveries(forecasts)
    return forecasts, explicit_issue_time


def _to_forecast(row: dict[str, Any]) -> DispatchForecast:
    delivery_time = _as_datetime(row["event_time"])
    cutoff = day_ahead_issue_time(delivery_time)
    if "issue_time" in row:
        if row["issue_time"] is None:
            raise ValueError("explicit forecast issue_time cannot be null")
        issue_time = _as_datetime(row["issue_time"])
        if issue_time > cutoff:
            raise ValueError("forecast issue_time is later than the day-ahead cutoff")
    else:
        issue_time = cutoff
    return DispatchForecast(
        issue_time=issue_time,
        delivery_time=delivery_time,
        price_eur_mwh=float(row["q0_5"]),
        duration_hours=float(row.get("duration_hours", 1.0)),
    )


def _calendar_window(
    forecasts: list[DispatchForecast], config: DispatchConfig
) -> list[DispatchForecast]:
    ordered = sorted(forecasts, key=lambda forecast: forecast.delivery_time)
    start_day = _delivery_day(ordered[0].delivery_time)
    end_day = start_day + timedelta(days=config.horizon_days)
    return [forecast for forecast in ordered if _delivery_day(forecast.delivery_time) < end_day]


def _commit_first_day(result: DispatchResult) -> list[_Commitment]:
    first_day = min(_delivery_day(interval.delivery_time) for interval in result.intervals)
    return [
        _Commitment(interval, result.solver_status)
        for interval in result.intervals
        if _delivery_day(interval.delivery_time) == first_day
    ]


def _solve_windows(
    forecasts: list[DispatchForecast], config: DispatchConfig
) -> list[_Commitment]:
    ordered = sorted(forecasts, key=lambda item: (item.issue_time, item.delivery_time))
    commitments: list[_Commitment] = []
    committed_times: set[datetime] = set()
    initial_soc = config.initial_soc_mwh
    for _, grouped in groupby(ordered, key=lambda item: item.issue_time):
        candidates = [item for item in grouped if item.delivery_time not in committed_times]
        if not candidates:
            continue
        window = _calendar_window(candidates, config)
        window_config = config.model_copy(update={"initial_soc_mwh": initial_soc})
        new_commitments = _commit_first_day(solve_energy_dispatch(window, window_config))
        commitments.extend(new_commitments)
        committed_times.update(commitment.interval.delivery_time for commitment in new_commitments)
        initial_soc = new_commitments[-1].interval.soc_mwh
    return commitments


def _interval_row(
    commitment: _Commitment, terminal_value_eur: float
) -> dict[str, object]:
    interval = commitment.interval
    objective = interval.energy_revenue_eur - interval.degradation_cost_eur + terminal_value_eur
    return {
        "issue_time": interval.issue_time,
        "delivery_time": interval.delivery_time,
        "charge_mw": interval.charge_mw,
        "discharge_mw": interval.discharge_mw,
        "soc_mwh": interval.soc_mwh,
        "energy_revenue_eur": interval.energy_revenue_eur,
        "degradation_cost_eur": interval.degradation_cost_eur,
        "terminal_value_eur": terminal_value_eur,
        "objective_eur": objective,
        "solver_status": commitment.solver_status,
    }


def _schedule_rows(
    commitments: list[_Commitment], config: DispatchConfig
) -> list[dict[str, object]]:
    final_value = config.terminal_value_eur_mwh * commitments[-1].interval.soc_mwh
    return [
        _interval_row(commitment, final_value if index == len(commitments) - 1 else 0.0)
        for index, commitment in enumerate(commitments)
    ]


def _persist_rows(config: PipelineConfig, rows: list[dict[str, object]]) -> int:
    conn = get_connection(config.duckdb_path)
    try:
        return write_table(conn, "dispatch_energy", rows)
    finally:
        conn.close()


def _result(rows: list[dict[str, object]], row_count: int) -> DispatchRunResult:
    energy = sum(float(row["energy_revenue_eur"]) for row in rows)
    degradation = sum(float(row["degradation_cost_eur"]) for row in rows)
    terminal = sum(float(row["terminal_value_eur"]) for row in rows)
    return DispatchRunResult(
        table="dispatch_energy",
        row_count=row_count,
        energy_revenue_eur=energy,
        degradation_cost_eur=degradation,
        terminal_value_eur=terminal,
        objective_eur=energy - degradation + terminal,
    )


def run_energy_dispatch(config: PipelineConfig) -> DispatchRunResult:
    """Run energy-only dispatch from promoted median forecast vintages."""
    forecasts, explicit_issue_time = _load_promoted_forecasts(config)
    if config.optimizer.horizon_days > 1 and not explicit_issue_time:
        raise ValueError("multi-day horizons require explicit issue_time forecast vintages")
    commitments = _solve_windows(forecasts, config.optimizer)
    rows = _schedule_rows(commitments, config.optimizer)
    return _result(rows, _persist_rows(config, rows))


__all__ = ["DispatchRunResult", "run_energy_dispatch"]
