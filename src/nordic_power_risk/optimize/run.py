"""Load forecasts, solve causal rolling dispatch, and persist separate decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import groupby
from math import isfinite
from typing import Any

from nordic_power_risk.config import DispatchConfig, PipelineConfig
from nordic_power_risk.facts.rules import day_ahead_issue_time, imbalance_forecast_issue_time
from nordic_power_risk.ingest.duckdb_io import get_connection, write_table
from nordic_power_risk.optimize.dispatch import (
    DispatchForecast,
    DispatchInterval,
    DispatchResult,
    ImbalanceDispatchInput,
    ImbalanceInterval,
    _delivery_day,
    solve_energy_dispatch,
    solve_imbalance_dispatch,
)


@dataclass(frozen=True)
class DispatchRunResult:
    table: str
    row_count: int
    energy_revenue_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float
    imbalance: ImbalanceRunResult


@dataclass(frozen=True)
class ImbalanceRunResult:
    table: str
    row_count: int
    forecast_value_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float


@dataclass(frozen=True)
class _Commitment:
    interval: DispatchInterval
    solver_status: str


@dataclass(frozen=True)
class _ImbalanceForecast:
    issue_time: datetime
    delivery_time: datetime
    price_eur_mwh: float


@dataclass(frozen=True)
class _ImbalanceCommitment:
    interval: ImbalanceInterval
    day_ahead_issue_time: datetime
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


def _validate_imbalance_forecast_columns(columns: list[str]) -> None:
    required = {"event_time", "issue_time", "q0_5"}
    if not required.issubset(columns):
        raise ValueError("forecast_imbalance is missing event_time, issue_time, or q0_5")


def _imbalance_price(row: dict[str, Any]) -> float | None:
    value = row["q0_5"]
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast_imbalance q0_5 must be numeric") from exc
    return price if isfinite(price) else None


def _read_imbalance_forecast_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    conn = get_connection(config.duckdb_path)
    try:
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'forecast_imbalance'"
        ).fetchone()[0]
        if not exists:
            return []
        cursor = conn.execute("SELECT * FROM forecast_imbalance")
        columns = [column[0] for column in cursor.description]
        _validate_imbalance_forecast_columns(columns)
        return [
            dict(zip(columns, values, strict=True)) for values in cursor.fetchall()
        ]
    finally:
        conn.close()


def _load_imbalance_forecasts(
    config: PipelineConfig,
) -> dict[datetime, _ImbalanceForecast]:
    forecasts: dict[datetime, _ImbalanceForecast] = {}
    for row in _read_imbalance_forecast_rows(config):
        price = _imbalance_price(row)
        if price is None:
            continue
        delivery_time = _as_datetime(row["event_time"])
        issue_time = _as_datetime(row["issue_time"])
        if issue_time != imbalance_forecast_issue_time(delivery_time):
            continue
        if delivery_time in forecasts:
            raise ValueError("forecast_imbalance contains duplicate eligible delivery_time")
        forecasts[delivery_time] = _ImbalanceForecast(
            issue_time=issue_time,
            delivery_time=delivery_time,
            price_eur_mwh=price,
        )
    return forecasts


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


def _imbalance_window(
    commitments: list[_Commitment], start: int, config: DispatchConfig
) -> list[_Commitment]:
    remaining = commitments[start:]
    start_day = _delivery_day(remaining[0].interval.delivery_time)
    end_day = start_day + timedelta(days=config.horizon_days)
    return [
        commitment
        for commitment in remaining
        if _delivery_day(commitment.interval.delivery_time) < end_day
    ]


def _recourse_inputs(
    commitments: list[_Commitment],
    forecast: _ImbalanceForecast | None,
) -> list[ImbalanceDispatchInput]:
    return [
        ImbalanceDispatchInput(
            delivery_time=commitment.interval.delivery_time,
            duration_hours=commitment.interval.duration_hours,
            day_ahead_charge_mw=commitment.interval.charge_mw,
            day_ahead_discharge_mw=commitment.interval.discharge_mw,
            forecast_issue_time=forecast.issue_time if index == 0 and forecast else None,
            forecast_price_eur_mwh=forecast.price_eur_mwh if index == 0 and forecast else None,
        )
        for index, commitment in enumerate(commitments)
    ]


def _solve_imbalance_windows(
    commitments: list[_Commitment],
    forecasts: dict[datetime, _ImbalanceForecast],
    config: DispatchConfig,
) -> list[_ImbalanceCommitment]:
    decisions: list[_ImbalanceCommitment] = []
    initial_soc = config.initial_soc_mwh
    for index, commitment in enumerate(commitments):
        window = _imbalance_window(commitments, index, config)
        forecast = forecasts.get(commitment.interval.delivery_time)
        inputs = _recourse_inputs(window, forecast)
        window_config = config.model_copy(update={"initial_soc_mwh": initial_soc})
        result = solve_imbalance_dispatch(inputs, window_config)
        interval = result.intervals[0]
        decisions.append(
            _ImbalanceCommitment(interval, commitment.interval.issue_time, result.solver_status)
        )
        initial_soc = interval.soc_mwh
    return decisions


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


def _imbalance_interval_row(
    commitment: _ImbalanceCommitment, terminal_value_eur: float
) -> dict[str, object]:
    interval = commitment.interval
    objective = interval.forecast_value_eur - interval.degradation_cost_eur + terminal_value_eur
    row = asdict(interval)
    row.update(
        {
            "day_ahead_issue_time": commitment.day_ahead_issue_time,
            "terminal_value_eur": terminal_value_eur,
            "objective_eur": objective,
            "solver_status": commitment.solver_status,
        }
    )
    return row


def _imbalance_rows(
    commitments: list[_ImbalanceCommitment], config: DispatchConfig
) -> list[dict[str, object]]:
    final_value = config.terminal_value_eur_mwh * commitments[-1].interval.soc_mwh
    return [
        _imbalance_interval_row(
            commitment, final_value if index == len(commitments) - 1 else 0.0
        )
        for index, commitment in enumerate(commitments)
    ]


def _persist_rows(
    config: PipelineConfig,
    energy_rows: list[dict[str, object]],
    imbalance_rows: list[dict[str, object]],
) -> tuple[int, int]:
    conn = get_connection(config.duckdb_path)
    conn.execute("BEGIN TRANSACTION")
    try:
        energy_count = write_table(conn, "dispatch_energy", energy_rows)
        imbalance_count = write_table(conn, "dispatch_imbalance", imbalance_rows)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
        return energy_count, imbalance_count
    finally:
        conn.close()


def _imbalance_result(
    rows: list[dict[str, object]], row_count: int
) -> ImbalanceRunResult:
    forecast_value = sum(float(row["forecast_value_eur"]) for row in rows)
    degradation = sum(float(row["degradation_cost_eur"]) for row in rows)
    terminal = sum(float(row["terminal_value_eur"]) for row in rows)
    return ImbalanceRunResult(
        table="dispatch_imbalance",
        row_count=row_count,
        forecast_value_eur=forecast_value,
        degradation_cost_eur=degradation,
        terminal_value_eur=terminal,
        objective_eur=forecast_value - degradation + terminal,
    )


def _result(
    rows: list[dict[str, object]],
    row_count: int,
    imbalance: ImbalanceRunResult,
) -> DispatchRunResult:
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
        imbalance=imbalance,
    )


def run_energy_dispatch(config: PipelineConfig) -> DispatchRunResult:
    """Run fixed day-ahead commitments, then causal T-60 imbalance recourse."""
    forecasts, explicit_issue_time = _load_promoted_forecasts(config)
    if config.optimizer.horizon_days > 1 and not explicit_issue_time:
        raise ValueError("multi-day horizons require explicit issue_time forecast vintages")
    energy_commitments = _solve_windows(forecasts, config.optimizer)
    energy_rows = _schedule_rows(energy_commitments, config.optimizer)
    imbalance_forecasts = _load_imbalance_forecasts(config)
    recourse = _solve_imbalance_windows(
        energy_commitments, imbalance_forecasts, config.optimizer
    )
    recourse_rows = _imbalance_rows(recourse, config.optimizer)
    energy_count, imbalance_count = _persist_rows(
        config, energy_rows, recourse_rows
    )
    imbalance = _imbalance_result(recourse_rows, imbalance_count)
    return _result(energy_rows, energy_count, imbalance)


__all__ = ["DispatchRunResult", "ImbalanceRunResult", "run_energy_dispatch"]
