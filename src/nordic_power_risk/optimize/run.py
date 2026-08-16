"""Load forecasts, solve causal rolling dispatch, and persist separate decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from itertools import groupby
from math import isfinite
from typing import Any

from nordic_power_risk.config import DispatchConfig, PipelineConfig
from nordic_power_risk.facts.rules import (
    day_ahead_issue_time,
    delivery_day,
    imbalance_forecast_issue_time,
)
from nordic_power_risk.ingest.duckdb_io import (
    coerce_datetime,
    fetch_scalar,
    get_connection,
    write_table,
)
from nordic_power_risk.optimize.dispatch import (
    DispatchForecast,
    DispatchInterval,
    DispatchResult,
    ImbalanceDispatchInput,
    ImbalanceInterval,
    ReserveForecast,
    ReserveHeadroom,
    ReserveInterval,
    solve_balancing_reserve_dispatch,
    solve_energy_dispatch,
    solve_imbalance_dispatch,
    solve_reserve_dispatch,
)
from nordic_power_risk.reserves import RESERVE_PRODUCTS, forecast_source

RESERVE_DISPATCH_COLUMNS = {
    "product": "VARCHAR",
    "direction": "VARCHAR",
    "issue_time": "TIMESTAMP",
    "delivery_time": "TIMESTAMP",
    "duration_hours": "DOUBLE",
    "forecast_value_eur_mw_h": "DOUBLE",
    "capacity_mw": "DOUBLE",
    "reserved_up_mw": "DOUBLE",
    "reserved_down_mw": "DOUBLE",
    "minimum_soc_mwh": "DOUBLE",
    "maximum_soc_mwh": "DOUBLE",
    "conditional_acceptance": "BOOLEAN",
    "capacity_value_eur": "DOUBLE",
    "solver_status": "VARCHAR",
}


class MissingForecastInputError(ValueError):
    """Required promoted day-ahead forecast input is unavailable."""


@dataclass(frozen=True)
class DispatchRunResult:
    table: str
    row_count: int
    energy_revenue_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float
    imbalance: ImbalanceRunResult
    reserve: ReserveRunResult


@dataclass(frozen=True)
class ImbalanceRunResult:
    table: str
    row_count: int
    forecast_value_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float


@dataclass(frozen=True)
class ReserveRunResult:
    table: str
    row_count: int
    capacity_value_eur: float


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
        raise MissingForecastInputError("forecast_day_ahead is empty")
    if "q0_5" not in columns:
        raise MissingForecastInputError("forecast_day_ahead is missing promoted median q0_5")
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
        exists = fetch_scalar(
            conn,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'forecast_imbalance'",
        )
        if not exists:
            return []
        cursor = conn.execute("SELECT * FROM forecast_imbalance")
        columns = [column[0] for column in cursor.description]
        _validate_imbalance_forecast_columns(columns)
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]
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
        delivery_time = coerce_datetime(row["event_time"])
        issue_time = coerce_datetime(row["issue_time"])
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


def _validate_reserve_forecast_columns(columns: list[str]) -> None:
    required = {
        "product",
        "direction",
        "issue_time",
        "delivery_time",
        "q0_1",
        "q0_5",
        "q0_9",
        "forecast_source",
    }
    if not required.issubset(columns):
        raise ValueError("forecast_reserve is missing required columns")


def _read_reserve_forecast_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    conn = get_connection(config.duckdb_path)
    try:
        exists = fetch_scalar(
            conn,
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'forecast_reserve'",
        )
        if not exists:
            return []
        cursor = conn.execute("SELECT * FROM forecast_reserve")
        columns = [column[0] for column in cursor.description]
        _validate_reserve_forecast_columns(columns)
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]
    finally:
        conn.close()


def _reserve_price(row: dict[str, Any]) -> float | None:
    value = row["q0_5"]
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast_reserve q0_5 must be numeric") from exc
    return price if isfinite(price) else None


def _reserve_datetime(row: dict[str, Any], key: str) -> datetime:
    value = row[key]
    if value is None:
        raise ValueError(f"forecast_reserve {key} must be a valid timestamp")
    try:
        return coerce_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"forecast_reserve {key} must be a valid timestamp") from exc


def _to_reserve_forecast(row: dict[str, Any], price: float) -> ReserveForecast:
    product = str(row["product"])
    if product == "FFR":
        raise ValueError("unsupported balancing reserve product or direction")
    if product not in {p for p, _ in RESERVE_PRODUCTS}:
        raise ValueError("unsupported FCR or balancing reserve product or direction")
    expected_source = forecast_source(product)
    if row["forecast_source"] != expected_source:
        raise ValueError(f"forecast_reserve {product} forecast_source must be {expected_source}")
    return ReserveForecast(
        product=product,
        direction=str(row["direction"]),
        issue_time=_reserve_datetime(row, "issue_time"),
        delivery_time=_reserve_datetime(row, "delivery_time"),
        forecast_value_eur_mw_h=price,
    )


def _load_reserve_forecasts(
    config: PipelineConfig,
) -> dict[datetime, list[ReserveForecast]]:
    forecasts: dict[datetime, list[ReserveForecast]] = {}
    for row in _read_reserve_forecast_rows(config):
        price = _reserve_price(row)
        if price is None:
            continue
        forecast = _to_reserve_forecast(row, price)
        forecasts.setdefault(forecast.delivery_time, []).append(forecast)
    return forecasts


def _to_forecast(row: dict[str, Any]) -> DispatchForecast:
    delivery_time = coerce_datetime(row["event_time"])
    cutoff = day_ahead_issue_time(delivery_time)
    if "issue_time" in row:
        if row["issue_time"] is None:
            raise ValueError("explicit forecast issue_time cannot be null")
        issue_time = coerce_datetime(row["issue_time"])
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
    start_day = delivery_day(ordered[0].delivery_time)
    end_day = start_day + timedelta(days=config.horizon_days)
    return [forecast for forecast in ordered if delivery_day(forecast.delivery_time) < end_day]


def _commit_first_day(result: DispatchResult) -> list[_Commitment]:
    first_day = min(delivery_day(interval.delivery_time) for interval in result.intervals)
    return [
        _Commitment(interval, result.solver_status)
        for interval in result.intervals
        if delivery_day(interval.delivery_time) == first_day
    ]


def _flat_commitments(commitments: list[_Commitment], initial_soc_mwh: float) -> list[_Commitment]:
    return [
        _Commitment(
            DispatchInterval(
                issue_time=item.interval.issue_time,
                delivery_time=item.interval.delivery_time,
                duration_hours=item.interval.duration_hours,
                charge_mw=0.0,
                discharge_mw=0.0,
                soc_mwh=initial_soc_mwh,
                energy_revenue_eur=0.0,
                degradation_cost_eur=0.0,
                terminal_value_eur=0.0,
                objective_eur=0.0,
            ),
            "risk_blocked",
        )
        for item in commitments
    ]


def _flat_reserve_forecast(forecast: ReserveForecast, config: DispatchConfig) -> ReserveInterval:
    return ReserveInterval(
        product=forecast.product,
        direction=forecast.direction,
        issue_time=forecast.issue_time,
        delivery_time=forecast.delivery_time,
        duration_hours=forecast.duration_hours,
        forecast_value_eur_mw_h=forecast.forecast_value_eur_mw_h,
        capacity_mw=0.0,
        reserved_up_mw=0.0,
        reserved_down_mw=0.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=config.energy_capacity_mwh,
        conditional_acceptance=False,
        capacity_value_eur=0.0,
        solver_status="risk_blocked",
    )


def _zero_blocked_reserve(
    intervals: list[ReserveInterval], blocked_days: set[date], config: DispatchConfig
) -> list[ReserveInterval]:
    return [
        replace(
            interval,
            capacity_mw=0.0,
            reserved_up_mw=0.0,
            reserved_down_mw=0.0,
            minimum_soc_mwh=0.0,
            maximum_soc_mwh=config.energy_capacity_mwh,
            conditional_acceptance=False,
            capacity_value_eur=0.0,
            solver_status="risk_blocked",
        )
        if delivery_day(interval.delivery_time) in blocked_days
        else interval
        for interval in intervals
    ]


def _rolling_events(
    energy: list[DispatchForecast], balancing: list[ReserveForecast]
) -> list[tuple[datetime, str, list[Any]]]:
    """Merge energy and balancing gates into one chronological issue-time sequence."""
    events: list[tuple[datetime, str, list[Any]]] = []
    energy_ordered = sorted(energy, key=lambda item: (item.issue_time, item.delivery_time))
    for issue_time, energy_group in groupby(energy_ordered, key=lambda item: item.issue_time):
        events.append((issue_time, "energy", list(energy_group)))
    balancing_ordered = sorted(balancing, key=lambda item: (item.issue_time, item.delivery_time))
    for issue_time, balancing_group in groupby(balancing_ordered, key=lambda item: item.issue_time):
        events.append((issue_time, "balancing", list(balancing_group)))
    events.sort(key=lambda event: (event[0], event[1] != "balancing"))
    return events


def _solve_reserve_windows(
    commitments: list[_Commitment],
    forecasts: dict[datetime, list[ReserveForecast]],
    config: DispatchConfig,
    prior: dict[datetime, ReserveHeadroom] | None = None,
) -> list[ReserveInterval]:
    intervals: list[ReserveInterval] = []
    for commitment in commitments:
        delivery_time = commitment.interval.delivery_time
        result = solve_reserve_dispatch(
            forecasts.get(delivery_time, []),
            commitment.interval,
            config,
            prior_reserve=(prior or {}).get(delivery_time),
        )
        intervals.extend(result.intervals)
    return intervals


def _balancing_forecasts(
    forecasts: dict[datetime, list[ReserveForecast]],
) -> list[ReserveForecast]:
    return [
        item for rows in forecasts.values() for item in rows if item.product in {"AFRR", "MFRR"}
    ]


def _fcr_forecasts(
    forecasts: dict[datetime, list[ReserveForecast]],
) -> dict[datetime, list[ReserveForecast]]:
    return {
        delivery: [item for item in rows if item.product in {"FCR_N", "FCR_D"}]
        for delivery, rows in forecasts.items()
    }


def _reserve_headroom(
    intervals: list[ReserveInterval], config: DispatchConfig
) -> dict[datetime, tuple[float, float, float, float]]:
    result: dict[datetime, tuple[float, float, float, float]] = {}
    ordered = sorted(intervals, key=lambda item: item.delivery_time)
    for delivery_time, grouped in groupby(ordered, key=lambda item: item.delivery_time):
        rows = list(grouped)
        result[delivery_time] = (
            max(0.0, min(config.power_limit_mw, sum(row.reserved_up_mw for row in rows))),
            max(0.0, min(config.power_limit_mw, sum(row.reserved_down_mw for row in rows))),
            max(row.minimum_soc_mwh for row in rows),
            min(row.maximum_soc_mwh for row in rows),
        )
    return result


def _reserve_headroom_objects(
    intervals: list[ReserveInterval], config: DispatchConfig
) -> dict[datetime, ReserveHeadroom]:
    return {
        delivery_time: ReserveHeadroom(up, down, minimum_soc, maximum_soc)
        for delivery_time, (up, down, minimum_soc, maximum_soc) in _reserve_headroom(
            intervals, config
        ).items()
    }


def _imbalance_window(
    commitments: list[_Commitment], start: int, config: DispatchConfig
) -> list[_Commitment]:
    remaining = commitments[start:]
    start_day = delivery_day(remaining[0].interval.delivery_time)
    end_day = start_day + timedelta(days=config.horizon_days)
    return [
        commitment
        for commitment in remaining
        if delivery_day(commitment.interval.delivery_time) < end_day
    ]


def _recourse_input(
    commitment: _Commitment,
    forecast: _ImbalanceForecast | None,
    is_decision_interval: bool,
    headroom: dict[datetime, tuple[float, float, float, float]],
    config: DispatchConfig,
) -> ImbalanceDispatchInput:
    interval = commitment.interval
    up, down, minimum_soc, maximum_soc = headroom.get(
        interval.delivery_time, (0.0, 0.0, 0.0, config.energy_capacity_mwh)
    )
    return ImbalanceDispatchInput(
        delivery_time=interval.delivery_time,
        duration_hours=interval.duration_hours,
        day_ahead_charge_mw=interval.charge_mw,
        day_ahead_discharge_mw=interval.discharge_mw,
        forecast_issue_time=forecast.issue_time if is_decision_interval and forecast else None,
        forecast_price_eur_mwh=forecast.price_eur_mwh
        if is_decision_interval and forecast
        else None,
        reserved_up_mw=up,
        reserved_down_mw=down,
        minimum_soc_mwh=minimum_soc,
        maximum_soc_mwh=maximum_soc,
    )


def _recourse_inputs(
    commitments: list[_Commitment],
    forecast: _ImbalanceForecast | None,
    headroom: dict[datetime, tuple[float, float, float, float]],
    config: DispatchConfig,
) -> list[ImbalanceDispatchInput]:
    return [
        _recourse_input(commitment, forecast, index == 0, headroom, config)
        for index, commitment in enumerate(commitments)
    ]


def _solve_imbalance_windows(
    commitments: list[_Commitment],
    forecasts: dict[datetime, _ImbalanceForecast],
    headroom: dict[datetime, tuple[float, float, float, float]],
    config: DispatchConfig,
) -> list[_ImbalanceCommitment]:
    decisions: list[_ImbalanceCommitment] = []
    initial_soc = config.initial_soc_mwh
    for index, commitment in enumerate(commitments):
        window = _imbalance_window(commitments, index, config)
        forecast = forecasts.get(commitment.interval.delivery_time)
        inputs = _recourse_inputs(window, forecast, headroom, config)
        window_config = config.model_copy(update={"initial_soc_mwh": initial_soc})
        result = solve_imbalance_dispatch(inputs, window_config)
        interval = result.intervals[0]
        decisions.append(
            _ImbalanceCommitment(interval, commitment.interval.issue_time, result.solver_status)
        )
        initial_soc = interval.soc_mwh
    return decisions


def _interval_row(commitment: _Commitment, terminal_value_eur: float) -> dict[str, object]:
    interval = commitment.interval
    objective = interval.energy_revenue_eur - interval.degradation_cost_eur + terminal_value_eur
    return {
        "issue_time": interval.issue_time,
        "delivery_time": interval.delivery_time,
        "duration_hours": interval.duration_hours,
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
        _imbalance_interval_row(commitment, final_value if index == len(commitments) - 1 else 0.0)
        for index, commitment in enumerate(commitments)
    ]


def _reserve_rows(intervals: list[ReserveInterval]) -> list[dict[str, object]]:
    return [asdict(interval) for interval in intervals]


def _write_dispatch_tables(
    conn: Any,
    energy_rows: list[dict[str, object]],
    imbalance_rows: list[dict[str, object]],
    reserve_rows: list[dict[str, object]],
) -> tuple[int, int, int]:
    energy_count = write_table(conn, "dispatch_energy", energy_rows)
    imbalance_count = write_table(conn, "dispatch_imbalance", imbalance_rows)
    reserve_count = write_table(
        conn, "dispatch_reserve", reserve_rows, columns=RESERVE_DISPATCH_COLUMNS
    )
    return energy_count, imbalance_count, reserve_count


def _persist_rows(
    config: PipelineConfig,
    energy_rows: list[dict[str, object]],
    imbalance_rows: list[dict[str, object]],
    reserve_rows: list[dict[str, object]],
) -> tuple[int, int, int]:
    conn = get_connection(config.duckdb_path)
    conn.execute("BEGIN TRANSACTION")
    try:
        counts = _write_dispatch_tables(conn, energy_rows, imbalance_rows, reserve_rows)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
        return counts
    finally:
        conn.close()


def _imbalance_result(rows: list[dict[str, Any]], row_count: int) -> ImbalanceRunResult:
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


def _reserve_result(rows: list[dict[str, Any]], row_count: int) -> ReserveRunResult:
    return ReserveRunResult(
        table="dispatch_reserve",
        row_count=row_count,
        capacity_value_eur=sum(float(row["capacity_value_eur"]) for row in rows),
    )


def _result(
    rows: list[dict[str, Any]],
    row_count: int,
    imbalance: ImbalanceRunResult,
    reserve: ReserveRunResult,
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
        reserve=reserve,
    )


def _causal_dispatch_stages(
    config: PipelineConfig,
    forecasts: list[DispatchForecast],
    *,
    energy_gate: Callable[[tuple[DispatchInterval, ...]], bool] | None = None,
) -> tuple[list[_Commitment], list[ReserveInterval], list[_ImbalanceCommitment]]:
    reserve_forecasts = _load_reserve_forecasts(config)
    balancing: list[ReserveInterval] = []
    commitments: list[_Commitment] = []
    committed_times: set[datetime] = set()
    blocked_days: set[date] = set()
    initial_soc = config.optimizer.initial_soc_mwh
    for _, stage, group in _rolling_events(forecasts, _balancing_forecasts(reserve_forecasts)):
        if stage == "balancing":
            gate_config = config.optimizer.model_copy(update={"initial_soc_mwh": initial_soc})
            balancing.extend(solve_balancing_reserve_dispatch(group, gate_config).intervals)
            continue
        candidates = [item for item in group if item.delivery_time not in committed_times]
        if not candidates:
            continue
        window = _calendar_window(candidates, config.optimizer)
        window_config = config.optimizer.model_copy(update={"initial_soc_mwh": initial_soc})
        headroom = _reserve_headroom_objects(balancing, config.optimizer)
        result = (
            solve_energy_dispatch(window, window_config, reserve_headroom=headroom)
            if headroom
            else solve_energy_dispatch(window, window_config)
        )
        new_commitments = _commit_first_day(result)
        if energy_gate is not None and energy_gate(
            tuple(item.interval for item in new_commitments)
        ):
            blocked_day = delivery_day(new_commitments[0].interval.delivery_time)
            blocked_days.add(blocked_day)
            new_commitments = _flat_commitments(new_commitments, initial_soc)
            balancing = _zero_blocked_reserve(balancing, blocked_days, config.optimizer)
        commitments.extend(new_commitments)
        committed_times.update(item.interval.delivery_time for item in new_commitments)
        initial_soc = new_commitments[-1].interval.soc_mwh

    balancing_headroom = _reserve_headroom_objects(balancing, config.optimizer)
    fcr_forecasts = _fcr_forecasts(reserve_forecasts)
    active_fcr = {
        issue_time: [item for item in items if delivery_day(item.delivery_time) not in blocked_days]
        for issue_time, items in fcr_forecasts.items()
    }
    active_fcr = {key: items for key, items in active_fcr.items() if items}
    fcr = _solve_reserve_windows(commitments, active_fcr, config.optimizer, balancing_headroom)
    fcr.extend(
        _flat_reserve_forecast(item, config.optimizer)
        for items in fcr_forecasts.values()
        for item in items
        if delivery_day(item.delivery_time) in blocked_days
    )
    reserve = balancing + fcr
    allowed_imbalance = {
        delivery_time: item
        for delivery_time, item in _load_imbalance_forecasts(config).items()
        if delivery_day(delivery_time) not in blocked_days
    }
    imbalance = _solve_imbalance_windows(
        commitments,
        allowed_imbalance,
        _reserve_headroom(reserve, config.optimizer),
        config.optimizer,
    )
    return commitments, reserve, imbalance


def run_energy_dispatch(
    config: PipelineConfig,
    *,
    energy_gate: Callable[[tuple[DispatchInterval, ...]], bool] | None = None,
) -> DispatchRunResult:
    """Run 07:00 balancing, 10:00 energy, 17:30 FCR, then T-60 recourse."""
    forecasts, explicit_issue_time = _load_promoted_forecasts(config)
    if config.optimizer.horizon_days > 1 and not explicit_issue_time:
        raise ValueError("multi-day horizons require explicit issue_time forecast vintages")
    energy, reserve, imbalance = _causal_dispatch_stages(config, forecasts, energy_gate=energy_gate)
    energy_rows = _schedule_rows(energy, config.optimizer)
    imbalance_rows = _imbalance_rows(imbalance, config.optimizer)
    reserve_rows = _reserve_rows(reserve)
    energy_count, imbalance_count, reserve_count = _persist_rows(
        config, energy_rows, imbalance_rows, reserve_rows
    )
    return _result(
        energy_rows,
        energy_count,
        _imbalance_result(imbalance_rows, imbalance_count),
        _reserve_result(reserve_rows, reserve_count),
    )


def run_flat_dispatch(
    config: PipelineConfig,
    forecasts: list[DispatchForecast],
    *,
    solver_status: str,
) -> DispatchRunResult:
    """Replace every forecast delivery with a physically flat decision."""
    commitments = [
        _Commitment(
            DispatchInterval(
                issue_time=forecast.issue_time,
                delivery_time=forecast.delivery_time,
                duration_hours=forecast.duration_hours,
                charge_mw=0.0,
                discharge_mw=0.0,
                soc_mwh=config.optimizer.initial_soc_mwh,
                energy_revenue_eur=0.0,
                degradation_cost_eur=0.0,
                terminal_value_eur=0.0,
                objective_eur=0.0,
            ),
            solver_status,
        )
        for forecast in sorted(forecasts, key=lambda item: item.delivery_time)
    ]
    imbalance = [
        _ImbalanceCommitment(
            ImbalanceInterval(
                decision_time=imbalance_forecast_issue_time(commitment.interval.delivery_time),
                forecast_issue_time=None,
                delivery_time=commitment.interval.delivery_time,
                duration_hours=commitment.interval.duration_hours,
                day_ahead_charge_mw=0.0,
                day_ahead_discharge_mw=0.0,
                reserved_up_mw=0.0,
                reserved_down_mw=0.0,
                minimum_soc_mwh=0.0,
                maximum_soc_mwh=config.optimizer.energy_capacity_mwh,
                imbalance_position_mw=0.0,
                actual_charge_mw=0.0,
                actual_discharge_mw=0.0,
                soc_mwh=config.optimizer.initial_soc_mwh,
                forecast_price_eur_mwh=None,
                forecast_value_eur=0.0,
                degradation_cost_eur=0.0,
                terminal_value_eur=0.0,
                objective_eur=0.0,
            ),
            commitment.interval.issue_time,
            solver_status,
        )
        for commitment in commitments
    ]
    energy_rows = _schedule_rows(commitments, config.optimizer) if commitments else []
    imbalance_rows = _imbalance_rows(imbalance, config.optimizer) if imbalance else []
    energy_count, imbalance_count, reserve_count = _persist_rows(
        config, energy_rows, imbalance_rows, []
    )
    return _result(
        energy_rows,
        energy_count,
        _imbalance_result(imbalance_rows, imbalance_count),
        _reserve_result([], reserve_count),
    )


__all__ = [
    "DispatchRunResult",
    "ImbalanceRunResult",
    "ReserveRunResult",
    "run_energy_dispatch",
    "run_flat_dispatch",
]
