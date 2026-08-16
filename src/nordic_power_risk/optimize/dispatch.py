"""Energy-only battery dispatch MILP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

import pyomo.environ as pyo
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import SolverStatus, TerminationCondition

from nordic_power_risk.config import DispatchConfig
from nordic_power_risk.energy import energy_value, soc_after, soc_before
from nordic_power_risk.facts.rules import (
    afrr_mfrr_capacity_issue_time,
    delivery_day,
    fcr_capacity_issue_time,
    imbalance_forecast_issue_time,
)

POWER_TOLERANCE_MW = 1e-8
FCR_N_POWER_HEADROOM_FACTOR = 1.34
FCR_D_OPPOSITE_POWER_FACTOR = 0.2
FCR_D_FULL_ACTIVATION_HOURS = 1.0 / 3.0
FCR_N_FULL_ACTIVATION_HOURS = 1.0
AFRR_FULL_ACTIVATION_HOURS = 1.0
MFRR_FULL_ACTIVATION_HOURS = 0.5
FLAT_DEVIATION_PENALTY_EUR_MWH = 1e6
RESERVE_SOC_VIOLATION_PENALTY_EUR_MWH = 100.0


@dataclass(frozen=True)
class DispatchForecast:
    issue_time: datetime
    delivery_time: datetime
    price_eur_mwh: float
    duration_hours: float = 1.0


@dataclass(frozen=True)
class ReserveHeadroom:
    reserved_up_mw: float = 0.0
    reserved_down_mw: float = 0.0
    minimum_soc_mwh: float = 0.0
    maximum_soc_mwh: float | None = None


@dataclass(frozen=True)
class DispatchInterval:
    """Solved-window values; terminal/objective fields are solver audit contributions."""

    issue_time: datetime
    delivery_time: datetime
    duration_hours: float
    charge_mw: float
    discharge_mw: float
    soc_mwh: float
    energy_revenue_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float


@dataclass(frozen=True)
class DispatchResult:
    intervals: tuple[DispatchInterval, ...]
    energy_revenue_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float
    solver_status: str


@dataclass(frozen=True)
class ReserveForecast:
    product: str
    direction: str
    issue_time: datetime
    delivery_time: datetime
    forecast_value_eur_mw_h: float
    duration_hours: float = 1.0


@dataclass(frozen=True)
class ReserveInterval:
    product: str
    direction: str
    issue_time: datetime
    delivery_time: datetime
    duration_hours: float
    forecast_value_eur_mw_h: float
    capacity_mw: float
    reserved_up_mw: float
    reserved_down_mw: float
    minimum_soc_mwh: float
    maximum_soc_mwh: float
    conditional_acceptance: bool
    capacity_value_eur: float
    solver_status: str


@dataclass(frozen=True)
class ReserveResult:
    intervals: tuple[ReserveInterval, ...]
    capacity_value_eur: float
    objective_eur: float
    solver_status: str


@dataclass(frozen=True)
class ImbalanceDispatchInput:
    delivery_time: datetime
    duration_hours: float
    day_ahead_charge_mw: float
    day_ahead_discharge_mw: float
    forecast_issue_time: datetime | None = None
    forecast_price_eur_mwh: float | None = None
    reserved_up_mw: float = 0.0
    reserved_down_mw: float = 0.0
    minimum_soc_mwh: float = 0.0
    maximum_soc_mwh: float | None = None


@dataclass(frozen=True)
class ImbalanceInterval:
    decision_time: datetime
    forecast_issue_time: datetime | None
    delivery_time: datetime
    duration_hours: float
    day_ahead_charge_mw: float
    day_ahead_discharge_mw: float
    reserved_up_mw: float
    reserved_down_mw: float
    minimum_soc_mwh: float
    maximum_soc_mwh: float
    imbalance_position_mw: float
    actual_charge_mw: float
    actual_discharge_mw: float
    soc_mwh: float
    forecast_price_eur_mwh: float | None
    forecast_value_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float


@dataclass(frozen=True)
class ImbalanceResult:
    intervals: tuple[ImbalanceInterval, ...]
    forecast_value_eur: float
    degradation_cost_eur: float
    terminal_value_eur: float
    objective_eur: float
    solver_status: str


def _validate_inputs(
    forecasts: Sequence[DispatchForecast],
    config: DispatchConfig,
    reserve_headroom: Mapping[datetime, ReserveHeadroom],
) -> None:
    if not forecasts:
        raise ValueError("at least one promoted forecast interval is required")
    start_day = min(delivery_day(forecast.delivery_time) for forecast in forecasts)
    end_day = start_day + timedelta(days=config.horizon_days)
    if any(delivery_day(forecast.delivery_time) >= end_day for forecast in forecasts):
        raise ValueError("forecast window exceeds configured Stockholm calendar day horizon")
    if len({forecast.issue_time for forecast in forecasts}) != 1:
        raise ValueError("one dispatch window requires one forecast issue_time")
    if any(not isfinite(item.duration_hours) or item.duration_hours <= 0 for item in forecasts):
        raise ValueError("forecast duration_hours must be finite and positive")
    if any(not isfinite(item.price_eur_mwh) for item in forecasts):
        raise ValueError("forecast prices must be finite")
    for headroom in reserve_headroom.values():
        _validate_headroom(headroom, config)


def _validate_headroom(headroom: ReserveHeadroom, config: DispatchConfig) -> None:
    maximum_soc = (
        config.energy_capacity_mwh if headroom.maximum_soc_mwh is None else headroom.maximum_soc_mwh
    )
    values = (
        headroom.reserved_up_mw,
        headroom.reserved_down_mw,
        headroom.minimum_soc_mwh,
        maximum_soc,
    )
    if any(not isfinite(value) for value in values):
        raise ValueError("reserve headroom values must be finite")
    if not 0 <= headroom.reserved_up_mw <= config.power_limit_mw:
        raise ValueError("reserved up power is outside physical limits")
    if not 0 <= headroom.reserved_down_mw <= config.power_limit_mw:
        raise ValueError("reserved down power is outside physical limits")
    if not 0 <= headroom.minimum_soc_mwh <= maximum_soc <= config.energy_capacity_mwh:
        raise ValueError("reserve SoC headroom is outside physical limits")


def _headroom(
    forecast: DispatchForecast,
    reserve_headroom: Mapping[datetime, ReserveHeadroom],
    config: DispatchConfig,
) -> ReserveHeadroom:
    item = reserve_headroom.get(forecast.delivery_time, ReserveHeadroom())
    if item.maximum_soc_mwh is None:
        return ReserveHeadroom(
            item.reserved_up_mw,
            item.reserved_down_mw,
            item.minimum_soc_mwh,
            config.energy_capacity_mwh,
        )
    return item


def _soc_balance(
    model: Any, index: int, forecasts: Sequence[DispatchForecast], config: DispatchConfig
) -> Any:
    previous = config.initial_soc_mwh if index == 0 else model.soc[index - 1]
    duration = forecasts[index].duration_hours
    return model.soc[index] == soc_after(
        previous,
        model.charge[index],
        model.discharge[index],
        duration,
        config.one_way_efficiency,
    )


def _build_model(
    forecasts: Sequence[DispatchForecast],
    config: DispatchConfig,
    reserve_headroom: Mapping[datetime, ReserveHeadroom],
) -> Any:
    model = pyo.ConcreteModel()
    model.intervals = pyo.RangeSet(0, len(forecasts) - 1)
    bounds = (0.0, config.power_limit_mw)
    model.charge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.discharge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.soc = pyo.Var(
        model.intervals, domain=pyo.NonNegativeReals, bounds=(0, config.energy_capacity_mwh)
    )
    model.is_charging = pyo.Var(model.intervals, domain=pyo.Binary)
    model.soc_balance = pyo.Constraint(
        model.intervals, rule=lambda m, i: _soc_balance(m, i, forecasts, config)
    )
    headroom = [_headroom(item, reserve_headroom, config) for item in forecasts]
    _add_energy_reserve_constraints(model, headroom, config)
    _add_objective(model, forecasts, config)
    return model


def _add_energy_reserve_constraints(
    model: Any, headroom: Sequence[ReserveHeadroom], config: DispatchConfig
) -> None:
    model.charge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: (
            m.charge[i] <= (config.power_limit_mw - headroom[i].reserved_down_mw) * m.is_charging[i]
        ),
    )
    model.discharge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: (
            m.discharge[i]
            <= (config.power_limit_mw - headroom[i].reserved_up_mw) * (1 - m.is_charging[i])
        ),
    )
    def start_soc(m: Any, i: int) -> Any:
        return config.initial_soc_mwh + 0 * m.soc[i] if i == 0 else m.soc[i - 1]
    model.reserve_soc_violation = pyo.Var(model.intervals, domain=pyo.NonNegativeReals)
    model.reserve_start_minimum = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: start_soc(m, i) + m.reserve_soc_violation[i]
        >= headroom[i].minimum_soc_mwh,
    )
    model.reserve_end_minimum = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.soc[i] + m.reserve_soc_violation[i] >= headroom[i].minimum_soc_mwh,
    )
    model.reserve_start_maximum = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: start_soc(m, i) - m.reserve_soc_violation[i]
        <= headroom[i].maximum_soc_mwh,
    )
    model.reserve_end_maximum = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.soc[i] - m.reserve_soc_violation[i] <= headroom[i].maximum_soc_mwh,
    )


def _add_objective(
    model: Any, forecasts: Sequence[DispatchForecast], config: DispatchConfig
) -> None:
    energy_revenue = sum(
        energy_value(
            forecast.price_eur_mwh,
            model.discharge[i] - model.charge[i],
            forecast.duration_hours,
        )
        for i, forecast in enumerate(forecasts)
    )
    throughput = sum(
        (model.charge[i] + model.discharge[i]) * forecast.duration_hours
        for i, forecast in enumerate(forecasts)
    )
    model.energy_revenue = pyo.Expression(expr=energy_revenue)
    model.degradation_cost = pyo.Expression(expr=config.degradation_cost_eur_mwh * throughput)
    model.terminal_value = pyo.Expression(
        expr=config.terminal_value_eur_mwh * model.soc[len(forecasts) - 1]
    )
    model.reserve_penalty = pyo.Expression(
        expr=RESERVE_SOC_VIOLATION_PENALTY_EUR_MWH
        * sum(model.reserve_soc_violation[i] for i in model.intervals)
    )
    objective = (
        model.energy_revenue
        - model.degradation_cost
        + model.terminal_value
        - model.reserve_penalty
    )
    model.objective = pyo.Objective(expr=objective, sense=pyo.maximize)


def _solve_model(model: Any) -> str:
    try:
        results = pyo.SolverFactory("highs").solve(model)
    except NoFeasibleSolutionError as exc:
        raise RuntimeError("HiGHS failed: infeasible") from exc
    status = results.solver.status
    termination = results.solver.termination_condition
    if status != SolverStatus.ok or termination != TerminationCondition.optimal:
        raise RuntimeError(f"HiGHS failed: status={status}, termination={termination}")
    return str(termination)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp HiGHS floating-point artifacts (tiny violations at a variable bound) into range."""
    return min(max(value, lower), upper)


def _interval_values(
    model: Any, forecast: DispatchForecast, index: int, config: DispatchConfig
) -> tuple[float, float, float, float]:
    charge = _clamp(float(pyo.value(model.charge[index])), 0.0, config.power_limit_mw)
    discharge = _clamp(float(pyo.value(model.discharge[index])), 0.0, config.power_limit_mw)
    duration = forecast.duration_hours
    energy_revenue = energy_value(forecast.price_eur_mwh, discharge - charge, duration)
    degradation_cost = config.degradation_cost_eur_mwh * (charge + discharge) * duration
    return charge, discharge, energy_revenue, degradation_cost


def _extract_interval(
    model: Any, forecast: DispatchForecast, index: int, config: DispatchConfig, is_last: bool
) -> DispatchInterval:
    charge, discharge, energy_revenue, degradation_cost = _interval_values(
        model, forecast, index, config
    )
    terminal_value = float(pyo.value(model.terminal_value)) if is_last else 0.0
    return DispatchInterval(
        issue_time=forecast.issue_time,
        delivery_time=forecast.delivery_time,
        duration_hours=forecast.duration_hours,
        charge_mw=charge,
        discharge_mw=discharge,
        soc_mwh=_clamp(float(pyo.value(model.soc[index])), 0.0, config.energy_capacity_mwh),
        energy_revenue_eur=energy_revenue,
        degradation_cost_eur=degradation_cost,
        terminal_value_eur=terminal_value,
        objective_eur=energy_revenue - degradation_cost + terminal_value,
    )


def solve_energy_dispatch(
    forecasts: Sequence[DispatchForecast],
    config: DispatchConfig,
    *,
    reserve_headroom: Mapping[datetime, ReserveHeadroom] | None = None,
) -> DispatchResult:
    """Solve one forecast window while preserving earlier reserve awards."""
    headroom = reserve_headroom or {}
    _validate_inputs(forecasts, config, headroom)
    model = _build_model(forecasts, config, headroom)
    solver_status = _solve_model(model)
    intervals = tuple(
        _extract_interval(model, forecast, index, config, index == len(forecasts) - 1)
        for index, forecast in enumerate(forecasts)
    )
    return DispatchResult(
        intervals=intervals,
        energy_revenue_eur=float(pyo.value(model.energy_revenue)),
        degradation_cost_eur=float(pyo.value(model.degradation_cost)),
        terminal_value_eur=float(pyo.value(model.terminal_value)),
        objective_eur=float(pyo.value(model.objective)),
        solver_status=solver_status,
    )


def _balancing_activation_hours(forecast: ReserveForecast) -> float:
    return AFRR_FULL_ACTIVATION_HOURS if forecast.product == "AFRR" else MFRR_FULL_ACTIVATION_HOURS


def _eligible_balancing_forecasts(
    forecasts: Sequence[ReserveForecast],
) -> list[ReserveForecast]:
    supported = {("AFRR", "up"), ("AFRR", "down"), ("MFRR", "up"), ("MFRR", "down")}
    for forecast in forecasts:
        if (forecast.product, forecast.direction) not in supported:
            raise ValueError("unsupported balancing reserve product or direction")
        if not isfinite(forecast.forecast_value_eur_mw_h):
            raise ValueError("balancing reserve forecast value must be finite")
        if not isfinite(forecast.duration_hours) or forecast.duration_hours <= 0:
            raise ValueError("balancing reserve duration_hours must be finite and positive")
    eligible = [
        item
        for item in forecasts
        if item.issue_time == afrr_mfrr_capacity_issue_time(item.delivery_time)
    ]
    keys = {(item.delivery_time, item.product, item.direction) for item in eligible}
    if len(keys) != len(eligible):
        raise ValueError(
            "eligible balancing forecasts contain duplicate delivery, product, and direction"
        )
    return sorted(eligible, key=lambda item: (item.delivery_time, item.product, item.direction))


def _balancing_delivery_rows(
    forecasts: Sequence[ReserveForecast],
) -> tuple[list[datetime], list[list[int]]]:
    deliveries = sorted({item.delivery_time for item in forecasts})
    rows = [
        [index for index, item in enumerate(forecasts) if item.delivery_time == delivery]
        for delivery in deliveries
    ]
    for indexes in rows:
        if len({forecasts[index].duration_hours for index in indexes}) != 1:
            raise ValueError("balancing forecasts for one delivery must share duration_hours")
    return deliveries, rows


def _balancing_soc_balance(
    model: Any,
    interval: int,
    rows: Sequence[Sequence[int]],
    forecasts: Sequence[ReserveForecast],
    config: DispatchConfig,
) -> Any:
    previous = config.initial_soc_mwh if interval == 0 else model.soc[interval - 1]
    duration = forecasts[rows[interval][0]].duration_hours
    return model.soc[interval] == soc_after(
        previous,
        model.charge[interval],
        model.discharge[interval],
        duration,
        config.one_way_efficiency,
    )


def _balancing_requirement(
    model: Any,
    indexes: Sequence[int],
    forecasts: Sequence[ReserveForecast],
    direction: str,
    config: DispatchConfig,
) -> Any:
    return sum(
        model.capacity[index]
        * _balancing_activation_hours(forecasts[index])
        * (1.0 / config.one_way_efficiency if direction == "up" else config.one_way_efficiency)
        for index in indexes
        if forecasts[index].direction == direction
    )


def _add_balancing_constraints(
    model: Any,
    rows: Sequence[Sequence[int]],
    forecasts: Sequence[ReserveForecast],
    config: DispatchConfig,
) -> None:
    def up(m: Any, i: int) -> Any:
        return sum(m.capacity[j] for j in rows[i] if forecasts[j].direction == "up")

    def down(m: Any, i: int) -> Any:
        return sum(m.capacity[j] for j in rows[i] if forecasts[j].direction == "down")

    def start(m: Any, i: int) -> Any:
        return config.initial_soc_mwh + 0 * m.soc[i] if i == 0 else m.soc[i - 1]

    def minimum(m: Any, i: int) -> Any:
        return _balancing_requirement(m, rows[i], forecasts, "up", config)

    def maximum(m: Any, i: int) -> Any:
        return config.energy_capacity_mwh - _balancing_requirement(
            m, rows[i], forecasts, "down", config
        )
    model.exclusive = pyo.Constraint(
        model.intervals, rule=lambda m, i: sum(m.capacity[j] for j in rows[i]) <= 1
    )
    model.charge_power = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.charge[i] + down(m, i) <= config.power_limit_mw
    )
    model.discharge_power = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.discharge[i] + up(m, i) <= config.power_limit_mw
    )
    model.charge_exclusive = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.charge[i] <= config.power_limit_mw * m.is_charging[i]
    )
    model.discharge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.discharge[i] <= config.power_limit_mw * (1 - m.is_charging[i]),
    )
    model.start_minimum_soc = pyo.Constraint(
        model.intervals, rule=lambda m, i: start(m, i) >= minimum(m, i)
    )
    model.end_minimum_soc = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.soc[i] >= minimum(m, i)
    )
    model.start_maximum_soc = pyo.Constraint(
        model.intervals, rule=lambda m, i: start(m, i) <= maximum(m, i)
    )
    model.end_maximum_soc = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.soc[i] <= maximum(m, i)
    )


def _build_balancing_model(forecasts: Sequence[ReserveForecast], config: DispatchConfig) -> Any:
    _, rows = _balancing_delivery_rows(forecasts)
    model = pyo.ConcreteModel()
    model.rows = pyo.RangeSet(0, len(forecasts) - 1)
    model.intervals = pyo.RangeSet(0, len(rows) - 1)
    model.capacity = pyo.Var(model.rows, domain=pyo.Binary)
    model.charge = pyo.Var(
        model.intervals, domain=pyo.NonNegativeReals, bounds=(0, config.power_limit_mw)
    )
    model.discharge = pyo.Var(
        model.intervals, domain=pyo.NonNegativeReals, bounds=(0, config.power_limit_mw)
    )
    model.soc = pyo.Var(
        model.intervals, domain=pyo.NonNegativeReals, bounds=(0, config.energy_capacity_mwh)
    )
    model.is_charging = pyo.Var(model.intervals, domain=pyo.Binary)
    model.soc_balance = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: _balancing_soc_balance(m, i, rows, forecasts, config),
    )
    _add_balancing_constraints(model, rows, forecasts, config)
    model.capacity_value = pyo.Expression(
        expr=sum(
            item.forecast_value_eur_mw_h * model.capacity[i] * item.duration_hours
            for i, item in enumerate(forecasts)
        )
    )
    model.objective = pyo.Objective(expr=model.capacity_value, sense=pyo.maximize)
    return model


def solve_balancing_reserve_dispatch(
    forecasts: Sequence[ReserveForecast], config: DispatchConfig
) -> ReserveResult:
    """Award exact-gate aFRR/mFRR capacity without day-ahead or later information."""
    eligible = _eligible_balancing_forecasts(forecasts)
    if not eligible:
        return ReserveResult((), 0.0, 0.0, "not_solved")
    model = _build_balancing_model(eligible, config)
    status = _solve_model(model)
    intervals = tuple(
        _reserve_interval(model, item, index, config, status) for index, item in enumerate(eligible)
    )
    value = float(pyo.value(model.capacity_value))
    return ReserveResult(intervals, value, value, status)


def _validate_reserve_forecast(forecast: ReserveForecast, energy: DispatchInterval) -> None:
    products = {("FCR_N", "symmetric"), ("FCR_D", "up"), ("FCR_D", "down")}
    if (forecast.product, forecast.direction) not in products:
        raise ValueError("unsupported FCR product or direction")
    if forecast.delivery_time != energy.delivery_time:
        raise ValueError("reserve forecast delivery_time must match fixed energy interval")
    if not isfinite(forecast.forecast_value_eur_mw_h):
        raise ValueError("reserve forecast value must be finite")
    if not isfinite(forecast.duration_hours) or forecast.duration_hours <= 0:
        raise ValueError("reserve duration_hours must be finite and positive")


def _eligible_reserve_forecasts(
    forecasts: Sequence[ReserveForecast], energy: DispatchInterval
) -> list[ReserveForecast]:
    for forecast in forecasts:
        _validate_reserve_forecast(forecast, energy)
    eligible = [
        forecast
        for forecast in forecasts
        if forecast.issue_time == fcr_capacity_issue_time(forecast.delivery_time)
    ]
    keys = {(forecast.product, forecast.direction) for forecast in eligible}
    if len(keys) != len(eligible):
        raise ValueError("eligible reserve forecasts contain duplicate product and direction")
    return eligible


def _validate_energy_headroom(energy: DispatchInterval, config: DispatchConfig) -> None:
    powers = (energy.charge_mw, energy.discharge_mw)
    if any(not isfinite(value) or not 0 <= value <= config.power_limit_mw for value in powers):
        raise ValueError("fixed day-ahead energy exceeds physical power limits")
    if all(value > POWER_TOLERANCE_MW for value in powers):
        raise ValueError("fixed day-ahead energy violates charge/discharge exclusivity")
    soc_values = (energy.soc_mwh, _energy_start_soc(energy, config))
    if any(
        not isfinite(value) or not 0 <= value <= config.energy_capacity_mwh for value in soc_values
    ):
        raise ValueError("fixed day-ahead state of charge is outside physical limits")


def _reserve_up(model: Any, index: int, forecasts: Sequence[ReserveForecast]) -> Any:
    forecast = forecasts[index]
    factor = (
        FCR_N_POWER_HEADROOM_FACTOR
        if forecast.product == "FCR_N"
        else 1.0
        if forecast.direction == "up"
        else FCR_D_OPPOSITE_POWER_FACTOR
    )
    return factor * model.capacity[index]


def _reserve_down(model: Any, index: int, forecasts: Sequence[ReserveForecast]) -> Any:
    forecast = forecasts[index]
    factor = (
        FCR_N_POWER_HEADROOM_FACTOR
        if forecast.product == "FCR_N"
        else 1.0
        if forecast.direction == "down"
        else FCR_D_OPPOSITE_POWER_FACTOR
    )
    return factor * model.capacity[index]


def _energy_start_soc(energy: DispatchInterval, config: DispatchConfig) -> float:
    return soc_before(
        energy.soc_mwh,
        energy.charge_mw,
        energy.discharge_mw,
        energy.duration_hours,
        config.one_way_efficiency,
    )


def _reserve_activation_hours(forecast: ReserveForecast) -> float:
    if forecast.product == "FCR_N":
        return FCR_N_FULL_ACTIVATION_HOURS
    if forecast.product == "FCR_D":
        return FCR_D_FULL_ACTIVATION_HOURS
    return _balancing_activation_hours(forecast)


def _requires_up_energy(forecast: ReserveForecast) -> bool:
    return forecast.product == "FCR_N" or forecast.direction == "up"


def _requires_down_energy(forecast: ReserveForecast) -> bool:
    return forecast.product == "FCR_N" or forecast.direction == "down"


def _reserve_minimum_soc(
    model: Any, index: int, forecasts: Sequence[ReserveForecast], config: DispatchConfig
) -> Any:
    hours = _reserve_activation_hours(forecasts[index])
    return model.capacity[index] * hours / config.one_way_efficiency


def _reserve_maximum_soc(
    model: Any, index: int, forecasts: Sequence[ReserveForecast], config: DispatchConfig
) -> Any:
    hours = _reserve_activation_hours(forecasts[index])
    return config.energy_capacity_mwh - config.one_way_efficiency * model.capacity[index] * hours


def _add_reserve_energy_constraints(
    model: Any,
    forecasts: Sequence[ReserveForecast],
    energy: DispatchInterval,
    config: DispatchConfig,
) -> None:
    start_soc = _energy_start_soc(energy, config)
    def minimum(m: Any, i: int) -> Any:
        return _reserve_minimum_soc(m, i, forecasts, config)

    def maximum(m: Any, i: int) -> Any:
        return _reserve_maximum_soc(m, i, forecasts, config)
    model.start_minimum_soc = pyo.Constraint(
        model.rows,
        rule=lambda m, i: (
            start_soc >= minimum(m, i) if _requires_up_energy(forecasts[i]) else pyo.Constraint.Skip
        ),
    )
    model.end_minimum_soc = pyo.Constraint(
        model.rows,
        rule=lambda m, i: (
            energy.soc_mwh >= minimum(m, i)
            if _requires_up_energy(forecasts[i])
            else pyo.Constraint.Skip
        ),
    )
    model.start_maximum_soc = pyo.Constraint(
        model.rows,
        rule=lambda m, i: (
            start_soc <= maximum(m, i)
            if _requires_down_energy(forecasts[i])
            else pyo.Constraint.Skip
        ),
    )
    model.end_maximum_soc = pyo.Constraint(
        model.rows,
        rule=lambda m, i: (
            energy.soc_mwh <= maximum(m, i)
            if _requires_down_energy(forecasts[i])
            else pyo.Constraint.Skip
        ),
    )


def _add_reserve_constraints(
    model: Any,
    forecasts: Sequence[ReserveForecast],
    energy: DispatchInterval,
    config: DispatchConfig,
    prior: ReserveHeadroom,
) -> None:
    model.selected_capacity = pyo.Constraint(
        model.rows, rule=lambda m, i: m.capacity[i] <= config.power_limit_mw * m.selected[i]
    )
    prior_award = (
        prior.reserved_up_mw > POWER_TOLERANCE_MW or prior.reserved_down_mw > POWER_TOLERANCE_MW
    )
    model.exclusive = pyo.Constraint(
        expr=sum(model.selected[i] for i in model.rows) <= (0 if prior_award else 1)
    )
    model.up_power = pyo.Constraint(
        expr=energy.discharge_mw
        + prior.reserved_up_mw
        + sum(_reserve_up(model, i, forecasts) for i in model.rows)
        <= config.power_limit_mw
    )
    model.down_power = pyo.Constraint(
        expr=energy.charge_mw
        + prior.reserved_down_mw
        + sum(_reserve_down(model, i, forecasts) for i in model.rows)
        <= config.power_limit_mw
    )
    _add_reserve_energy_constraints(model, forecasts, energy, config)


def _build_reserve_model(
    forecasts: Sequence[ReserveForecast],
    energy: DispatchInterval,
    config: DispatchConfig,
    prior: ReserveHeadroom,
) -> Any:
    model = pyo.ConcreteModel()
    model.rows = pyo.RangeSet(0, len(forecasts) - 1)
    model.capacity = pyo.Var(
        model.rows, domain=pyo.NonNegativeReals, bounds=(0.0, config.power_limit_mw)
    )
    model.selected = pyo.Var(model.rows, domain=pyo.Binary)
    _add_reserve_constraints(model, forecasts, energy, config, prior)
    model.capacity_value = pyo.Expression(
        expr=sum(
            forecast.forecast_value_eur_mw_h * model.capacity[i] * forecast.duration_hours
            for i, forecast in enumerate(forecasts)
        )
    )
    model.objective = pyo.Objective(expr=model.capacity_value, sense=pyo.maximize)
    return model


def _reserve_interval(
    model: Any, forecast: ReserveForecast, index: int, config: DispatchConfig, status: str
) -> ReserveInterval:
    capacity = float(pyo.value(model.capacity[index]))
    if forecast.product == "FCR_N":
        reserved_up = reserved_down = FCR_N_POWER_HEADROOM_FACTOR * capacity
    elif forecast.product == "FCR_D":
        reserved_up = (
            capacity if forecast.direction == "up" else FCR_D_OPPOSITE_POWER_FACTOR * capacity
        )
        reserved_down = (
            capacity if forecast.direction == "down" else FCR_D_OPPOSITE_POWER_FACTOR * capacity
        )
    else:
        reserved_up = capacity if forecast.direction == "up" else 0.0
        reserved_down = capacity if forecast.direction == "down" else 0.0
    hours = _reserve_activation_hours(forecast)
    minimum_soc = (
        capacity * hours / config.one_way_efficiency if _requires_up_energy(forecast) else 0.0
    )
    maximum_soc = (
        config.energy_capacity_mwh - config.one_way_efficiency * capacity * hours
        if _requires_down_energy(forecast)
        else config.energy_capacity_mwh
    )
    return ReserveInterval(
        product=forecast.product,
        direction=forecast.direction,
        issue_time=forecast.issue_time,
        delivery_time=forecast.delivery_time,
        duration_hours=forecast.duration_hours,
        forecast_value_eur_mw_h=forecast.forecast_value_eur_mw_h,
        capacity_mw=capacity,
        reserved_up_mw=reserved_up,
        reserved_down_mw=reserved_down,
        minimum_soc_mwh=minimum_soc,
        maximum_soc_mwh=maximum_soc,
        conditional_acceptance=capacity > POWER_TOLERANCE_MW,
        capacity_value_eur=forecast.forecast_value_eur_mw_h * capacity * forecast.duration_hours,
        solver_status=status,
    )


def solve_reserve_dispatch(
    forecasts: Sequence[ReserveForecast],
    energy: DispatchInterval,
    config: DispatchConfig,
    *,
    prior_reserve: ReserveHeadroom | None = None,
) -> ReserveResult:
    """Allocate later conditional FCR capacity around fixed earlier commitments."""
    prior = prior_reserve or ReserveHeadroom(maximum_soc_mwh=config.energy_capacity_mwh)
    _validate_headroom(prior, config)
    _validate_energy_headroom(energy, config)
    eligible = _eligible_reserve_forecasts(forecasts, energy)
    if not eligible:
        return ReserveResult(
            intervals=(),
            capacity_value_eur=0.0,
            objective_eur=0.0,
            solver_status="not_solved",
        )
    model = _build_reserve_model(eligible, energy, config, prior)
    status = _solve_model(model)
    intervals = tuple(
        _reserve_interval(model, forecast, index, config, status)
        for index, forecast in enumerate(eligible)
    )
    return ReserveResult(
        intervals=intervals,
        capacity_value_eur=float(pyo.value(model.capacity_value)),
        objective_eur=float(pyo.value(model.objective)),
        solver_status=status,
    )


def _eligible_imbalance_price(dispatch_input: ImbalanceDispatchInput) -> float | None:
    cutoff = imbalance_forecast_issue_time(dispatch_input.delivery_time)
    if dispatch_input.forecast_issue_time != cutoff:
        return None
    return dispatch_input.forecast_price_eur_mwh


def _maximum_soc(dispatch_input: ImbalanceDispatchInput, config: DispatchConfig) -> float:
    return (
        config.energy_capacity_mwh
        if dispatch_input.maximum_soc_mwh is None
        else dispatch_input.maximum_soc_mwh
    )


def _validate_reserved_headroom(
    dispatch_input: ImbalanceDispatchInput, config: DispatchConfig
) -> None:
    reserves = (dispatch_input.reserved_up_mw, dispatch_input.reserved_down_mw)
    if any(not isfinite(value) or not 0 <= value <= config.power_limit_mw for value in reserves):
        raise ValueError("reserved headroom exceeds physical power limits")
    maximum_soc = _maximum_soc(dispatch_input, config)
    soc_bounds = (dispatch_input.minimum_soc_mwh, maximum_soc)
    if any(
        not isfinite(value) or not 0 <= value <= config.energy_capacity_mwh for value in soc_bounds
    ):
        raise ValueError("reserved state-of-charge headroom exceeds physical limits")
    if dispatch_input.minimum_soc_mwh > maximum_soc:
        raise ValueError("reserved state-of-charge bounds are inverted")


def _validate_imbalance_input(
    dispatch_input: ImbalanceDispatchInput, config: DispatchConfig
) -> None:
    if not isfinite(dispatch_input.duration_hours) or dispatch_input.duration_hours <= 0:
        raise ValueError("imbalance duration_hours must be finite and positive")
    powers = (dispatch_input.day_ahead_charge_mw, dispatch_input.day_ahead_discharge_mw)
    if any(not isfinite(power) or not 0 <= power <= config.power_limit_mw for power in powers):
        raise ValueError("day-ahead commitment exceeds physical power limits")
    if all(power > POWER_TOLERANCE_MW for power in powers):
        raise ValueError("day-ahead commitment violates charge/discharge exclusivity")
    forecast_fields = (dispatch_input.forecast_issue_time, dispatch_input.forecast_price_eur_mwh)
    if (forecast_fields[0] is None) != (forecast_fields[1] is None):
        raise ValueError("imbalance forecast issue time and price must be provided together")
    if forecast_fields[1] is not None and not isfinite(forecast_fields[1]):
        raise ValueError("imbalance forecast price must be finite")
    _validate_reserved_headroom(dispatch_input, config)


def _validate_imbalance_inputs(
    inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    if not inputs:
        raise ValueError("at least one fixed day-ahead commitment is required")
    if len({item.delivery_time for item in inputs}) != len(inputs):
        raise ValueError("imbalance window contains duplicate delivery_time")
    start_day = min(delivery_day(item.delivery_time) for item in inputs)
    end_day = start_day + timedelta(days=config.horizon_days)
    if any(delivery_day(item.delivery_time) >= end_day for item in inputs):
        raise ValueError("imbalance window exceeds configured Stockholm calendar day horizon")
    for dispatch_input in inputs:
        _validate_imbalance_input(dispatch_input, config)


def _imbalance_soc_balance(
    model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> Any:
    previous = config.initial_soc_mwh if index == 0 else model.soc[index - 1]
    duration = inputs[index].duration_hours
    return model.soc[index] == soc_after(
        previous,
        model.charge[index],
        model.discharge[index],
        duration,
        config.one_way_efficiency,
    )


def _imbalance_start_soc(model: Any, index: int, config: DispatchConfig) -> Any:
    return config.initial_soc_mwh if index == 0 else model.soc[index - 1]


def _add_imbalance_reserve_soc_constraints(
    model: Any, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    model.start_soc = pyo.Expression(
        model.intervals, rule=lambda m, i: _imbalance_start_soc(m, i, config)
    )
    model.reserve_soc_violation = pyo.Var(model.intervals, domain=pyo.NonNegativeReals)
    model.start_minimum_reserve_soc = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.start_soc[i] + m.reserve_soc_violation[i]
        >= inputs[i].minimum_soc_mwh,
    )
    model.start_maximum_reserve_soc = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.start_soc[i] - m.reserve_soc_violation[i]
        <= _maximum_soc(inputs[i], config),
    )
    model.minimum_reserve_soc = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.soc[i] + m.reserve_soc_violation[i] >= inputs[i].minimum_soc_mwh,
    )
    model.maximum_reserve_soc = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.soc[i] - m.reserve_soc_violation[i] <= _maximum_soc(inputs[i], config),
    )


def _imbalance_position(model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput]) -> Any:
    commitment = inputs[index].day_ahead_discharge_mw - inputs[index].day_ahead_charge_mw
    return model.discharge[index] - model.charge[index] - commitment


def _is_flat_interval(index: int, inputs: Sequence[ImbalanceDispatchInput]) -> bool:
    return _eligible_imbalance_price(inputs[index]) is None


def _flat_lower_bound(model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput]) -> Any:
    if not _is_flat_interval(index, inputs):
        return pyo.Constraint.Skip
    return model.imbalance_position[index] + model.flat_deviation[index] >= 0.0


def _flat_upper_bound(model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput]) -> Any:
    if not _is_flat_interval(index, inputs):
        return pyo.Constraint.Skip
    return model.flat_deviation[index] - model.imbalance_position[index] >= 0.0


def _imbalance_forecast_value(model: Any, inputs: Sequence[ImbalanceDispatchInput]) -> Any:
    return sum(
        energy_value(
            _eligible_imbalance_price(item) or 0.0,
            model.imbalance_position[index],
            item.duration_hours,
        )
        for index, item in enumerate(inputs)
    )


def _imbalance_throughput(model: Any, inputs: Sequence[ImbalanceDispatchInput]) -> Any:
    return sum(
        (model.charge[index] + model.discharge[index]) * item.duration_hours
        for index, item in enumerate(inputs)
    )


def _add_imbalance_objective(
    model: Any, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    model.forecast_value = pyo.Expression(expr=_imbalance_forecast_value(model, inputs))
    model.degradation_cost = pyo.Expression(
        expr=config.degradation_cost_eur_mwh * _imbalance_throughput(model, inputs)
    )
    model.terminal_value = pyo.Expression(
        expr=config.terminal_value_eur_mwh * model.soc[len(inputs) - 1]
    )
    model.flat_penalty = pyo.Expression(
        expr=FLAT_DEVIATION_PENALTY_EUR_MWH
        * sum(model.flat_deviation[i] * inputs[i].duration_hours for i in model.intervals)
    )
    model.reserve_penalty = pyo.Expression(
        expr=RESERVE_SOC_VIOLATION_PENALTY_EUR_MWH
        * sum(model.reserve_soc_violation[i] for i in model.intervals)
    )
    model.objective = pyo.Objective(
        expr=model.forecast_value
        - model.degradation_cost
        + model.terminal_value
        - model.flat_penalty
        - model.reserve_penalty,
        sense=pyo.maximize,
    )


def _add_imbalance_constraints(
    model: Any, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    model.imbalance_position = pyo.Expression(
        model.intervals, rule=lambda m, i: _imbalance_position(m, i, inputs)
    )
    model.soc_balance = pyo.Constraint(
        model.intervals, rule=lambda m, i: _imbalance_soc_balance(m, i, inputs, config)
    )
    _add_imbalance_reserve_soc_constraints(model, inputs, config)
    model.flat_deviation = pyo.Var(model.intervals, domain=pyo.NonNegativeReals)
    model.flat_lower = pyo.Constraint(
        model.intervals, rule=lambda m, i: _flat_lower_bound(m, i, inputs)
    )
    model.flat_upper = pyo.Constraint(
        model.intervals, rule=lambda m, i: _flat_upper_bound(m, i, inputs)
    )


def _add_imbalance_power_constraints(
    model: Any, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    model.charge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: (
            m.charge[i] <= (config.power_limit_mw - inputs[i].reserved_down_mw) * m.is_charging[i]
        ),
    )
    model.discharge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: (
            m.discharge[i]
            <= (config.power_limit_mw - inputs[i].reserved_up_mw) * (1 - m.is_charging[i])
        ),
    )


def _build_imbalance_model(inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig) -> Any:
    model = pyo.ConcreteModel()
    model.intervals = pyo.RangeSet(0, len(inputs) - 1)
    bounds = (0.0, config.power_limit_mw)
    model.charge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.discharge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.soc = pyo.Var(model.intervals, bounds=(0.0, config.energy_capacity_mwh))
    model.is_charging = pyo.Var(model.intervals, domain=pyo.Binary)
    _add_imbalance_constraints(model, inputs, config)
    _add_imbalance_power_constraints(model, inputs, config)
    _add_imbalance_objective(model, inputs, config)
    return model


def _imbalance_values(
    model: Any,
    dispatch_input: ImbalanceDispatchInput,
    index: int,
    config: DispatchConfig,
    is_last: bool,
) -> tuple[float, float, float, float | None, float, float, float]:
    charge = float(pyo.value(model.charge[index]))
    discharge = float(pyo.value(model.discharge[index]))
    position = float(pyo.value(model.imbalance_position[index]))
    price = _eligible_imbalance_price(dispatch_input)
    forecast_value = energy_value(price or 0.0, position, dispatch_input.duration_hours)
    degradation = (
        config.degradation_cost_eur_mwh * (charge + discharge) * dispatch_input.duration_hours
    )
    terminal = float(pyo.value(model.terminal_value)) if is_last else 0.0
    return charge, discharge, position, price, forecast_value, degradation, terminal


def _extract_imbalance_interval(
    model: Any, item: ImbalanceDispatchInput, index: int, config: DispatchConfig, is_last: bool
) -> ImbalanceInterval:
    charge, discharge, position, price, forecast_value, degradation, terminal = _imbalance_values(
        model, item, index, config, is_last
    )
    return ImbalanceInterval(
        decision_time=imbalance_forecast_issue_time(item.delivery_time),
        forecast_issue_time=item.forecast_issue_time if price is not None else None,
        delivery_time=item.delivery_time,
        duration_hours=item.duration_hours,
        day_ahead_charge_mw=item.day_ahead_charge_mw,
        day_ahead_discharge_mw=item.day_ahead_discharge_mw,
        reserved_up_mw=item.reserved_up_mw,
        reserved_down_mw=item.reserved_down_mw,
        minimum_soc_mwh=item.minimum_soc_mwh,
        maximum_soc_mwh=_maximum_soc(item, config),
        imbalance_position_mw=position,
        actual_charge_mw=charge,
        actual_discharge_mw=discharge,
        soc_mwh=float(pyo.value(model.soc[index])),
        forecast_price_eur_mwh=price,
        forecast_value_eur=forecast_value,
        degradation_cost_eur=degradation,
        terminal_value_eur=terminal,
        objective_eur=forecast_value - degradation + terminal,
    )


def solve_imbalance_dispatch(
    inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> ImbalanceResult:
    """Optimize physical recourse without changing fixed day-ahead commitments."""
    _validate_imbalance_inputs(inputs, config)
    model = _build_imbalance_model(inputs, config)
    solver_status = _solve_model(model)
    intervals = tuple(
        _extract_imbalance_interval(model, item, index, config, index == len(inputs) - 1)
        for index, item in enumerate(inputs)
    )
    return ImbalanceResult(
        intervals=intervals,
        forecast_value_eur=float(pyo.value(model.forecast_value)),
        degradation_cost_eur=float(pyo.value(model.degradation_cost)),
        terminal_value_eur=float(pyo.value(model.terminal_value)),
        objective_eur=float(pyo.value(model.objective)),
        solver_status=solver_status,
    )


__all__ = [
    "DispatchForecast",
    "DispatchInterval",
    "DispatchResult",
    "ImbalanceDispatchInput",
    "ImbalanceInterval",
    "ImbalanceResult",
    "ReserveForecast",
    "ReserveHeadroom",
    "ReserveInterval",
    "ReserveResult",
    "solve_balancing_reserve_dispatch",
    "solve_energy_dispatch",
    "solve_imbalance_dispatch",
    "solve_reserve_dispatch",
]
