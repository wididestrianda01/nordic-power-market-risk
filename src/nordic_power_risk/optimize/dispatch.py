"""Energy-only battery dispatch MILP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from typing import Any

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from zoneinfo import ZoneInfo

from nordic_power_risk.config import DispatchConfig
from nordic_power_risk.facts.rules import imbalance_forecast_issue_time

_STOCKHOLM = ZoneInfo("Europe/Stockholm")
POWER_TOLERANCE_MW = 1e-8


@dataclass(frozen=True)
class DispatchForecast:
    issue_time: datetime
    delivery_time: datetime
    price_eur_mwh: float
    duration_hours: float = 1.0


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
class ImbalanceDispatchInput:
    delivery_time: datetime
    duration_hours: float
    day_ahead_charge_mw: float
    day_ahead_discharge_mw: float
    forecast_issue_time: datetime | None = None
    forecast_price_eur_mwh: float | None = None


@dataclass(frozen=True)
class ImbalanceInterval:
    decision_time: datetime
    forecast_issue_time: datetime | None
    delivery_time: datetime
    duration_hours: float
    day_ahead_charge_mw: float
    day_ahead_discharge_mw: float
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


def _delivery_day(delivery_time: datetime) -> date:
    return delivery_time.replace(tzinfo=timezone.utc).astimezone(_STOCKHOLM).date()


def _validate_inputs(forecasts: Sequence[DispatchForecast], config: DispatchConfig) -> None:
    if not forecasts:
        raise ValueError("at least one promoted forecast interval is required")
    start_day = min(_delivery_day(forecast.delivery_time) for forecast in forecasts)
    end_day = start_day + timedelta(days=config.horizon_days)
    if any(_delivery_day(forecast.delivery_time) >= end_day for forecast in forecasts):
        raise ValueError("forecast window exceeds configured Stockholm calendar day horizon")
    if len({forecast.issue_time for forecast in forecasts}) != 1:
        raise ValueError("one dispatch window requires one forecast issue_time")
    if any(
        not isfinite(forecast.duration_hours) or forecast.duration_hours <= 0
        for forecast in forecasts
    ):
        raise ValueError("forecast duration_hours must be finite and positive")
    if any(not isfinite(forecast.price_eur_mwh) for forecast in forecasts):
        raise ValueError("forecast prices must be finite")


def _soc_balance(model: Any, index: int, forecasts: Sequence[DispatchForecast], config: DispatchConfig) -> Any:
    previous = config.initial_soc_mwh if index == 0 else model.soc[index - 1]
    duration = forecasts[index].duration_hours
    return model.soc[index] == (
        previous
        + config.one_way_efficiency * model.charge[index] * duration
        - model.discharge[index] * duration / config.one_way_efficiency
    )


def _build_model(forecasts: Sequence[DispatchForecast], config: DispatchConfig) -> Any:
    model = pyo.ConcreteModel()
    model.intervals = pyo.RangeSet(0, len(forecasts) - 1)
    bounds = (0.0, config.power_limit_mw)
    model.charge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.discharge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.soc = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=(0, config.energy_capacity_mwh))
    model.is_charging = pyo.Var(model.intervals, domain=pyo.Binary)
    model.soc_balance = pyo.Constraint(
        model.intervals, rule=lambda m, i: _soc_balance(m, i, forecasts, config)
    )
    model.charge_exclusive = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.charge[i] <= config.power_limit_mw * m.is_charging[i]
    )
    model.discharge_exclusive = pyo.Constraint(
        model.intervals, rule=lambda m, i: m.discharge[i] <= config.power_limit_mw * (1 - m.is_charging[i])
    )
    _add_objective(model, forecasts, config)
    return model


def _add_objective(model: Any, forecasts: Sequence[DispatchForecast], config: DispatchConfig) -> None:
    energy_revenue = sum(
        forecast.price_eur_mwh * (model.discharge[i] - model.charge[i]) * forecast.duration_hours
        for i, forecast in enumerate(forecasts)
    )
    throughput = sum(
        (model.charge[i] + model.discharge[i]) * forecast.duration_hours
        for i, forecast in enumerate(forecasts)
    )
    model.energy_revenue = pyo.Expression(expr=energy_revenue)
    model.degradation_cost = pyo.Expression(
        expr=config.degradation_cost_eur_mwh * throughput
    )
    model.terminal_value = pyo.Expression(
        expr=config.terminal_value_eur_mwh * model.soc[len(forecasts) - 1]
    )
    objective = model.energy_revenue - model.degradation_cost + model.terminal_value
    model.objective = pyo.Objective(expr=objective, sense=pyo.maximize)


def _solve_model(model: Any) -> str:
    results = pyo.SolverFactory("highs").solve(model)
    status = results.solver.status
    termination = results.solver.termination_condition
    if status != SolverStatus.ok or termination != TerminationCondition.optimal:
        raise RuntimeError(f"HiGHS failed: status={status}, termination={termination}")
    return str(termination)


def _interval_values(
    model: Any, forecast: DispatchForecast, index: int, config: DispatchConfig
) -> tuple[float, float, float, float]:
    charge = float(pyo.value(model.charge[index]))
    discharge = float(pyo.value(model.discharge[index]))
    duration = forecast.duration_hours
    energy_revenue = forecast.price_eur_mwh * (discharge - charge) * duration
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
        soc_mwh=float(pyo.value(model.soc[index])),
        energy_revenue_eur=energy_revenue,
        degradation_cost_eur=degradation_cost,
        terminal_value_eur=terminal_value,
        objective_eur=energy_revenue - degradation_cost + terminal_value,
    )


def solve_energy_dispatch(
    forecasts: Sequence[DispatchForecast], config: DispatchConfig
) -> DispatchResult:
    """Solve one forecast window without realized delivery prices."""
    _validate_inputs(forecasts, config)
    model = _build_model(forecasts, config)
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


def _eligible_imbalance_price(dispatch_input: ImbalanceDispatchInput) -> float | None:
    cutoff = imbalance_forecast_issue_time(dispatch_input.delivery_time)
    if dispatch_input.forecast_issue_time != cutoff:
        return None
    return dispatch_input.forecast_price_eur_mwh


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


def _validate_imbalance_inputs(
    inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    if not inputs:
        raise ValueError("at least one fixed day-ahead commitment is required")
    if len({item.delivery_time for item in inputs}) != len(inputs):
        raise ValueError("imbalance window contains duplicate delivery_time")
    start_day = min(_delivery_day(item.delivery_time) for item in inputs)
    end_day = start_day + timedelta(days=config.horizon_days)
    if any(_delivery_day(item.delivery_time) >= end_day for item in inputs):
        raise ValueError("imbalance window exceeds configured Stockholm calendar day horizon")
    for dispatch_input in inputs:
        _validate_imbalance_input(dispatch_input, config)


def _imbalance_soc_balance(
    model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> Any:
    previous = config.initial_soc_mwh if index == 0 else model.soc[index - 1]
    duration = inputs[index].duration_hours
    return model.soc[index] == (
        previous
        + config.one_way_efficiency * model.charge[index] * duration
        - model.discharge[index] * duration / config.one_way_efficiency
    )


def _imbalance_position(
    model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput]
) -> Any:
    commitment = inputs[index].day_ahead_discharge_mw - inputs[index].day_ahead_charge_mw
    return model.discharge[index] - model.charge[index] - commitment


def _flat_imbalance(
    model: Any, index: int, inputs: Sequence[ImbalanceDispatchInput]
) -> Any:
    if _eligible_imbalance_price(inputs[index]) is not None:
        return pyo.Constraint.Skip
    return model.imbalance_position[index] == 0.0


def _imbalance_forecast_value(
    model: Any, inputs: Sequence[ImbalanceDispatchInput]
) -> Any:
    return sum(
        (_eligible_imbalance_price(item) or 0.0)
        * model.imbalance_position[index]
        * item.duration_hours
        for index, item in enumerate(inputs)
    )


def _imbalance_throughput(
    model: Any, inputs: Sequence[ImbalanceDispatchInput]
) -> Any:
    return sum(
        (model.charge[index] + model.discharge[index]) * item.duration_hours
        for index, item in enumerate(inputs)
    )


def _add_imbalance_objective(
    model: Any, inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> None:
    model.forecast_value = pyo.Expression(
        expr=_imbalance_forecast_value(model, inputs)
    )
    model.degradation_cost = pyo.Expression(
        expr=config.degradation_cost_eur_mwh * _imbalance_throughput(model, inputs)
    )
    model.terminal_value = pyo.Expression(
        expr=config.terminal_value_eur_mwh * model.soc[len(inputs) - 1]
    )
    model.objective = pyo.Objective(
        expr=model.forecast_value - model.degradation_cost + model.terminal_value,
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
    model.flat_unavailable = pyo.Constraint(
        model.intervals, rule=lambda m, i: _flat_imbalance(m, i, inputs)
    )


def _add_imbalance_power_constraints(model: Any, config: DispatchConfig) -> None:
    model.charge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.charge[i] <= config.power_limit_mw * m.is_charging[i],
    )
    model.discharge_exclusive = pyo.Constraint(
        model.intervals,
        rule=lambda m, i: m.discharge[i]
        <= config.power_limit_mw * (1 - m.is_charging[i]),
    )


def _build_imbalance_model(
    inputs: Sequence[ImbalanceDispatchInput], config: DispatchConfig
) -> Any:
    model = pyo.ConcreteModel()
    model.intervals = pyo.RangeSet(0, len(inputs) - 1)
    bounds = (0.0, config.power_limit_mw)
    model.charge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.discharge = pyo.Var(model.intervals, domain=pyo.NonNegativeReals, bounds=bounds)
    model.soc = pyo.Var(model.intervals, bounds=(0.0, config.energy_capacity_mwh))
    model.is_charging = pyo.Var(model.intervals, domain=pyo.Binary)
    _add_imbalance_constraints(model, inputs, config)
    _add_imbalance_power_constraints(model, config)
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
    forecast_value = (price or 0.0) * position * dispatch_input.duration_hours
    degradation = (
        config.degradation_cost_eur_mwh
        * (charge + discharge)
        * dispatch_input.duration_hours
    )
    terminal = float(pyo.value(model.terminal_value)) if is_last else 0.0
    return charge, discharge, position, price, forecast_value, degradation, terminal


def _extract_imbalance_interval(
    model: Any, item: ImbalanceDispatchInput, index: int, config: DispatchConfig, is_last: bool
) -> ImbalanceInterval:
    charge, discharge, position, price, forecast_value, degradation, terminal = (
        _imbalance_values(model, item, index, config, is_last)
    )
    return ImbalanceInterval(
        decision_time=imbalance_forecast_issue_time(item.delivery_time),
        forecast_issue_time=item.forecast_issue_time if price is not None else None,
        delivery_time=item.delivery_time, duration_hours=item.duration_hours,
        day_ahead_charge_mw=item.day_ahead_charge_mw,
        day_ahead_discharge_mw=item.day_ahead_discharge_mw,
        imbalance_position_mw=position,
        actual_charge_mw=charge, actual_discharge_mw=discharge,
        soc_mwh=float(pyo.value(model.soc[index])),
        forecast_price_eur_mwh=price, forecast_value_eur=forecast_value,
        degradation_cost_eur=degradation, terminal_value_eur=terminal,
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
    "solve_energy_dispatch",
    "solve_imbalance_dispatch",
]
