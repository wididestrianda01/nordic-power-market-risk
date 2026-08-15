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

_STOCKHOLM = ZoneInfo("Europe/Stockholm")


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


__all__ = [
    "DispatchForecast",
    "DispatchInterval",
    "DispatchResult",
    "solve_energy_dispatch",
]
