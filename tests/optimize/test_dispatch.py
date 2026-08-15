from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pyomo.opt import SolverStatus, TerminationCondition

from nordic_power_risk.config import DispatchConfig
from nordic_power_risk.optimize import dispatch
from nordic_power_risk.optimize.dispatch import DispatchForecast, solve_energy_dispatch

ISSUE_TIME = datetime(2025, 1, 1, 9)


def _forecast(hour: int, price: float) -> DispatchForecast:
    return DispatchForecast(
        issue_time=ISSUE_TIME,
        delivery_time=datetime(2025, 1, 2) + timedelta(hours=hour),
        price_eur_mwh=price,
        duration_hours=1.0,
    )


def test_soc_is_conserved_across_charge_and_discharge() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    result = solve_energy_dispatch([_forecast(0, -100.0), _forecast(1, 200.0)], config)

    previous_soc = config.initial_soc_mwh
    for interval in result.intervals:
        expected_soc = (
            previous_soc
            + config.one_way_efficiency * interval.charge_mw * interval.duration_hours
            - interval.discharge_mw * interval.duration_hours / config.one_way_efficiency
        )
        assert interval.soc_mwh == pytest.approx(expected_soc)
        assert 0.0 <= interval.soc_mwh <= config.energy_capacity_mwh
        previous_soc = interval.soc_mwh


def test_terminal_value_preserves_stored_energy() -> None:
    no_value = solve_energy_dispatch(
        [_forecast(0, 0.0)],
        DispatchConfig(initial_soc_mwh=0.0, terminal_value_eur_mwh=0.0),
    )
    with_value = solve_energy_dispatch(
        [_forecast(0, 0.0)],
        DispatchConfig(initial_soc_mwh=0.0, terminal_value_eur_mwh=100.0),
    )

    assert no_value.intervals[-1].soc_mwh == pytest.approx(0.0)
    assert with_value.intervals[-1].soc_mwh > no_value.intervals[-1].soc_mwh
    assert with_value.terminal_value_eur == pytest.approx(100.0 * with_value.intervals[-1].soc_mwh)


def test_negative_price_never_charges_and_discharges_simultaneously() -> None:
    result = solve_energy_dispatch(
        [_forecast(0, -500.0)],
        DispatchConfig(initial_soc_mwh=2.0, terminal_value_eur_mwh=100.0),
    )

    interval = result.intervals[0]
    assert interval.charge_mw * interval.discharge_mw == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("degradation_cost", [15.0, 40.0])
def test_throughput_degradation_supports_frozen_study_bounds(
    degradation_cost: float,
) -> None:
    config = DispatchConfig(
        initial_soc_mwh=1.0,
        terminal_value_eur_mwh=0.0,
        degradation_cost_eur_mwh=degradation_cost,
    )
    result = solve_energy_dispatch([_forecast(0, -100.0), _forecast(1, 200.0)], config)
    throughput = sum(
        (interval.charge_mw + interval.discharge_mw) * interval.duration_hours
        for interval in result.intervals
    )

    assert result.degradation_cost_eur == pytest.approx(degradation_cost * throughput)


def test_fixed_asset_constants_bind_strong_price_dispatch() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    result = solve_energy_dispatch([_forecast(0, -1_000.0), _forecast(1, 1_000.0)], config)

    assert config.power_limit_mw == 1.0
    assert config.energy_capacity_mwh == 2.0
    assert config.one_way_efficiency == 0.9487
    assert result.intervals[0].charge_mw == pytest.approx(1.0)
    assert result.intervals[1].discharge_mw == pytest.approx(1.0)
    assert result.intervals[0].soc_mwh == pytest.approx(1.0 + 0.9487)
    assert result.intervals[1].soc_mwh == pytest.approx(1.0 + 0.9487 - 1.0 / 0.9487)


def test_mixed_durations_drive_soc_revenue_degradation_and_objective() -> None:
    forecasts = [
        DispatchForecast(ISSUE_TIME, datetime(2025, 1, 2), -1_000.0, 0.5),
        DispatchForecast(ISSUE_TIME, datetime(2025, 1, 2, 0, 30), 1_000.0, 1.0),
    ]
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=25.0)

    result = solve_energy_dispatch(forecasts, config)

    expected_revenue = sum(
        forecast.price_eur_mwh
        * (interval.discharge_mw - interval.charge_mw)
        * forecast.duration_hours
        for forecast, interval in zip(forecasts, result.intervals, strict=True)
    )
    throughput = sum(
        (interval.charge_mw + interval.discharge_mw) * interval.duration_hours
        for interval in result.intervals
    )
    assert result.energy_revenue_eur == pytest.approx(expected_revenue)
    assert result.degradation_cost_eur == pytest.approx(15.0 * throughput)
    assert result.objective_eur == pytest.approx(
        result.energy_revenue_eur - result.degradation_cost_eur + result.terminal_value_eur
    )


def test_horizon_is_stockholm_delivery_calendar_days() -> None:
    forecasts = [
        DispatchForecast(ISSUE_TIME, datetime(2025, 1, 2, 22, 30), 0.0, 0.5),
        DispatchForecast(ISSUE_TIME, datetime(2025, 1, 2, 23), 0.0, 0.5),
    ]

    with pytest.raises(ValueError, match="calendar day"):
        solve_energy_dispatch(forecasts, DispatchConfig(horizon_days=1))


def test_degradation_coefficient_changes_objective_by_throughput_cost() -> None:
    forecasts = [_forecast(0, -1_000.0), _forecast(1, 1_000.0)]
    low = solve_energy_dispatch(
        forecasts, DispatchConfig(terminal_value_eur_mwh=0.0, degradation_cost_eur_mwh=15.0)
    )
    high = solve_energy_dispatch(
        forecasts, DispatchConfig(terminal_value_eur_mwh=0.0, degradation_cost_eur_mwh=40.0)
    )
    throughput = sum(
        (interval.charge_mw + interval.discharge_mw) * interval.duration_hours
        for interval in low.intervals
    )

    assert high.objective_eur == pytest.approx(low.objective_eur - 25.0 * throughput)


def test_nonoptimal_solver_result_raises(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class NonoptimalSolver:
        def solve(self, model):  # type: ignore[no-untyped-def]
            solver = SimpleNamespace(
                status=SolverStatus.warning,
                termination_condition=TerminationCondition.infeasible,
            )
            return SimpleNamespace(solver=solver)

    monkeypatch.setattr(dispatch.pyo, "SolverFactory", lambda _: NonoptimalSolver())

    with pytest.raises(RuntimeError, match="termination=infeasible"):
        solve_energy_dispatch([_forecast(0, 10.0)], DispatchConfig())


@pytest.mark.parametrize("duration_hours", [float("nan"), float("inf")])
def test_duration_must_be_finite(duration_hours: float) -> None:
    forecast = DispatchForecast(ISSUE_TIME, datetime(2025, 1, 2), 10.0, duration_hours)

    with pytest.raises(ValueError, match="finite and positive"):
        solve_energy_dispatch([forecast], DispatchConfig())


def _imbalance_input(
    *,
    day_ahead_charge_mw: float,
    day_ahead_discharge_mw: float,
    forecast_price_eur_mwh: float | None,
    issue_offset: timedelta | None = timedelta(),
) -> object:
    delivery_time = datetime(2025, 1, 2)
    issue_time = (
        None if issue_offset is None else delivery_time - timedelta(minutes=60) + issue_offset
    )
    return dispatch.ImbalanceDispatchInput(
        delivery_time=delivery_time,
        duration_hours=1.0,
        day_ahead_charge_mw=day_ahead_charge_mw,
        day_ahead_discharge_mw=day_ahead_discharge_mw,
        forecast_issue_time=issue_time,
        forecast_price_eur_mwh=forecast_price_eur_mwh,
    )


def test_favorable_imbalance_forecast_changes_only_actual_setpoint() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    dispatch_input = _imbalance_input(
        day_ahead_charge_mw=1.0,
        day_ahead_discharge_mw=0.0,
        forecast_price_eur_mwh=1_000.0,
    )

    result = dispatch.solve_imbalance_dispatch([dispatch_input], config)
    interval = result.intervals[0]

    assert interval.day_ahead_charge_mw == pytest.approx(1.0)
    assert interval.day_ahead_discharge_mw == pytest.approx(0.0)
    assert interval.actual_charge_mw * interval.actual_discharge_mw == pytest.approx(0.0, abs=1e-8)
    assert interval.actual_charge_mw <= config.power_limit_mw
    assert interval.actual_discharge_mw <= config.power_limit_mw
    assert interval.soc_mwh == pytest.approx(
        config.initial_soc_mwh
        + config.one_way_efficiency * interval.actual_charge_mw
        - interval.actual_discharge_mw / config.one_way_efficiency,
        abs=1e-8,
    )
    assert interval.imbalance_position_mw == pytest.approx(
        interval.actual_discharge_mw
        - interval.actual_charge_mw
        - (interval.day_ahead_discharge_mw - interval.day_ahead_charge_mw)
    )
    assert interval.imbalance_position_mw > config.power_limit_mw


@pytest.mark.parametrize(
    "issue_offset",
    [None, -timedelta(minutes=1), timedelta(minutes=1)],
    ids=["missing", "stale", "late"],
)
def test_unavailable_imbalance_forecast_keeps_only_imbalance_leg_flat(
    issue_offset: timedelta | None,
) -> None:
    dispatch_input = _imbalance_input(
        day_ahead_charge_mw=0.5,
        day_ahead_discharge_mw=0.0,
        forecast_price_eur_mwh=None if issue_offset is None else 10_000.0,
        issue_offset=issue_offset,
    )

    interval = dispatch.solve_imbalance_dispatch(
        [dispatch_input],
        DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.imbalance_position_mw == pytest.approx(0.0)
    assert interval.actual_charge_mw == pytest.approx(0.5)
    assert interval.actual_discharge_mw == pytest.approx(0.0)


def _energy_interval(
    *,
    charge_mw: float = 0.0,
    discharge_mw: float = 0.0,
    soc_mwh: float = 1.0,
    duration_hours: float = 1.0,
    delivery_time: datetime = datetime(2025, 1, 2),
) -> dispatch.DispatchInterval:
    return dispatch.DispatchInterval(
        issue_time=ISSUE_TIME,
        delivery_time=delivery_time,
        duration_hours=duration_hours,
        charge_mw=charge_mw,
        discharge_mw=discharge_mw,
        soc_mwh=soc_mwh,
        energy_revenue_eur=0.0,
        degradation_cost_eur=0.0,
        terminal_value_eur=0.0,
        objective_eur=0.0,
    )


def _reserve_forecast(
    product: str,
    direction: str,
    price: float = 100.0,
    issue_offset: timedelta = timedelta(),
    duration_hours: float = 1.0,
) -> object:
    delivery_time = datetime(2025, 1, 2)
    exact_issue = datetime(2025, 1, 1, 16, 30)
    return dispatch.ReserveForecast(
        product=product,
        direction=direction,
        issue_time=exact_issue + issue_offset,
        delivery_time=delivery_time,
        forecast_value_eur_mw_h=price,
        duration_hours=duration_hours,
    )


def test_fcr_n_is_symmetric_and_reserves_34_percent_margin() -> None:
    result = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_N", "symmetric")],
        _energy_interval(),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    )

    interval = result.intervals[0]
    assert interval.capacity_mw == pytest.approx(1.0 / 1.34)
    assert interval.reserved_up_mw == pytest.approx(1.0)
    assert interval.reserved_down_mw == pytest.approx(1.0)
    assert interval.conditional_acceptance is True


def test_fcr_d_up_endurance_binds_low_soc() -> None:
    config = DispatchConfig(initial_soc_mwh=0.2, terminal_value_eur_mwh=0.0)
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up")],
        _energy_interval(soc_mwh=0.2),
        config,
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.2 * config.one_way_efficiency * 3.0)
    assert interval.minimum_soc_mwh == pytest.approx(
        interval.capacity_mw / (3.0 * config.one_way_efficiency)
    )


def test_fcr_d_down_endurance_binds_high_soc() -> None:
    config = DispatchConfig(initial_soc_mwh=1.9, terminal_value_eur_mwh=0.0)
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "down")],
        _energy_interval(soc_mwh=1.9),
        config,
    ).intervals[0]

    expected = 0.1 * 3.0 / config.one_way_efficiency
    assert interval.capacity_mw == pytest.approx(expected)
    assert interval.maximum_soc_mwh == pytest.approx(
        config.energy_capacity_mwh - config.one_way_efficiency * interval.capacity_mw / 3.0
    )


def test_fcr_d_opposite_direction_margin_uses_remaining_power() -> None:
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up")],
        _energy_interval(charge_mw=0.9, soc_mwh=1.1),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.5)
    assert interval.reserved_down_mw == pytest.approx(0.1)


def test_day_ahead_energy_competes_with_reserve_power() -> None:
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up")],
        _energy_interval(discharge_mw=0.8),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.2)


def test_reserve_commitments_are_conservatively_exclusive() -> None:
    forecasts = [
        _reserve_forecast("FCR_N", "symmetric", 90.0),
        _reserve_forecast("FCR_D", "up", 100.0),
        _reserve_forecast("FCR_D", "down", 80.0),
    ]
    result = dispatch.solve_reserve_dispatch(
        forecasts, _energy_interval(), DispatchConfig(terminal_value_eur_mwh=0.0)
    )

    awarded = [interval for interval in result.intervals if interval.capacity_mw > 1e-8]
    assert [(interval.product, interval.direction) for interval in awarded] == [("FCR_D", "up")]
    assert result.capacity_value_eur == pytest.approx(
        sum(
            interval.forecast_value_eur_mw_h * interval.capacity_mw * interval.duration_hours
            for interval in result.intervals
        )
    )


def test_only_exact_fcr_gate_forecast_is_eligible() -> None:
    result = dispatch.solve_reserve_dispatch(
        [
            _reserve_forecast("FCR_D", "up", 10.0),
            _reserve_forecast("FCR_D", "up", 10_000.0, -timedelta(minutes=1)),
            _reserve_forecast("FCR_D", "up", 10_000.0, timedelta(minutes=1)),
        ],
        _energy_interval(),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].forecast_value_eur_mw_h == pytest.approx(10.0)


def test_imbalance_recourse_preserves_awarded_reserve_headroom() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    minimum_soc = config.initial_soc_mwh - 0.2 / config.one_way_efficiency
    dispatch_input = dispatch.ImbalanceDispatchInput(
        delivery_time=datetime(2025, 1, 2),
        duration_hours=1.0,
        day_ahead_charge_mw=0.0,
        day_ahead_discharge_mw=0.0,
        forecast_issue_time=datetime(2025, 1, 1, 23),
        forecast_price_eur_mwh=1_000.0,
        reserved_up_mw=0.8,
        reserved_down_mw=0.2,
        minimum_soc_mwh=minimum_soc,
        maximum_soc_mwh=config.energy_capacity_mwh,
    )

    interval = dispatch.solve_imbalance_dispatch([dispatch_input], config).intervals[0]

    assert interval.actual_discharge_mw == pytest.approx(0.2)
    assert interval.actual_charge_mw == pytest.approx(0.0)
    assert interval.soc_mwh == pytest.approx(minimum_soc)


def test_reserve_optimizer_public_api_is_exported() -> None:
    from nordic_power_risk.optimize import (
        ReserveForecast,
        ReserveHeadroom,
        ReserveInterval,
        ReserveResult,
        ReserveRunResult,
        solve_balancing_reserve_dispatch,
        solve_reserve_dispatch,
    )

    assert ReserveForecast is dispatch.ReserveForecast
    assert ReserveHeadroom is dispatch.ReserveHeadroom
    assert ReserveInterval is dispatch.ReserveInterval
    assert ReserveResult is dispatch.ReserveResult
    assert ReserveRunResult.__name__ == "ReserveRunResult"
    assert solve_reserve_dispatch is dispatch.solve_reserve_dispatch
    assert solve_balancing_reserve_dispatch is dispatch.solve_balancing_reserve_dispatch


def test_fcr_mechanics_use_named_constants() -> None:
    assert dispatch.FCR_N_POWER_HEADROOM_FACTOR == pytest.approx(1.34)
    assert dispatch.FCR_D_OPPOSITE_POWER_FACTOR == pytest.approx(0.2)
    assert dispatch.FCR_D_FULL_ACTIVATION_HOURS == pytest.approx(1.0 / 3.0)
    assert dispatch.FCR_N_FULL_ACTIVATION_HOURS == pytest.approx(1.0)


def test_fcr_d_up_endurance_uses_start_and_end_soc() -> None:
    config = DispatchConfig(terminal_value_eur_mwh=0.0)
    start_soc = 0.05
    charge_mw = 0.5
    end_soc = start_soc + config.one_way_efficiency * charge_mw
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up")],
        _energy_interval(charge_mw=charge_mw, soc_mwh=end_soc),
        config,
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(
        start_soc * config.one_way_efficiency / dispatch.FCR_D_FULL_ACTIVATION_HOURS
    )


def test_fcr_d_down_endurance_uses_start_and_end_soc() -> None:
    config = DispatchConfig(terminal_value_eur_mwh=0.0)
    start_soc = 1.95
    discharge_mw = 0.5
    end_soc = start_soc - discharge_mw / config.one_way_efficiency
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "down")],
        _energy_interval(discharge_mw=discharge_mw, soc_mwh=end_soc),
        config,
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(
        (config.energy_capacity_mwh - start_soc)
        / (config.one_way_efficiency * dispatch.FCR_D_FULL_ACTIVATION_HOURS)
    )


def test_fcr_d_up_end_soc_binds_while_discharging() -> None:
    config = DispatchConfig(terminal_value_eur_mwh=0.0)
    end_soc = 0.05
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up")],
        _energy_interval(discharge_mw=0.2, soc_mwh=end_soc),
        config,
    ).intervals[0]

    expected = end_soc * config.one_way_efficiency / dispatch.FCR_D_FULL_ACTIVATION_HOURS
    assert interval.capacity_mw == pytest.approx(expected)
    assert interval.capacity_mw < config.power_limit_mw - 0.2


def test_fcr_d_down_end_soc_binds_while_charging() -> None:
    config = DispatchConfig(terminal_value_eur_mwh=0.0)
    end_soc = 1.95
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "down")],
        _energy_interval(charge_mw=0.2, soc_mwh=end_soc),
        config,
    ).intervals[0]

    expected = (config.energy_capacity_mwh - end_soc) / (
        config.one_way_efficiency * dispatch.FCR_D_FULL_ACTIVATION_HOURS
    )
    assert interval.capacity_mw == pytest.approx(expected)
    assert interval.capacity_mw < config.power_limit_mw - 0.2


@pytest.mark.parametrize("soc_mwh", [0.0, 2.0], ids=["empty", "full"])
def test_fcr_n_requires_one_hour_energy_in_both_directions(soc_mwh: float) -> None:
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_N", "symmetric")],
        _energy_interval(soc_mwh=soc_mwh),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.0)
    assert interval.conditional_acceptance is False


@pytest.mark.parametrize(
    "charge_mw,discharge_mw,start_soc,end_soc,boundary",
    [
        (0.2, 0.0, 0.05, 0.05 + 0.9487 * 0.2, "start-min"),
        (0.0, 0.2, 0.05 + 0.2 / 0.9487, 0.05, "end-min"),
        (0.0, 0.2, 1.95, 1.95 - 0.2 / 0.9487, "start-max"),
        (0.2, 0.0, 1.95 - 0.9487 * 0.2, 1.95, "end-max"),
    ],
)
def test_fcr_n_one_hour_energy_and_power_with_nonzero_day_ahead(
    charge_mw: float,
    discharge_mw: float,
    start_soc: float,
    end_soc: float,
    boundary: str,
) -> None:
    config = DispatchConfig(terminal_value_eur_mwh=0.0)
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_N", "symmetric")],
        _energy_interval(charge_mw=charge_mw, discharge_mw=discharge_mw, soc_mwh=end_soc),
        config,
    ).intervals[0]

    binding_soc = start_soc if boundary.startswith("start") else end_soc
    expected = (
        binding_soc * config.one_way_efficiency
        if boundary.endswith("min")
        else (config.energy_capacity_mwh - binding_soc) / config.one_way_efficiency
    )
    power_limit = (config.power_limit_mw - 0.2) / dispatch.FCR_N_POWER_HEADROOM_FACTOR
    assert interval.capacity_mw == pytest.approx(expected)
    assert interval.capacity_mw < power_limit
    assert interval.minimum_soc_mwh == pytest.approx(
        interval.capacity_mw / config.one_way_efficiency
    )
    assert interval.maximum_soc_mwh == pytest.approx(
        config.energy_capacity_mwh - config.one_way_efficiency * interval.capacity_mw
    )


def test_fcr_d_down_opposite_direction_margin_uses_remaining_power() -> None:
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "down")],
        _energy_interval(discharge_mw=0.9, soc_mwh=0.89),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.5)
    assert interval.reserved_up_mw == pytest.approx(0.1)


def test_capacity_value_uses_non_unit_delivery_duration() -> None:
    result = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up", duration_hours=0.5)],
        _energy_interval(duration_hours=0.5),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    )

    interval = result.intervals[0]
    assert interval.capacity_mw == pytest.approx(1.0)
    assert interval.capacity_value_eur == pytest.approx(50.0)
    assert sum(row.capacity_value_eur for row in result.intervals) == pytest.approx(50.0)
    assert result.capacity_value_eur == pytest.approx(50.0)
    assert result.objective_eur == pytest.approx(50.0)


def test_negative_capacity_price_has_no_conditional_award() -> None:
    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "up", price=-1.0)],
        _energy_interval(),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.0)
    assert interval.capacity_value_eur == pytest.approx(0.0)
    assert interval.conditional_acceptance is False


def test_summer_fcr_gate_is_1530_utc_and_exact() -> None:
    delivery_time = datetime(2025, 7, 2)
    exact_issue = datetime(2025, 7, 1, 15, 30)
    forecasts = [
        dispatch.ReserveForecast("FCR_D", "up", exact_issue + offset, delivery_time, price)
        for offset, price in (
            (-timedelta(minutes=1), 10_000.0),
            (timedelta(), 10.0),
            (timedelta(minutes=1), 10_000.0),
        )
    ]

    result = dispatch.solve_reserve_dispatch(
        forecasts,
        _energy_interval(delivery_time=delivery_time),
        DispatchConfig(terminal_value_eur_mwh=0.0),
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].issue_time == exact_issue
    assert result.intervals[0].forecast_value_eur_mw_h == pytest.approx(10.0)


@pytest.mark.parametrize(
    "initial_soc,minimum_soc,maximum_soc,price",
    [
        (0.2, 0.3, 2.0, 1_000.0),
        (1.8, 0.0, 1.7, -1_000.0),
    ],
    ids=["below-minimum", "above-maximum"],
)
def test_imbalance_start_soc_must_preserve_awarded_bounds(
    initial_soc: float, minimum_soc: float, maximum_soc: float, price: float
) -> None:
    dispatch_input = dispatch.ImbalanceDispatchInput(
        delivery_time=datetime(2025, 1, 2),
        duration_hours=1.0,
        day_ahead_charge_mw=0.0,
        day_ahead_discharge_mw=0.0,
        forecast_issue_time=datetime(2025, 1, 1, 23),
        forecast_price_eur_mwh=price,
        minimum_soc_mwh=minimum_soc,
        maximum_soc_mwh=maximum_soc,
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        dispatch.solve_imbalance_dispatch(
            [dispatch_input],
            DispatchConfig(initial_soc_mwh=initial_soc, terminal_value_eur_mwh=0.0),
        )


def test_negative_recourse_preserves_down_power_and_maximum_soc() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    maximum_soc = config.initial_soc_mwh + config.one_way_efficiency * 0.2
    dispatch_input = dispatch.ImbalanceDispatchInput(
        delivery_time=datetime(2025, 1, 2),
        duration_hours=1.0,
        day_ahead_charge_mw=0.0,
        day_ahead_discharge_mw=0.0,
        forecast_issue_time=datetime(2025, 1, 1, 23),
        forecast_price_eur_mwh=-1_000.0,
        reserved_up_mw=0.2,
        reserved_down_mw=0.8,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=maximum_soc,
    )

    interval = dispatch.solve_imbalance_dispatch([dispatch_input], config).intervals[0]

    assert interval.actual_charge_mw == pytest.approx(0.2)
    assert interval.actual_discharge_mw == pytest.approx(0.0)
    assert interval.soc_mwh == pytest.approx(maximum_soc)


def _balancing_forecast(
    product: str,
    direction: str,
    price: float = 100.0,
    *,
    delivery_time: datetime = datetime(2025, 1, 2),
    issue_offset: timedelta = timedelta(),
) -> dispatch.ReserveForecast:
    exact_issue = datetime(2025, 1, 1, 6)
    return dispatch.ReserveForecast(
        product=product,
        direction=direction,
        issue_time=exact_issue + issue_offset,
        delivery_time=delivery_time,
        forecast_value_eur_mw_h=price,
    )


def test_balancing_capacity_is_binary_exclusive_and_negative_price_stays_zero() -> None:
    forecasts = [
        _balancing_forecast("AFRR", "down", 90.0),
        _balancing_forecast("MFRR", "up", 100.0),
        _balancing_forecast("MFRR", "down", -1.0),
    ]

    result = dispatch.solve_balancing_reserve_dispatch(
        forecasts,
        DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0),
    )

    awarded = [row for row in result.intervals if row.capacity_mw > 1e-8]
    assert [(row.product, row.direction, row.capacity_mw) for row in awarded] == [
        ("MFRR", "up", 1.0)
    ]
    assert (
        next(
            row for row in result.intervals if row.direction == "down" and row.product == "MFRR"
        ).capacity_mw
        == 0.0
    )


@pytest.mark.parametrize(
    ("product", "direction", "hours", "initial_soc"),
    [
        ("AFRR", "up", 1.0, 1.2),
        ("AFRR", "down", 1.0, 0.8),
        ("MFRR", "up", 0.5, 0.8),
        ("MFRR", "down", 0.5, 1.0),
    ],
)
def test_balancing_endurance_and_efficiency_bind_both_soc_boundaries(
    product: str,
    direction: str,
    hours: float,
    initial_soc: float,
) -> None:
    config = DispatchConfig(initial_soc_mwh=initial_soc, terminal_value_eur_mwh=0.0)

    interval = dispatch.solve_balancing_reserve_dispatch(
        [_balancing_forecast(product, direction)], config
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(1.0)
    if direction == "up":
        assert interval.minimum_soc_mwh == pytest.approx(hours / config.one_way_efficiency)
        assert interval.maximum_soc_mwh == pytest.approx(config.energy_capacity_mwh)
    else:
        assert interval.minimum_soc_mwh == pytest.approx(0.0)
        assert interval.maximum_soc_mwh == pytest.approx(
            config.energy_capacity_mwh - config.one_way_efficiency * hours
        )


def test_balancing_gate_is_exact_and_dst_aware() -> None:
    delivery_time = datetime(2025, 7, 1)
    exact = dispatch.ReserveForecast(
        product="MFRR",
        direction="up",
        issue_time=datetime(2025, 6, 30, 5),
        delivery_time=delivery_time,
        forecast_value_eur_mw_h=10.0,
    )

    result = dispatch.solve_balancing_reserve_dispatch(
        [
            exact,
            dispatch.ReserveForecast(
                product="AFRR",
                direction="down",
                issue_time=datetime(2025, 6, 30, 4, 59),
                delivery_time=delivery_time,
                forecast_value_eur_mw_h=10_000.0,
            ),
        ],
        DispatchConfig(terminal_value_eur_mwh=0.0),
    )

    assert [(row.product, row.direction) for row in result.intervals] == [("MFRR", "up")]


def test_balancing_rejects_ffr() -> None:
    with pytest.raises(ValueError, match="unsupported balancing"):
        dispatch.solve_balancing_reserve_dispatch(
            [_balancing_forecast("FFR", "up")],
            DispatchConfig(terminal_value_eur_mwh=0.0),
        )


def test_early_balancing_award_constrains_later_day_ahead_energy() -> None:
    delivery_time = datetime(2025, 1, 2)
    config = DispatchConfig(initial_soc_mwh=1.2, terminal_value_eur_mwh=0.0)
    headroom = dispatch.ReserveHeadroom(
        reserved_up_mw=1.0,
        reserved_down_mw=0.0,
        minimum_soc_mwh=1.0 / config.one_way_efficiency,
        maximum_soc_mwh=config.energy_capacity_mwh,
    )

    interval = dispatch.solve_energy_dispatch(
        [DispatchForecast(ISSUE_TIME, delivery_time, 1_000.0)],
        config,
        reserve_headroom={delivery_time: headroom},
    ).intervals[0]

    assert interval.discharge_mw == pytest.approx(0.0)
    assert interval.soc_mwh >= headroom.minimum_soc_mwh


def test_later_fcr_is_exclusive_with_early_balancing_award() -> None:
    config = DispatchConfig(initial_soc_mwh=1.0, terminal_value_eur_mwh=0.0)
    prior = dispatch.ReserveHeadroom(
        reserved_up_mw=1.0,
        reserved_down_mw=0.0,
        minimum_soc_mwh=0.5 / config.one_way_efficiency,
        maximum_soc_mwh=config.energy_capacity_mwh,
    )

    interval = dispatch.solve_reserve_dispatch(
        [_reserve_forecast("FCR_D", "down", 10_000.0)],
        _energy_interval(),
        config,
        prior_reserve=prior,
    ).intervals[0]

    assert interval.capacity_mw == pytest.approx(0.0)
