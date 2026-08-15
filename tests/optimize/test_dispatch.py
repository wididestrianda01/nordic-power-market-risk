from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pyomo.opt import SolverStatus, TerminationCondition

from nordic_power_risk.optimize import dispatch

from nordic_power_risk.config import DispatchConfig
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
    assert with_value.terminal_value_eur == pytest.approx(
        100.0 * with_value.intervals[-1].soc_mwh
    )


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
        None
        if issue_offset is None
        else delivery_time - timedelta(minutes=60) + issue_offset
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
    assert interval.actual_charge_mw * interval.actual_discharge_mw == pytest.approx(
        0.0, abs=1e-8
    )
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
