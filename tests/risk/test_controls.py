from datetime import date

import pytest

from nordic_power_risk.risk.controls import (
    RiskState,
    empirical_var_cvar,
    historical_loss_limit,
    scenario_losses,
)


def test_empirical_var_and_cvar_use_upper_tail() -> None:
    var, cvar = empirical_var_cvar([-10.0, 0.0, 10.0, 20.0], 0.95)

    assert var == pytest.approx(20.0)
    assert cvar == pytest.approx(20.0)


def test_scenario_losses_preserve_dispatch_sign_and_duration() -> None:
    losses = scenario_losses(
        net_discharge_mw=[-1.0, 1.0],
        duration_hours=[0.5, 1.0],
        degradation_cost_eur=5.0,
        price_paths=[[-100.0, 200.0], [100.0, -200.0]],
    )

    assert losses == pytest.approx([-245.0, 255.0])


def test_historical_limit_is_candidate_schedule_loss_percentile() -> None:
    limit = historical_loss_limit(
        net_discharge_mw=[1.0],
        duration_hours=[1.0],
        degradation_cost_eur=5.0,
        training_price_paths=[[100.0], [-20.0], [50.0]],
    )

    assert limit == pytest.approx(25.0)


def test_risk_state_enforces_three_day_cooldown_and_drawdown_recovery() -> None:
    state = RiskState()
    state.observe(realized_loss_eur=120.0, loss_limit_eur=100.0, observed_on=date(2025, 1, 1))

    assert state.drawdown_eur == pytest.approx(120.0)
    assert state.gate_reason(date(2025, 1, 1), 100.0) == "cooldown"
    assert state.gate_reason(date(2025, 1, 3), 100.0) == "cooldown"
    assert state.gate_reason(date(2025, 1, 4), 100.0) == "drawdown_limit"

    state.observe(realized_loss_eur=-30.0, loss_limit_eur=100.0, observed_on=date(2025, 1, 2))

    assert state.drawdown_eur == pytest.approx(90.0)
    assert state.gate_reason(date(2025, 1, 4), 100.0) is None


def test_candidate_breach_starts_three_blocked_delivery_days() -> None:
    state = RiskState()
    state.start_cooldown(date(2025, 1, 1))

    assert state.gate_reason(date(2025, 1, 1), 1_000.0) == "cooldown"
    assert state.gate_reason(date(2025, 1, 3), 1_000.0) == "cooldown"
    assert state.gate_reason(date(2025, 1, 4), 1_000.0) is None
