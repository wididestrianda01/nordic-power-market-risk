from nordic_power_risk.energy import energy_value, soc_after, soc_before


def test_soc_after_forward_identity() -> None:
    assert soc_after(1.0, 1.0, 0.0, 1.0, 0.9) == 1.0 + 0.9 * 1.0 * 1.0
    assert soc_after(1.0, 0.0, 1.0, 1.0, 0.9) == 1.0 - 1.0 * 1.0 / 0.9


def test_soc_before_inverts_soc_after() -> None:
    eff = 0.9487
    after = soc_after(0.5, 0.3, 0.2, 1.0, eff)
    assert soc_before(after, 0.3, 0.2, 1.0, eff) == 0.5


def test_soc_before_recovers_start_from_after() -> None:
    # _energy_start_soc / risk flat_soc convention: end soc minus the charge/discharge delta.
    assert soc_before(1.0, 0.5, 0.0, 1.0, 0.9) == 1.0 - 0.9 * 0.5 * 1.0


def test_energy_value_is_price_times_net_discharge_times_duration() -> None:
    assert energy_value(100.0, 1.5, 1.0) == 150.0
    assert energy_value(100.0, -1.0, 2.0) == -200.0
    assert energy_value(0.0, 3.0, 1.0) == 0.0
