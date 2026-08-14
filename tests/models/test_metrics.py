import numpy as np
import pytest

from nordic_power_risk.models.metrics import (
    diebold_mariano_test,
    interval_coverage,
    pinball_loss,
    pit_values,
    winkler_score,
)


def test_pinball_loss_zero_for_perfect_forecast() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)


def test_pinball_loss_penalizes_underprediction_more_at_high_quantile() -> None:
    y_true = np.array([10.0])
    under = np.array([5.0])
    over = np.array([15.0])
    loss_under = pinball_loss(y_true, under, 0.9)
    loss_over = pinball_loss(y_true, over, 0.9)
    assert loss_under > loss_over


def test_interval_coverage_counts_fraction_within_bounds() -> None:
    y = np.array([1.0, 5.0, 10.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([2.0, 6.0, 2.0])
    assert interval_coverage(y, lower, upper) == pytest.approx(2 / 3)


def test_winkler_score_zero_width_penalty_when_inside_interval() -> None:
    y = np.array([1.0])
    lower = np.array([0.0])
    upper = np.array([2.0])
    assert winkler_score(y, lower, upper, alpha=0.2) == pytest.approx(2.0)


def test_pit_values_recovers_known_quantile_level() -> None:
    quantile_preds = {0.1: np.array([1.0]), 0.5: np.array([5.0]), 0.9: np.array([9.0])}
    pit = pit_values(np.array([5.0]), quantile_preds)
    assert pit[0] == pytest.approx(0.5)


def test_dm_test_favors_lower_loss_challenger() -> None:
    rng = np.random.default_rng(0)
    baseline_loss = rng.normal(10.0, 1.0, 200)
    challenger_loss = baseline_loss - 2.0  # consistently better
    dm_stat, p_value = diebold_mariano_test(challenger_loss, baseline_loss)
    assert dm_stat < 0
    assert p_value < 0.05


def test_dm_test_not_significant_for_identical_loss_series() -> None:
    rng = np.random.default_rng(1)
    loss = rng.normal(10.0, 1.0, 200)
    dm_stat, p_value = diebold_mariano_test(loss, loss.copy())
    assert dm_stat == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
