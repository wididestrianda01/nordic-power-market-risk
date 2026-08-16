
import pytest

from nordic_power_risk.bridge.mde import (
    FRAMING_INFERENTIAL,
    FRAMING_METHODOLOGY,
    MDE_RELEVANCE_FRACTION,
    compute_mde_gate,
    decide_framing,
    minimum_detectable_effect,
    relative_mde,
)


def test_mde_scales_with_sigma_and_inverse_with_samples():
    base = minimum_detectable_effect(sigma=1.0, n_pre=1000, n_post=1000)
    assert minimum_detectable_effect(2.0, 1000, 1000) == pytest.approx(2 * base)
    # larger samples shrink the MDE
    assert minimum_detectable_effect(1.0, 10000, 10000) < base


def test_mde_requires_positive_inputs():
    with pytest.raises(ValueError):
        minimum_detectable_effect(-1.0, 10, 10)
    with pytest.raises(ValueError):
        minimum_detectable_effect(1.0, 0, 10)
    with pytest.raises(ValueError):
        minimum_detectable_effect(1.0, 10, 0)


def test_relative_mde_is_fraction_of_baseline():
    assert relative_mde(0.10, 1.0) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        relative_mde(0.10, 0.0)


def test_framing_threshold_is_pre_declared_and_inclusive():
    baseline = 1.0
    assert decide_framing(0.10, baseline) == FRAMING_INFERENTIAL
    # exactly at the threshold counts as policy-relevant (<=)
    assert decide_framing(MDE_RELEVANCE_FRACTION, baseline) == FRAMING_INFERENTIAL
    assert decide_framing(0.25, baseline) == FRAMING_METHODOLOGY


def test_compute_mde_gate_is_self_consistent():
    gate = compute_mde_gate(sigma=2.0, n_pre=5000, n_post=30000, baseline_loss=3.0)
    assert gate.mde == pytest.approx(minimum_detectable_effect(2.0, 5000, 30000))
    assert gate.relative_mde == pytest.approx(gate.mde / 3.0)
    assert gate.framing == decide_framing(gate.mde, 3.0)
    assert gate.relevance_fraction == MDE_RELEVANCE_FRACTION
    assert gate.sigma == 2.0
    assert gate.n_pre == 5000
    assert gate.n_post == 30000
