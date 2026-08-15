"""Pure daily loss, tail-risk, and cooldown controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite

import numpy as np


def _finite_values(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def empirical_var_cvar(losses: Sequence[float], confidence: float) -> tuple[float, float]:
    """Return empirical upper-tail VaR and CVaR at ``confidence``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    values = _finite_values(losses, "losses")
    value_at_risk = float(np.quantile(values, confidence, method="higher"))
    tail = values[values >= value_at_risk]
    return value_at_risk, float(np.mean(tail))


def scenario_losses(
    net_discharge_mw: Sequence[float],
    duration_hours: Sequence[float],
    degradation_cost_eur: float,
    price_paths: Sequence[Sequence[float]],
) -> list[float]:
    """Reprice one fixed schedule across price paths; positive values are losses."""
    position = _finite_values(net_discharge_mw, "net_discharge_mw")
    durations = _finite_values(duration_hours, "duration_hours")
    if position.shape != durations.shape:
        raise ValueError("position and duration lengths must match")
    if np.any(durations <= 0.0):
        raise ValueError("duration_hours must be positive")
    if not isfinite(degradation_cost_eur) or degradation_cost_eur < 0.0:
        raise ValueError("degradation_cost_eur must be finite and non-negative")

    losses: list[float] = []
    for path in price_paths:
        prices = _finite_values(path, "price path")
        if prices.shape != position.shape:
            raise ValueError("price-path length must match the schedule")
        profit = float(np.sum(prices * position * durations)) - degradation_cost_eur
        losses.append(-profit)
    if not losses:
        raise ValueError("at least one price path is required")
    return losses


def historical_loss_limit(
    net_discharge_mw: Sequence[float],
    duration_hours: Sequence[float],
    degradation_cost_eur: float,
    training_price_paths: Sequence[Sequence[float]],
) -> float:
    """Calibrate the 99% daily-loss limit by historical schedule repricing."""
    losses = scenario_losses(
        net_discharge_mw,
        duration_hours,
        degradation_cost_eur,
        training_price_paths,
    )
    return float(np.quantile(losses, 0.99, method="higher"))


@dataclass
class RiskState:
    cumulative_pnl_eur: float = 0.0
    peak_pnl_eur: float = 0.0
    drawdown_eur: float = 0.0
    last_realized_loss_eur: float | None = None
    cooldown_until: date | None = None

    def start_cooldown(self, observed_on: date) -> None:
        candidate = observed_on + timedelta(days=3)
        if self.cooldown_until is None or candidate > self.cooldown_until:
            self.cooldown_until = candidate

    def observe(
        self,
        *,
        realized_loss_eur: float,
        loss_limit_eur: float,
        observed_on: date,
    ) -> bool:
        if not isfinite(realized_loss_eur) or not isfinite(loss_limit_eur):
            raise ValueError("risk observations must be finite")
        if loss_limit_eur < 0.0:
            raise ValueError("loss_limit_eur must be non-negative")
        self.last_realized_loss_eur = realized_loss_eur
        self.cumulative_pnl_eur -= realized_loss_eur
        self.peak_pnl_eur = max(self.peak_pnl_eur, self.cumulative_pnl_eur)
        self.drawdown_eur = self.peak_pnl_eur - self.cumulative_pnl_eur
        breached = realized_loss_eur > loss_limit_eur or self.drawdown_eur > loss_limit_eur
        if breached:
            self.start_cooldown(observed_on)
        return breached

    def gate_reason(self, delivery_day: date, loss_limit_eur: float) -> str | None:
        if self.cooldown_until is not None and delivery_day < self.cooldown_until:
            return "cooldown"
        if self.drawdown_eur > loss_limit_eur:
            return "drawdown_limit"
        return None


__all__ = [
    "RiskState",
    "empirical_var_cvar",
    "historical_loss_limit",
    "scenario_losses",
]
