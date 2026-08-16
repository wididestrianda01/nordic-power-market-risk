"""Narrative stress re-pricing (Phase 4, ticket 09).

Re-prices the frozen observations under declared stress scenarios and reports the
P&L delta versus baseline. Outputs are scenario analysis, never observed results.
Price scenarios re-settle the persisted energy dispatch against perturbed prices;
parameter scenarios are documented approximations rather than re-optimizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.energy import energy_value
from nordic_power_risk.ingest.duckdb_io import (
    coerce_datetime,
    coerce_float,
    get_connection,
    read_table,
    write_table,
)
from nordic_power_risk.settle.run import reconcile

STRESS_COLUMNS = {"scenario": "VARCHAR", "delta_eur": "DOUBLE"}

# price_fn perturbs day-ahead prices; battery_fn perturbs the battery configuration.
_STRESS_SCENARIOS: dict[str, tuple[Any, Any]] = {
    "negative_price": (lambda p: p - 100.0, None),
    "price_spike": (lambda p: p * 3.0, None),
    "forecast_outage": (None, None),  # flat schedule -> zero P&L
    "reduced_capacity": (None, "capacity"),
    "efficiency_loss": (None, "efficiency"),
    "correlated_reserve": (None, "reserve"),
}


@dataclass(frozen=True)
class StressResult:
    table: str
    baseline_eur: float
    scenarios: dict[str, float]


def _energy_pnl_at_prices(config: PipelineConfig, prices: dict[datetime, float]) -> float:
    """Re-settle dispatch_energy against the given prices, positions fixed."""
    total = 0.0
    for interval in read_table(config, "dispatch_energy"):
        delivery = coerce_datetime(interval["delivery_time"])
        price = prices.get(delivery)
        if price is None or not isfinite(price):
            continue
        duration = coerce_float(interval.get("duration_hours"))
        charge = coerce_float(interval["charge_mw"])
        discharge = coerce_float(interval["discharge_mw"])
        degradation = coerce_float(interval.get("degradation_cost_eur"))
        total += energy_value(price, discharge - charge, duration) - degradation
    return total


def _reserve_value(config: PipelineConfig) -> float:
    """Sum of reserve capacity + activation components in the settlement."""
    components = reconcile(config).components
    return components.get("reserve_capacity", 0.0) + components.get("reserve_activation", 0.0)


def run_stresses(config: PipelineConfig) -> StressResult:
    """Compute each scenario's P&L delta versus the baseline settlement."""
    baseline = reconcile(config).total_pnl_eur

    prices = {
        coerce_datetime(row["event_time"]): coerce_float(row["price_eur_mwh"])
        for row in read_table(config, "fact_day_ahead_price")
    }

    scenarios: dict[str, float] = {}
    for name, (price_fn, battery_fn) in _STRESS_SCENARIOS.items():
        if name == "forecast_outage":
            scenarios[name] = -baseline  # flat schedule -> zero P&L
        elif price_fn is not None:
            perturbed = {t: price_fn(p) for t, p in prices.items()}
            scenarios[name] = _energy_pnl_at_prices(config, perturbed) - baseline
        elif battery_fn == "capacity":
            # Reduced capacity halves arbitrage throughput (documented approximation).
            scenarios[name] = 0.5 * baseline - baseline
        elif battery_fn == "efficiency":
            # Efficiency loss scales revenue (documented approximation).
            scenarios[name] = 0.95 * baseline - baseline
        elif battery_fn == "reserve":
            # Correlated reserve/energy doubles reserve value (documented approximation).
            scenarios[name] = _reserve_value(config)
        else:  # pragma: no cover - defensive
            scenarios[name] = 0.0

    rows = [{"scenario": name, "delta_eur": delta} for name, delta in scenarios.items()]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "stress", rows, columns=STRESS_COLUMNS)
    finally:
        conn.close()
    return StressResult(table="stress", baseline_eur=baseline, scenarios=scenarios)


__all__ = ["STRESS_COLUMNS", "StressResult", "run_stresses"]
