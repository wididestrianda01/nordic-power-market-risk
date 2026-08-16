"""Comparison policies and benchmarks (Phase 4, ticket 07).

Compares the optimized paper policy against no-trade, a simple heuristic, and a
perfect-foresight upper bound, each settled net of declared costs on the same
observed day-ahead prices. Perfect foresight is an upper bound, not a benchmark
to claim victory over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from nordic_power_risk.config import PipelineConfig
from nordic_power_risk.facts.rules import delivery_day
from nordic_power_risk.ingest.duckdb_io import coerce_datetime, get_connection, write_table
from nordic_power_risk.settle.run import reconcile

COMPARISON_COLUMNS = {"policy": "VARCHAR", "total_pnl_eur": "DOUBLE"}


@dataclass(frozen=True)
class ComparisonResult:
    table: str
    policies: dict[str, float]


def _read_day_ahead_prices(config: PipelineConfig) -> list[dict[str, Any]]:
    conn = get_connection(config.duckdb_path)
    try:
        return conn.execute("SELECT * FROM fact_day_ahead_price").fetchdf().to_dict("records")
    finally:
        conn.close()


def _daily_arbitrage(
    days: list[list[float]],
    power_mw: float,
    round_trip_eff: float,
    degradation_eur_mwh: float,
    cycles: int,
) -> float:
    """Buy at the K cheapest hours, sell at the K most expensive, per delivery day."""
    total = 0.0
    for prices in days:
        if len(prices) < 2 * cycles:
            continue
        ordered = sorted(prices)
        buy = ordered[:cycles]
        sell = ordered[-cycles:]
        revenue = power_mw * round_trip_eff * sum(sell)
        cost = power_mw * sum(buy)
        throughput_mwh = 2.0 * cycles * power_mw  # charge + discharge
        total += revenue - cost - throughput_mwh * degradation_eur_mwh
    return total


def compare_policies(config: PipelineConfig) -> ComparisonResult:
    """Settle no-trade, heuristic, optimized, and perfect-foresight policies."""
    prices = _read_day_ahead_prices(config)
    grouped: dict[date, list[float]] = {}
    for row in prices:
        event_time = coerce_datetime(row["event_time"])
        day = delivery_day(event_time)
        grouped.setdefault(day, []).append(float(row["price_eur_mwh"]))
    days = list(grouped.values())

    battery = config.optimizer
    round_trip_eff = battery.one_way_efficiency**2
    full_cycles = max(1, int(battery.energy_capacity_mwh / battery.power_limit_mw))

    # Reserve capacity and activation are revenue the optimized policy realizes
    # but the energy-arbitrage bound omits. With perfect foresight the asset
    # captures at least the realized reserve on top of the perfect spread, so
    # add it to keep the upper bound comparable to the co-optimized policy.
    recon = reconcile(config)
    reserve_revenue = recon.components.get("reserve_capacity", 0.0) + recon.components.get(
        "reserve_activation", 0.0
    )
    energy_arbitrage = _daily_arbitrage(
        days,
        battery.power_limit_mw,
        round_trip_eff,
        battery.degradation_cost_eur_mwh,
        full_cycles,
    )

    policies = {
        "no_trade": 0.0,
        "heuristic": _daily_arbitrage(
            days,
            battery.power_limit_mw,
            round_trip_eff,
            battery.degradation_cost_eur_mwh,
            1,
        ),
        "perfect_foresight": energy_arbitrage + reserve_revenue,
        "optimized": recon.total_pnl_eur,
    }

    rows = [{"policy": name, "total_pnl_eur": value} for name, value in policies.items()]
    conn = get_connection(config.duckdb_path)
    try:
        write_table(conn, "comparison", rows, columns=COMPARISON_COLUMNS)
    finally:
        conn.close()
    return ComparisonResult(table="comparison", policies=policies)


__all__ = ["COMPARISON_COLUMNS", "ComparisonResult", "compare_policies"]
